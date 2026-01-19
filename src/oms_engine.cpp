// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]
#define ARMA_USE_CURRENT
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// ===================================================================
//           Q-FUNCTIONS (Objective Helpers)
// ===================================================================

// [[Rcpp::export]]
double Q_balance_cpp(const arma::vec& population_probs,
                     const arma::mat& population_x,
                     const arma::vec& population_x_totals,
                     const arma::uvec& panel_indices_global,
                     int n_master,
                     int k_panel) {
  
  double scaling = static_cast<double>(k_panel) / n_master;
  
  // Convert R indices (1-based) to C++ indices (0-based)
  arma::uvec cpp_indices = panel_indices_global - 1;
  arma::vec probs_for_panel = population_probs.elem(cpp_indices) * scaling;
  
  // Check for zero probabilities
  if (any(probs_for_panel == 0)) {
    return arma::datum::inf;
  }
  
  arma::mat panel_x = population_x.rows(cpp_indices);
  arma::vec ht_estimates = arma::sum(panel_x.each_col() / probs_for_panel, 0).t();
  
  arma::vec relative_errors = (ht_estimates - population_x_totals) / population_x_totals;
  arma::vec squared_errors = arma::square(relative_errors);
  
  return arma::mean(squared_errors);
}

// [[Rcpp::export]]
double Q_spread_cpp(const arma::mat& master_sample_coords,
                    const arma::uvec& panel_indices_local,
                    int n_master,
                    int k_panel) {
  
  // This function now calls R's sb() function from BalancedSampling
  Environment pkg = Environment::namespace_env("BalancedSampling");
  Function sb = pkg["sb"]; 
  
  // Create probability vector for sb
  arma::vec prob_for_sb(n_master);
  prob_for_sb.fill(static_cast<double>(k_panel) / n_master);
  
  // Convert uvec to IntegerVector for R
  IntegerVector r_indices = wrap(panel_indices_local);
  
  // Call sb function
  SEXP result = sb(prob_for_sb, master_sample_coords, r_indices);
  
  // Extract the numeric result
  double sb_score = as<double>(result);
  
  return sb_score;
}

// ===================================================================
//           CIRCULAR INDEXING HELPER
// ===================================================================

inline arma::uvec get_circular_indices(const arma::uvec& sequence, 
                                       int start_pos, 
                                       int k, 
                                       int n) {
  arma::uvec indices(k);
  for (int i = 0; i < k; i++) {
    indices(i) = sequence(((start_pos + i) % n));
  }
  return indices;
}

// ===================================================================
//           PANEL SCORE CACHE STRUCTURE
// ===================================================================

struct PanelScoreCache {
  std::map<std::pair<int, int>, double> balance_scores;  // (k, start_pos) -> score
  std::map<std::pair<int, int>, double> spread_scores;
  
  void clear() {
    balance_scores.clear();
    spread_scores.clear();
  }
};

// ===================================================================
//           COMPUTE SCORES FOR SPECIFIC PANELS
// ===================================================================

void compute_panel_score(const arma::uvec& sequence,
                         int k,
                         int start_pos,
                         int n,
                         bool compute_balance,
                         bool compute_spread,
                         const arma::uvec& initial_sample_indices_global,
                         const arma::mat& population_x,
                         const arma::vec& population_x_totals,
                         const arma::vec& population_probs,
                         const arma::mat& master_sample_coords,
                         PanelScoreCache& cache) {
  
  // Get panel indices
  arma::uvec panel_indices_local = get_circular_indices(sequence, start_pos, k, n);
  std::pair<int, int> key = std::make_pair(k, start_pos);
  
  if (compute_balance && cache.balance_scores.find(key) == cache.balance_scores.end()) {
    arma::uvec panel_indices_global(k);
    for (int i = 0; i < k; i++) {
      panel_indices_global(i) = initial_sample_indices_global(panel_indices_local(i) - 1);
    }
    cache.balance_scores[key] = Q_balance_cpp(population_probs, population_x, 
                                              population_x_totals, panel_indices_global, n, k);
  }
  
  if (compute_spread && cache.spread_scores.find(key) == cache.spread_scores.end()) {
    cache.spread_scores[key] = Q_spread_cpp(master_sample_coords, panel_indices_local, n, k);
  }
}

// ===================================================================
//           IDENTIFY AFFECTED PANELS BY A SWAP
// ===================================================================

std::set<std::pair<int, int>> get_affected_panels(int pos1, int pos2, 
                                                  const arma::uvec& critical_k, 
                                                  int n) {
  std::set<std::pair<int, int>> affected;
  
  for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
    int k = critical_k(k_idx);
    for (int pos : {pos1, pos2}) {
      for (int start_offset = 0; start_offset < k; start_offset++) {
        int start_pos = (pos - start_offset + n) % n;
        affected.insert(std::make_pair(k, start_pos));
      }
    }
  }
  return affected;
}

// ===================================================================
//           COMPUTE O-SCORES FROM CACHE (MEAN)
// ===================================================================

List compute_O_scores_from_cache(const PanelScoreCache& cache,
                                 const arma::uvec& critical_k,
                                 int n,
                                 bool compute_balance,
                                 bool compute_spread) {
  
  double O_balance = NA_REAL;
  double O_spread = NA_REAL;
  
  if (compute_balance) {
    arma::vec mean_balance_scores(critical_k.n_elem);
    for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
      int k = critical_k(k_idx);
      double sum_score = 0.0;
      int count = 0;
      
      for (int start_pos = 0; start_pos < n; start_pos++) {
        auto it = cache.balance_scores.find(std::make_pair(k, start_pos));
        if (it != cache.balance_scores.end()) {
          sum_score += it->second;
          count++;
        }
      }
      mean_balance_scores(k_idx) = (count > 0) ? (sum_score / count) : arma::datum::inf;
    }
    O_balance = arma::mean(mean_balance_scores);
  }
  
  if (compute_spread) {
    arma::vec mean_spread_scores(critical_k.n_elem);
    for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
      int k = critical_k(k_idx);
      double sum_score = 0.0;
      int count = 0;
      
      for (int start_pos = 0; start_pos < n; start_pos++) {
        auto it = cache.spread_scores.find(std::make_pair(k, start_pos));
        if (it != cache.spread_scores.end()) {
          sum_score += it->second;
          count++;
        }
      }
      mean_spread_scores(k_idx) = (count > 0) ? (sum_score / count) : arma::datum::inf;
    }
    O_spread = arma::mean(mean_spread_scores);
  }
  
  return List::create(
    Named("O_balance") = O_balance,
    Named("O_spread") = O_spread
  );
}

// ===================================================================
//           INITIALIZE CACHE
// ===================================================================

void initialize_cache(const arma::uvec& sequence,
                      const arma::uvec& critical_k,
                      bool compute_balance,
                      bool compute_spread,
                      const arma::uvec& initial_sample_indices_global,
                      const arma::mat& population_x,
                      const arma::vec& population_x_totals,
                      const arma::vec& population_probs,
                      const arma::mat& master_sample_coords,
                      PanelScoreCache& cache) {
  
  int n = sequence.n_elem;
  
  for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
    int k = critical_k(k_idx);
    for (int start_pos = 0; start_pos < n; start_pos++) {
      compute_panel_score(sequence, k, start_pos, n, compute_balance, compute_spread,
                          initial_sample_indices_global, population_x, population_x_totals,
                          population_probs, master_sample_coords, cache);
    }
  }
}

// ===================================================================
//           WORST PANEL FINDER (Exported helper)
// ===================================================================

// [[Rcpp::export]]
List get_worst_panel_for_k_cpp(const arma::uvec& sequence,
                               int k_val,
                               std::string objective_type,
                               const arma::uvec& initial_sample_indices_global,
                               const arma::mat& population_x,
                               const arma::vec& population_x_totals,
                               const arma::vec& population_probs,
                               const arma::mat& master_sample_coords) {
  
  int n_len = sequence.n_elem;
  arma::vec slice_scores(n_len);
  
  for (int start_pos = 0; start_pos < n_len; start_pos++) {
    arma::uvec panel_idx_local = get_circular_indices(sequence, start_pos, k_val, n_len);
    
    if (objective_type == "balance") {
      arma::uvec panel_idx_global(k_val);
      for (int i = 0; i < k_val; i++) {
        panel_idx_global(i) = initial_sample_indices_global(panel_idx_local(i) - 1);
      }
      slice_scores(start_pos) = Q_balance_cpp(population_probs, population_x, 
                   population_x_totals, panel_idx_global, 
                   n_len, k_val);
    } else {
      slice_scores(start_pos) = Q_spread_cpp(master_sample_coords, panel_idx_local, 
                   n_len, k_val);
    }
  }
  
  arma::uword max_idx = slice_scores.index_max();
  
  return List::create(
    Named("pos") = max_idx + 1,
    Named("k") = k_val
  );
}

// ===================================================================
//           MAIN SA OPTIMIZER (PROBABILISTIC TARGETING)
// ===================================================================

// [[Rcpp::export]]
arma::uvec optimize_sequence_sa_cpp(arma::uvec initial_sequence,
                                    const arma::uvec& initial_sample_indices_global,
                                    const arma::mat& population_x,
                                    const arma::vec& population_probs,
                                    const arma::mat& master_sample_coords,
                                    const arma::uvec& critical_k,
                                    bool compute_balance,
                                    bool compute_spread,
                                    double w_balance,
                                    double q, // Probability to choose worst panel (Optimization vs Exploration)
                                    int max_iter = 100000,
                                    double initial_temperature = 0.1,
                                    double cooling_rate = 0.999,
                                    int reheat_patience = 5000,
                                    int print_every = 500) {
  
  RNGScope scope;
  
  int n = initial_sequence.n_elem;
  arma::vec population_x_totals = arma::sum(population_x, 0).t();
  
  // Initialize cache
  Rcout << "Initializing panel score cache...\n";
  PanelScoreCache cache_current;
  initialize_cache(initial_sequence, critical_k, compute_balance, compute_spread,
                   initial_sample_indices_global, population_x, population_x_totals,
                   population_probs, master_sample_coords, cache_current);
  
  // Initialize
  arma::uvec s_current = initial_sequence;
  arma::uvec s_best = s_current;
  PanelScoreCache cache_best = cache_current;
  
  List scores_current = compute_O_scores_from_cache(cache_current, critical_k, n,
                                                    compute_balance, compute_spread);
  
  double O_bal_curr = scores_current["O_balance"];
  double O_spr_curr = scores_current["O_spread"];
  double cost_current = 0.0;
  
  if (compute_balance && !compute_spread) {
    cost_current = O_bal_curr;
  } else if (compute_spread && !compute_balance) {
    cost_current = O_spr_curr;
  } else {
    cost_current = w_balance * O_bal_curr + (1 - w_balance) * O_spr_curr;
  }
  
  double cost_best = cost_current;
  double O_bal_best = O_bal_curr;
  double O_spr_best = O_spr_curr;
  
  double temperature = initial_temperature;
  int last_improvement_iter = 0;
  
  Rcout << "Starting SA (Objective: MEAN, Update: WEIGHTED). Initial Cost=" << cost_best 
        << " (O_balance=" << O_bal_best << ", O_spread=" << O_spr_best << ")\n";
  
  // Main loop
  for (int i = 0; i < max_iter; i++) {
    if ((i + 1) % print_every == 0) {
      Rcout << "Iter: " << i + 1 << ", Temp: " << temperature 
            << ", Best Cost: " << cost_best 
            << " (O_balance=" << O_bal_best << ", O_spread=" << O_spr_best << ")\n";
    }
    
    // Reheat
    if ((i - last_improvement_iter) > reheat_patience && temperature < (initial_temperature / 100)) {
      temperature = initial_temperature * 0.5;
      last_improvement_iter = i;
      Rcout << "Reheating to " << temperature << "\n";
    }
    
    // Select focus k (panel size)
    GetRNGstate();
    double rand_k = unif_rand();
    PutRNGstate();
    int k_idx = static_cast<int>(rand_k * critical_k.n_elem);
    if (k_idx >= static_cast<int>(critical_k.n_elem)) k_idx = critical_k.n_elem - 1;
    int k_focus = critical_k(k_idx);
    
    // --- TARGET PANEL SELECTION STRATEGY ---
    // With probability q: Greedy (select worst panel)
    // With probability 1-q: Random (select random panel for connectivity)
    
    GetRNGstate();
    double rand_strategy = unif_rand();
    PutRNGstate();
    
    int target_start_pos = 0;
    
    if (rand_strategy < q) {
      // STRATEGY A: Greedy Selection (Find worst performing panel)
      double max_composite_score = -arma::datum::inf;
      
      for (int start_pos = 0; start_pos < n; start_pos++) {
        std::pair<int, int> key = std::make_pair(k_focus, start_pos);
        double val_bal = 0.0;
        double val_spr = 0.0;
        
        // Safely retrieve scores
        if (compute_balance) {
          auto it = cache_current.balance_scores.find(key);
          if (it != cache_current.balance_scores.end()) val_bal = it->second;
        }
        if (compute_spread) {
          auto it = cache_current.spread_scores.find(key);
          if (it != cache_current.spread_scores.end()) val_spr = it->second;
        }
        
        // Calculate composite score
        double current_composite = 0.0;
        if (compute_balance && compute_spread) {
          current_composite = w_balance * val_bal + (1.0 - w_balance) * val_spr;
        } else if (compute_balance) {
          current_composite = val_bal;
        } else {
          current_composite = val_spr;
        }
        
        // Identify max (worst)
        if (current_composite > max_composite_score) {
          max_composite_score = current_composite;
          target_start_pos = start_pos;
        }
      }
    } else {
      // STRATEGY B: Random Selection (Ensure connectivity)
      GetRNGstate();
      double rand_pos = unif_rand();
      PutRNGstate();
      
      target_start_pos = static_cast<int>(rand_pos * n);
      if (target_start_pos >= n) target_start_pos = n - 1;
    }
    
    // Determine indices inside and outside the target panel
    std::vector<int> indices_in_panel;
    std::vector<int> indices_outside_panel;
    
    for (int idx = 0; idx < n; idx++) {
      int circular_idx = (target_start_pos + idx) % n;
      if (idx < k_focus) {
        indices_in_panel.push_back(circular_idx);
      } else {
        indices_outside_panel.push_back(circular_idx);
      }
    }
    
    // Perform Swap
    arma::uvec s_new = s_current;
    int pos_in = -1, pos_out = -1;
    
    if (indices_outside_panel.size() > 0) {
      GetRNGstate();
      double rand_val_in = unif_rand();
      double rand_val_out = unif_rand();
      PutRNGstate();
      
      int pos_in_idx = static_cast<int>(rand_val_in * indices_in_panel.size());
      if (pos_in_idx >= static_cast<int>(indices_in_panel.size())) pos_in_idx = indices_in_panel.size() - 1;
      pos_in = indices_in_panel[pos_in_idx];
      
      int pos_out_idx = static_cast<int>(rand_val_out * indices_outside_panel.size());
      if (pos_out_idx >= static_cast<int>(indices_outside_panel.size())) pos_out_idx = indices_outside_panel.size() - 1;
      pos_out = indices_outside_panel[pos_out_idx];
      
      int temp = s_new(pos_in);
      s_new(pos_in) = s_new(pos_out);
      s_new(pos_out) = temp;
      
      PanelScoreCache cache_new;
      cache_new.balance_scores = cache_current.balance_scores;
      cache_new.spread_scores = cache_current.spread_scores;
      
      // Get affected panels
      std::set<std::pair<int, int>> affected = get_affected_panels(pos_in, pos_out, critical_k, n);
      
      // Recompute affected
      for (const auto& panel_key : affected) {
        int k_panel = panel_key.first;
        int start_panel = panel_key.second;
        
        arma::uvec panel_indices_local = get_circular_indices(s_new, start_panel, k_panel, n);
        
        if (compute_balance) {
          arma::uvec panel_indices_global(k_panel);
          for (int idx = 0; idx < k_panel; idx++) {
            panel_indices_global(idx) = initial_sample_indices_global(panel_indices_local(idx) - 1);
          }
          double new_score = Q_balance_cpp(population_probs, population_x, 
                                           population_x_totals, panel_indices_global, n, k_panel);
          cache_new.balance_scores[panel_key] = new_score;
        }
        
        if (compute_spread) {
          double new_score = Q_spread_cpp(master_sample_coords, panel_indices_local, n, k_panel);
          cache_new.spread_scores[panel_key] = new_score;
        }
      }
      
      // Evaluate new global Mean Cost
      List scores_new = compute_O_scores_from_cache(cache_new, critical_k, n,
                                                    compute_balance, compute_spread);
      
      double O_bal_new = scores_new["O_balance"];
      double O_spr_new = scores_new["O_spread"];
      double cost_new = 0.0;
      
      if (compute_balance && !compute_spread) {
        cost_new = O_bal_new;
      } else if (compute_spread && !compute_balance) {
        cost_new = O_spr_new;
      } else {
        cost_new = w_balance * O_bal_new + (1 - w_balance) * O_spr_new;
      }
      
      // Accept/reject
      double delta_cost = cost_new - cost_current;
      bool accept = false;
      if (delta_cost < 0) {
        accept = true;
      } else {
        GetRNGstate();
        double rand_accept = unif_rand();
        PutRNGstate();
        accept = (rand_accept < exp(-delta_cost / temperature));
      }
      
      if (accept) {
        s_current = s_new;
        cache_current = cache_new;
        cost_current = cost_new;
        O_bal_curr = O_bal_new;
        O_spr_curr = O_spr_new;
      }
      
      // Update best
      if (cost_current < cost_best) {
        s_best = s_current;
        cache_best = cache_current;
        cost_best = cost_current;
        O_bal_best = O_bal_curr;
        O_spr_best = O_spr_curr;
        last_improvement_iter = i;
        
        Rcout << ">>> New best @ " << i + 1 << ": Cost=" << cost_best
              << " (O_balance=" << O_bal_best << ", O_spread=" << O_spr_best << ") <<<\n";
      }
    }
    
    temperature *= cooling_rate;
  }
  
  return s_best;
}

// ===================================================================
//           O_SCORES FUNCTION (MEAN)
// ===================================================================

// [[Rcpp::export]]
List O_scores_cpp(const arma::uvec& sequence,
                  const arma::uvec& initial_sample_indices_global,
                  const arma::mat& population_x,
                  const arma::vec& population_x_totals,
                  const arma::vec& population_probs,
                  const arma::mat& master_sample_coords,
                  const arma::uvec& critical_k,
                  bool compute_balance,
                  bool compute_spread) {
  
  int n = sequence.n_elem;
  double O_balance = NA_REAL;
  double O_spread = NA_REAL;
  
  if (compute_balance) {
    arma::vec mean_balance_scores(critical_k.n_elem);
    
    for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
      int k = critical_k(k_idx);
      double sum_score = 0.0;
      
      for (int start_pos = 0; start_pos < n; start_pos++) {
        arma::uvec panel_indices_local = get_circular_indices(sequence, start_pos, k, n);
        arma::uvec panel_indices_global(k);
        for (int i = 0; i < k; i++) {
          panel_indices_global(i) = initial_sample_indices_global(panel_indices_local(i) - 1);
        }
        
        double score = Q_balance_cpp(population_probs, population_x, population_x_totals,
                                     panel_indices_global, n, k);
        sum_score += score;
      }
      mean_balance_scores(k_idx) = sum_score / n;
    }
    O_balance = arma::mean(mean_balance_scores);
  }
  
  if (compute_spread) {
    arma::vec mean_spread_scores(critical_k.n_elem);
    
    for (size_t k_idx = 0; k_idx < critical_k.n_elem; k_idx++) {
      int k = critical_k(k_idx);
      double sum_score = 0.0;
      
      for (int start_pos = 0; start_pos < n; start_pos++) {
        arma::uvec panel_indices_local = get_circular_indices(sequence, start_pos, k, n);
        double score = Q_spread_cpp(master_sample_coords, panel_indices_local, n, k);
        sum_score += score;
      }
      mean_spread_scores(k_idx) = sum_score / n;
    }
    O_spread = arma::mean(mean_spread_scores);
  }
  
  return List::create(
    Named("O_balance") = O_balance,
    Named("O_spread") = O_spread
  );
}