# ===================================================================
#       *** Example 3, double balance optimization (Rcpp) ***
# ===================================================================

# -- 1. Load libraries and source functions --
library(BalancedSampling)
library(dplyr)

source("wrapper.R") # Contains optimize_sequence_sa_fast()

start_time <- Sys.time()


# -- 2. Setup the problem
set.seed(456)
N <- 100000 
n_master <- 500
k_to_test <- c(50, 100)

p <- rep(n_master / N, N)

# --- Population Data ---
X_balance <- as.matrix(cbind(p, rnorm(N, 10, 2), rnorm(N, 100, 25)))
population_x_totals <- colSums(X_balance)
X_coords <- as.matrix(cbind(runif(N), runif(N)))

# -- 3. Select initial master sample ---
cat("--- Selecting initial doubly balanced sample using lcube ---\n")
initial_sample_indices_global <- lcube(p, X_coords, X_balance)
master_sample_coords <- X_coords[initial_sample_indices_global, , drop = FALSE]

# -- 4. Generate initial random order ---
cat("\n--- Generating initial random order ---\n")
start_order_local <- sample(1:n_master)

# -- 5. Call the universal optimizer for the DOUBLY BALANCED case
cat("\n--- Starting SA Optimization for Doubly Balanced Sample (Rcpp) ---\n")
improved_order_doubly <- optimize_sequence_sa_fast(
  initial_sequence = start_order_local,
  critical_k = k_to_test,
  
  # Indicate we want a doubly balanced optimization
  compute_balance = TRUE,
  compute_spread = TRUE,
  
  # q is probability for targeting worst panel
  q = 0.9,
  
  # Let w_balance be auto-calculated by passing NULL
  w_balance = NULL, 
  
  # Provide all necessary data
  initial_sample_indices_global = initial_sample_indices_global,
  population_x = X_balance,
  population_probs = p,
  master_sample_coords = master_sample_coords,
  
  # --- SA Control Parameters ---
  max_iter = 100000,
  initial_temperature = 0.0001, 
  cooling_rate = 0.999,
  reheat_patience = 10000,
  print_every = 500
)
cat("\n--- SA Optimization Finished ---\n")


# ===================================================================
#           *** FINAL VALIDATION ***
#           (Unconditional, All Panels, Min/Mean/Max per k)
# ===================================================================
cat("\n--- Starting final validation ---\n")

# --- Validation Helper for Spread (Unconditional) ---
spread_validation_unconditional <- function(population_coords, panel_indices_global, 
                                            population_probs, n_master, k_panel) {
  probs_for_sb <- population_probs * (k_panel / n_master)
  return(sb(prob = probs_for_sb, x = population_coords, sample = panel_indices_global))
}

all_validation_scores <- data.frame()

sequences_to_validate <- list(
  "Optimized (Rcpp SA)" = improved_order_doubly,
  "Original (Random)" = start_order_local
)

for (method_name in names(sequences_to_validate)) {
  local_order <- sequences_to_validate[[method_name]]
  n_seq <- length(local_order)
  circular_order <- c(local_order, local_order)
  cat(sprintf("\n--- Validating all panels for: %s ---\n", method_name))
  
  for (k_val in k_to_test) {
    cat(sprintf("  Calculating scores for k=%d...\n", k_val))
    
    all_balance_scores_for_k <- sapply(1:n_seq, function(start_pos) {
      panel_indices_local <- circular_order[start_pos:(start_pos + k_val - 1)]
      panel_indices_global <- initial_sample_indices_global[panel_indices_local]
      # Using the fast Q_balance_cpp for validation
      Q_balance_cpp(p, X_balance, population_x_totals, panel_indices_global, n_master, k_val)
    })
    
    all_spread_scores_for_k <- sapply(1:n_seq, function(start_pos) {
      panel_indices_local <- circular_order[start_pos:(start_pos + k_val - 1)]
      panel_indices_global <- initial_sample_indices_global[panel_indices_local]
      spread_validation_unconditional(X_coords, panel_indices_global, p, n_master, k_val)
    })
    
    result_rows <- data.frame(
      k = k_val, balance_score = all_balance_scores_for_k,
      spread_score = all_spread_scores_for_k, method = method_name)
    all_validation_scores <- rbind(all_validation_scores, result_rows)
  }
}

# --- Summarize the results in a clear table ---
if (nrow(all_validation_scores) > 0) {
  summary_table <- all_validation_scores %>%
    group_by(method, k) %>%
    summarise(
      min_balance = min(balance_score, na.rm = TRUE),
      mean_balance = mean(balance_score, na.rm = TRUE),
      max_balance = max(balance_score, na.rm = TRUE),
      min_spread = min(spread_score, na.rm = TRUE),
      mean_spread = mean(spread_score, na.rm = TRUE),
      max_spread = max(spread_score, na.rm = TRUE),
      .groups = 'drop'
    )
  
  cat("\n--- Final Unconditional Validation Summary ---\n")
  print(as.data.frame(summary_table))
} else {
  print("Validation produced no results.")
}

end_time <- Sys.time()
duration <- end_time - start_time
cat("\n--- Total Run Time ---\n")
print(duration)

