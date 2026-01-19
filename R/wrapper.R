# -- Required Libraries --
library(Rcpp)
library(RcppArmadillo)
library(BalancedSampling)

# Source the Rcpp file
sourceCpp("../src/oms_engine.cpp")

# ===================================================================
#           WRAPPER FUNCTIONS (R interface to Rcpp)
# ===================================================================

Q_balance_fast <- function(population_probs, population_x, population_x_totals, 
                           panel_indices_global, n_master, k_panel) {
  if (!is.matrix(population_x)) population_x <- as.matrix(population_x)
  # Ensure indices are integers (R 1-based)
  return(Q_balance_cpp(population_probs, population_x, population_x_totals,
                       as.integer(panel_indices_global), 
                       as.integer(n_master), as.integer(k_panel)))
}

Q_spread_fast <- function(master_sample_coords, panel_indices_local, n_master, k_panel) {
  # Ensure indices are integers (R 1-based)
  return(Q_spread_cpp(master_sample_coords, as.integer(panel_indices_local),
                      as.integer(n_master), as.integer(k_panel)))
}

O_scores_fast <- function(sequence, initial_sample_indices_global,
                          population_x, population_x_totals, population_probs, 
                          master_sample_coords, critical_k,
                          compute_balance = TRUE, compute_spread = TRUE) {
  
  if (!is.matrix(population_x)) population_x <- as.matrix(population_x)
  if (!is.matrix(master_sample_coords)) master_sample_coords <- as.matrix(master_sample_coords)
  
  result <- O_scores_cpp(
    as.integer(sequence),
    as.integer(initial_sample_indices_global),
    population_x,
    population_x_totals,
    population_probs,
    master_sample_coords,
    as.integer(critical_k),
    compute_balance,
    compute_spread
  )
  
  return(list(O_balance = result$O_balance, O_spread = result$O_spread))
}

# ===================================================================
#           MAIN OPTIMIZER (Rcpp version)
# ===================================================================

optimize_sequence_sa_fast <- function(initial_sequence, 
                                      # Data arguments
                                      initial_sample_indices_global = NULL,
                                      population_x = NULL, 
                                      population_probs = NULL,
                                      master_sample_coords = NULL,
                                      # Objective function arguments
                                      compute_balance = FALSE,
                                      compute_spread = FALSE,
                                      w_balance = NULL,
                                      q = 0.9, # Probability of greedy swap (vs random connectivity swap)
                                      # SA control parameters
                                      max_iter = 100000, 
                                      initial_temperature = 0.1, 
                                      cooling_rate = 0.999,
                                      reheat_patience = 5000,
                                      checkpoint_file = "sa_checkpoint.rds",
                                      print_every = 500,
                                      k_neighbors = 3,
                                      # Problem-specific parameters
                                      critical_k) {
  
  n <- length(initial_sequence)
  
  # Handle data preparation
  if (!is.null(population_x) && !is.matrix(population_x)) {
    population_x <- as.matrix(population_x)
  }
  if (!is.null(master_sample_coords) && !is.matrix(master_sample_coords)) {
    master_sample_coords <- as.matrix(master_sample_coords)
  }
  
  # Set default empty matrices if not provided
  if (is.null(population_x)) {
    population_x <- matrix(0, nrow = 1, ncol = 1)
  }
  if (is.null(population_probs)) {
    population_probs <- numeric(0)
  }
  if (is.null(master_sample_coords)) {
    master_sample_coords <- matrix(0, nrow = 1, ncol = 1)
  }
  if (is.null(initial_sample_indices_global)) {
    initial_sample_indices_global <- seq_along(initial_sequence)
  }
  
  # Determine problem type
  is_doubly_balanced <- compute_balance && compute_spread
  
  # Auto-calculate w_balance if needed
  if (is_doubly_balanced && is.null(w_balance)) {
    cat("--- Doubly balanced problem detected. Auto-calculating w_balance... ---\n")
    
    population_x_totals <- colSums(population_x)
    scores_initial <- O_scores_cpp(
      as.integer(initial_sequence),
      as.integer(initial_sample_indices_global),
      population_x,
      population_x_totals,
      population_probs,
      master_sample_coords,
      as.integer(critical_k),
      TRUE,
      TRUE
    )
    
    w_balance <- scores_initial$O_spread / 
      (scores_initial$O_balance + scores_initial$O_spread + 1e-20)
    cat(sprintf("Auto-calculated w_balance: %.4f\n", w_balance))
  } else if (!is.null(w_balance)) {
    cat(sprintf("Using provided w_balance: %.4f\n", w_balance))
  } else {
    w_balance <- 0.5  # Default for single-objective (won't be used)
  }
  
  # Run the optimized C++ version
  result <- optimize_sequence_sa_cpp(
    as.integer(initial_sequence),
    as.integer(initial_sample_indices_global),
    population_x,
    population_probs,
    master_sample_coords,
    as.integer(critical_k),
    compute_balance,
    compute_spread,
    w_balance,
    q,
    as.integer(max_iter),
    initial_temperature,
    cooling_rate,
    as.integer(reheat_patience),
    as.integer(print_every)
  )
  
  # Save checkpoint
  if (!is.null(checkpoint_file)) {
    saveRDS(as.integer(result), file = checkpoint_file)
  }
  
  return(as.integer(result))
}