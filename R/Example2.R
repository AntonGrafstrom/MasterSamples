# ===================================================================
#           *** Example 2, balance optimization (Rcpp) ***
# ===================================================================

library(dplyr)

# -- 1. Source the R wrapper functions
source("wrapper.R")  # Contains optimize_sequence_sa_fast()

start_time <- Sys.time()

# -- 2. Setup the Problem --
set.seed(123) 
N <- 100000 
n_master <- 1000
k_to_test <- c(100, 200) 

p <- rep(n_master / N, N)
X_balance <- as.matrix(cbind(p, rnorm(N, 10, 2), rnorm(N, 100, 25)))
population_x_totals <- colSums(X_balance)

# -- 3. Create the initial well-balanced master sample --
cat("--- Selecting initial balanced sample using cube method ---\n")
initial_sample_indices_global <- cube(p, X_balance)

# Quick validation using the fast Rcpp version
initial_balance <- Q_balance_cpp(
  p, 
  X_balance, 
  population_x_totals, 
  as.integer(initial_sample_indices_global), 
  as.integer(n_master), 
  as.integer(n_master)
)
cat("Balance of full master sample:", initial_balance, "\n")

# -- 4. Generate initial random order --
cat("\n--- Generating initial random order ---\n")
start_order_local <- sample(1:n_master)

# -- 5. Run the Universal Optimizer for BALANCE ONLY (Rcpp version) --
cat("\n--- Starting SA Optimization for Marginal Balance (Rcpp) ---\n")
improved_order_local <- optimize_sequence_sa_fast(
  initial_sequence = start_order_local,
  critical_k = k_to_test,
  
  # Provide data for the BALANCE objective
  initial_sample_indices_global = initial_sample_indices_global,
  population_x = X_balance,
  population_probs = p,
  
  # Indicate we want balance optimization only
  compute_balance = TRUE,
  compute_spread = FALSE,
  
  # q is probability for targeting worst panel
  q = 0.9,
  
  # SA control parameters
  max_iter = 100000,
  reheat_patience = 10000,
  initial_temperature = 0.00003,
  cooling_rate = 0.999,
  checkpoint_file = "sa_balance_only_checkpoint_rcpp.rds",
  print_every = 500
)
cat("\n--- SA Optimization Finished ---\n")

# ===================================================================
#           *** Final Validation (Min/Mean/Max per k) ***
# ===================================================================
cat("\n--- Starting validation for Marginal Balance ---\n")

# Data structure to hold all validation scores
all_scores_data <- data.frame()

sequences_to_validate <- list(
  "Optimized (Rcpp SA)" = improved_order_local,
  "Original (Random)" = start_order_local
)

for (method_name in names(sequences_to_validate)) {
  local_order <- sequences_to_validate[[method_name]]
  n_seq <- length(local_order)
  circular_order <- c(local_order, local_order)
  cat(sprintf("\n--- Validating all panels for: %s ---\n", method_name))
  
  # Loop through each k value to test
  for (k_val in k_to_test) {
    cat(sprintf("  Calculating scores for k=%d...\n", k_val))
    
    # Calculate the score for EVERY possible panel of size k_val
    # Use Rcpp version for faster validation
    all_slice_scores <- sapply(1:n_seq, function(start_pos) {
      slice_indices_local <- circular_order[start_pos:(start_pos + k_val - 1)]
      subsample_for_validation <- initial_sample_indices_global[slice_indices_local]
      
      # Use Q_balance_cpp for validation
      Q_balance_cpp(p, X_balance, population_x_totals, 
                    as.integer(subsample_for_validation), 
                    as.integer(n_master), as.integer(k_val))
    })
    
    # Store the results for this k and method
    result_rows <- data.frame(
      k = k_val,
      balance_score = all_slice_scores,
      method = method_name
    )
    all_scores_data <- rbind(all_scores_data, result_rows)
  }
}

# --- Summarize the results in a clear table ---
if (nrow(all_scores_data) > 0) {
  summary_table <- all_scores_data %>%
    group_by(method, k) %>%
    summarise(
      min_balance = min(balance_score, na.rm = TRUE),
      mean_balance = mean(balance_score, na.rm = TRUE),
      max_balance = max(balance_score, na.rm = TRUE),
      .groups = 'drop'
    )
  
  cat("\n--- Final Validation Summary (Marginal Balance - Rcpp) ---\n")
  print(as.data.frame(summary_table))
} else {
  cat("Validation produced no results.\n")
}

end_time <- Sys.time()
duration <- end_time - start_time
cat("\n--- Total Run Time ---\n")
print(duration)
