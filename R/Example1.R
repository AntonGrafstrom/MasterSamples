# ===================================================================
#           *** Example 1, spread optimization  (Rcpp) ***
# ===================================================================

library(dplyr)

# -- 1. Source the R wrapper functions
source("wrapper.R")  # Contains optimize_sequence_sa_fast()

start_time <- Sys.time()

# -- 2. Setup the Problem --
set.seed(456)
N <- 100000
n_master <- 360
k_to_test <- c(72) #panel sizes to optimize 
p <- rep(n_master/N,N)
X_coords <- as.matrix(scale(cbind(runif(N), runif(N))))

# -- 3. Create the initial well-spread master sample --
cat("--- Selecting initial LPM sample ---\n")
initial_sample_indices_global <- lpm2(p, X_coords)

# Extract coordinates for the n units in our master sample
master_sample_coords <- X_coords[initial_sample_indices_global, , drop = FALSE]

initial_spread <- sb(p, X_coords, initial_sample_indices_global) 
cat("\nInitial Spread of Master Sample (vs. Population):", initial_spread, "\n")

# -- 4. Generate initial random order --
cat("\n--- Generating initial random order ---\n")
start_order_local <- sample(1:n_master)

# -- 5. Run the Universal Optimizer for SPREAD ONLY (Rcpp version) --
cat("\n--- Starting SA Optimization for Spread (Rcpp) ---\n")
improved_order_local <- optimize_sequence_sa_fast(
  initial_sequence = start_order_local,
  critical_k = k_to_test,
  
  # Provide data for the SPREAD objective
  master_sample_coords = master_sample_coords,
  initial_sample_indices_global = initial_sample_indices_global, # Needed by helper functions
  
  # Indicate we want spread optimization only
  compute_balance = FALSE,
  compute_spread = TRUE,
  w_balance = NULL,
  
  # q is probability for targeting worst panel
  q = 0.9,
  
  # SA control parameters
  max_iter = 100000,
  initial_temperature = 0.03,
  cooling_rate = 0.999,
  reheat_patience = 10000,
  checkpoint_file = "sa_spread_only_checkpoint_rcpp.rds",
  print_every = 500
)
cat("\n--- SA Optimization Finished ---\n")

# ===================================================================
#                    *** Final Validation ***
# ===================================================================
cat("\n--- Starting Final Validation (Evaluating Against Population) ---\n")

# This will store ALL scores, not just the max
all_scores_data <- data.frame()

sequences_to_validate <- list(
  "Optimized (SA)" = improved_order_local,
  "Original (random)" = start_order_local
)

for (method_name in names(sequences_to_validate)) {
  local_order <- sequences_to_validate[[method_name]]
  n_val <- length(local_order)
  circular_order <- c(local_order, local_order)
  
  cat(sprintf("\n--- Validating all panels for: %s ---\n", method_name))
  
  # Loop through each k value to test
  for (k_val in k_to_test) {
    cat(sprintf("  Calculating scores for k=%d...\n", k_val))
    
    # Calculate the score for EVERY possible panel
    all_slice_scores <- sapply(1:n_val, function(start_pos) {
      slice_indices_local <- circular_order[start_pos:(start_pos + k_val - 1)]
      subsample_for_validation <- initial_sample_indices_global[slice_indices_local]
      
      # Use the correct UNCONDITIONAL measure for validation.
      sb(prob = p * k_val / n_master, x = X_coords, sample = subsample_for_validation)
    })
    
    # Store the results in a long-format data frame suitable for ggplot
    result_rows <- data.frame(
      k = as.factor(k_val), # Treat k as a categorical variable for plotting
      score = all_slice_scores,
      method = method_name
    )
    all_scores_data <- rbind(all_scores_data, result_rows)
  }
}

# --- Summarize the results in a table (max, min, mean) ---
summary_table <- all_scores_data %>%
  group_by(method, k) %>%
  summarise(
    min_score = min(score),
    mean_score = mean(score),
    max_score = max(score),
    .groups = 'drop'
  )

cat("\n--- Summary Statistics of Panel Scores ---\n")
print(summary_table)

end_time <- Sys.time()
duration <- end_time - start_time
cat("\n--- Total Run Time ---\n")
print(duration)
