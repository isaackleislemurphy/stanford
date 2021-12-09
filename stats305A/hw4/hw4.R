suppressMessages(library(dplyr))
suppressMessages(library(ggplot2))
source("~/Stanford/STATS305A/generate_outlier_data.R")


# Ingest + Setup ----------------------------------------------------------
# # make data
# data = generate.data()
# # split it up
# x_train = data$X.train; x_test = data$X.test
# y_train = data$y.train; y_test = data$y.test
# # print dims for sanity
# cat(
#   paste0(rep("#", 75), collapse=""), "\n",
#   "Dimensions:\n",
#   "\t X-train: \t", dim(x_train), "\n",
#   "\t Y-train: \t", dim(y_train), "\n",
#   "\t X-test: \t", dim(x_test), "\n",
#   "\t X-test: \t", dim(y_test), "\n",
#   paste0(rep("#", 75), collapse="")
# )


# Utils -------------------------------------------------------------------

# establish sigmoid function
sigmoid <- function(x, op='-'){
  #' Basic sigmoid, you pick the sign
  if (op == '-'){
    x_sign = -x
  }else{
    x_sign = x
  }
  exp(x_sign) / (1 + exp(x_sign))
}

loss_fn <- function(x){log(1 + exp(x)) + log(1 + exp(-x))}

loss_prime <- function(x){
  # sigmoid(x, "+") - sigmoid(x, "-")
  tanh(x/2)
}


loss_double_prime <- function(x){
  # second derivative, no chain rule, of loss function
  sigmoid(x, '+')/(1 + exp(x)) +
    sigmoid(x, '-')/(1 + exp(-x))
}



score_loo <- function(X, y, lambda=1e-3){
  suppressMessages(require(dplyr))
  # number of observations
  N = nrow(X)
  # does a single round of LOO
  beta_hat = minimize.robust.ridge(X, y, lambda)
  # residuals
  eps_hat = as.numeric(y - (X %*% beta_hat))
  # diagonal matrix of losses, double primed
  L = eps_hat %>%
    loss_double_prime() %>%
    diag()
  # solve H: X^T(L'')X
  H = lambda * diag(dim(X)[2]) + (t(X) %*% L %*% X) / N
  # invert it
  H_inv = solve(H)
  # compute H_ks. Forgetting broadcasting rules off top of my head for R, 
  # so just going to stuff into list
  loo_errors = sapply(1:N, function(k){
    # apply problem 1: Sherman-Morrison to get Hk
    #   A = H
    #   u = l''(\hat\epsilon_k)x_k
    #   v^T = x_k^T
    # setup SMW decomp; easier to track
    A_inv = H_inv
    u = (-L[k, k] / N) * X[k, ]
    v = X[k, ]
    # solve for H_k^{-1}
    Hk_inv = A_inv - 
      (A_inv %*% u %*% t(v) %*% A_inv) /
        as.numeric(
          1 + t(v) %*% A_inv %*% u
        )
    
    
    # direct solve, for comparison
    # Hk_inv_ = solve(H - (1/N) * L[k, k] * X[k, ] %*% t(X[k, ]) )
    
    # l'(t) = exp(t) / (1 + exp(t)) - exp(-t) / (1 + exp(-t)) = tanh(t/2)
    # by above
    # sigmoid(eps_hat[k], "+") - sigmoid(eps_hat[k], "-") == tanh(eps_hat[k] / 2)
    g_k = tanh(eps_hat[k] / 2) * X[k, ] / N 
    
    # solve out
    yhat_not_k = as.numeric(t(X[k, ]) %*% beta_hat) - #\hat y
      as.numeric(t(X[k, ]) %*% Hk_inv %*% g_k)
    
    # LOO error
    eps_hat_not_k = y[k,] - yhat_not_k
    # out
    eps_hat_not_k
  })
  mean(loss_fn(loo_errors))
}







