## Helper file for generating data with outliers and also solving the
## robust ridge regression problem with logistic-type loss.
## This loss is
##
## l(t) = log(1 + e^t) + log(1 + e^{-t}).
##
## Note that for numerical stability, when evaluating this loss, it may
## be important to use the identities
##
## log(1 + e^t) = log(1 + e^{-t}) + t
## log(1 + e^{-t}) = log(1 + e^t) - t,
##
## depending on whether t is highly positive or highly negative.

library(CVXR);

## generate.data(n, d, num.outliers)
##
## Generates training data for the robust regression problem. The
## number of outliers num.outliers defaults to 10.
generate.data <- function(nn = 100, dd = 25, num.outliers = 10, random_state=1234) {
  ##############################################
  # This is the only modification made to the starter code
  ##############################################
  set.seed(random_state)
  X.train = matrix(rnorm(nn * dd, 0, 1), nn, dd);
  X.test = matrix(rnorm(nn * dd, 0, 1), nn, dd);
  
  beta.star = rnorm(dd, 0, 1);
  beta.star = beta.star / sqrt(sum(beta.star^2));  # Makes X * beta ~ N(0, 1)
  
  train.noise = rnorm(nn, 0, 1);
  train.outliers = sample.int(nn, num.outliers);
  test.noise = rnorm(nn, 0, 1);
  test.outliers = sample.int(nn, num.outliers);
  
  ## Generate outlier measurements
  
  y.train = X.train %*% beta.star + train.noise;
  signs = sign(rnorm(num.outliers)) # Symmetric random outliers
  y.train[train.outliers] = 5 * signs * rnorm(num.outliers)^4
  y.test = X.test %*% beta.star + test.noise;
  signs = sign(rnorm(num.outliers)) # Symmetric noise
  y.test[test.outliers] = 5 * signs * rnorm(num.outliers)^4
  return(list("X.train" = X.train, "y.train" = y.train,
              "X.test" = X.test, "y.test" = y.test))
}

## Function to fit the best model to this data using the log(1 + exp)
## loss. To use this function, simply call
##
## minimize.robust.ridge(X.train, y.train, lambda),
##
## which will return a vector minimizing
##
##  (1/n) sum_{i = 1}^n l(y - x_i' * b) + (lambda/2) * ||b||^2
##
## where ||.|| denotes the l2 norm.
minimize.robust.ridge <- function(X, y, lambda) {
  nn = dim(X)[1]
  dd = dim(X)[2]
  beta <- Variable(dd)
  obj <- ((1/nn) * sum(logistic(X %*% beta - y))
          + (1/nn) * sum(logistic(y - X %*% beta))
          + (lambda/2) * sum_squares(beta))
  problem <- Problem(Minimize(obj))
  result <- solve(problem)
  return(result$getValue(beta))
}













