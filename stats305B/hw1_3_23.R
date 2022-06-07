

# 3.7 ---------------------------------------------------------------------




# 3.17 --------------------------------------------------------------------
library(MESS)

### a.) Regular Chi-Sq Test ###
data = matrix(
  c(
    c(9, 44, 13, 10),
    c(11, 52, 23, 22),
    c(9, 41, 12, 27)
  ),
  byrow=T,
  ncol=4
)
colnames(data) = c("SH", "HS", "SC", "C")
rownames(data) = c("L", "M", "H")

### test fails to reject the hypothesis of independence.
# Key deficiency is that the test does not account for the ordinal 
# nature of the data, i.e. SH < HS < SC < C or L < M < H. They're just standard
# buckets, and the directed nature of the relationship is not baked into the test. 
# In this way, key information is overlooked by the test.
chi_sq_test_result = chisq.test(data)

### b.) Standardized Residuals ###
N = sum(data)
# marginal MLEs
pi_row = rowSums(data) / sum(data)
pi_col = colSums(data) / sum(data)
# model expectations
u_hat = pi_row %*% t(pi_col) * N
r_scale = sqrt(
  ((1 - pi_row)%*% t(1 - pi_col)) * u_hat
)
# standardized residuals
(data - u_hat) / r_scale

# as education increases (i.e. columns moving L-> R), we see increasingly large standardized residuals --
# eventually exceeding the problematic |residual| > 2 or 3 admonished by the book. Per Agresti,
# these large standardized residuals indicate a lack of fit, suggesting that the ordinary Chi-Squared
# is increasingly inadequate as education increases. 

### c.) Kruskal Test ###
# In light of the apparent ordinality borking the Chi-Squared test, a Kruskal test
# may be preferable here
MESS::gkgamma(data)


# 3.23 --------------------------------------------------------------------

# As prescribed in Agresti 3.6.2 (p. 97), if we assume row-wise binomial independence,
# we may put beta priors on the row-wise binomials to obtain a row-wise posterior beta. 
# As further prescribed in 3.6.2, we may further simulate the logs ratio by leveraging 
# the presumed independence, and taking S draws from the row 1 posterior, followed by 
# S draws from the row 2 posterior (independent of row 1), and then go element-by-element
# through these two sets of draws to compute Monte Carlo odds ratios, for our inferential 
# purposes. 

library(ggplot2)
library(dplyr)
library(HDInterval)
set.seed(2022)

### number of simulated draws ###
S = 100000
### interval width ###
Q = .95

### specify priors: uniform(1, 1) ###
ALPHA = c(1, 1)

### make data ###
# first row of data
y1 = c(763, 65)
# second row of data
y2 = c(59, 680)

### row-wise posteriors ###
y1_post = rbeta(S, y1[1] + ALPHA[1], y1[2] + ALPHA[2])
y2_post = rbeta(S, y2[1] + ALPHA[1], y2[2] + ALPHA[2])

### use simulations for odds ratio, and sort for ECDF ease ###
# strictly due to Bush being on the LHS of the table, define that as the 1 outcome.
odds_ratio_sim = sort(
  (y1_post * (1 - y2_post)) / 
  ((1 - y1_post) * y2_post) 
)

### plot density ###
ggplot(
  data.frame(odds_ratio=odds_ratio_sim),
  aes(x=odds_ratio)
) +
  geom_density() +
  labs(y="Density", x="Simulated Odds Ratio")

### compute CI size and tail size ###
tail_size = as.integer((S * (1 - Q)) / 2) # chose a convenient S
ci_size = S - 2 * tail_size


### compute equital CI ###
equitail_ci = c(
  odds_ratio_sim[tail_size + 1],
  odds_ratio_sim[S - (tail_size + 1)]
)
print(equitail_ci)

### compute HPD interval ###
print(hdi(odds_ratio_sim, .95))

# Importantly, the HPD interval here on an odds ratio is dangerous due to its non-invariance
# under nonlinear parameter transformation (Agresti 3.6.5. For example, suppose we wanted to 
# invert our odds ratio -- i.e. in the case we relabeled the data or redefined "success" -- 
# we could not just invert the HPD interval, and instead we would have to start from scratch. The example
# in the second paragraph of 3.6.5 is a perfect cautionary tale of what inversion could do in this problem. 
  
  
  
  
  
  
  
  