
suppressMessages(library(dplyr))
suppressMessages(library(stringr))
suppressMessages(library(glmnet))

sigmoid <- function(z){exp(z) / (1 + exp(z))}

read_digits <- function(zip_obj){
  lapply(1:nrow(zip_obj),  
         function(i){
           zip_rows = zip_obj[i, ] %>% 
             str_split(., " ") %>%
             unlist() 
           # sometimes it catches an extra ""
           if (zip_rows[length(zip_rows)] == ""){
             zip_rows = zip_rows[1:(length(zip_rows) - 1)]
           }
           as.numeric(zip_rows)
         }
  ) %>%
    do.call("rbind", .) %>%
    data.frame() %>%
    `colnames<-`(c("digit", paste0("X", 1:256))) %>%
    filter(digit == 6. | digit == 8.) %>%
    mutate(y=ifelse(digit == 6., 1, 0))
}

zip_train = read_digits(read.delim2("/Users/IKleisle/Downloads/zip_train.txt")) 
zip_test = read_digits(read.delim2("/Users/IKleisle/Downloads/zip_test.txt"))

# letting glmnet decide the lambda sequence
fit_lasso = cv.glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=1
)

fit_ridge = cv.glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=0
)

fit_enet = cv.glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=0.5
)

### extract lambda corresponding to 1SE
# lam_idx_lasso = which(fit_lasso$lambda == fit_lasso$lambda.1se)
# lam_idx_ridge = which(fit_ridge$lambda == fit_ridge$lambda.1se)
# lam_idx_enet = which(fit_enet$lambda == fit_enet$lambda.1se)


### refit
fit_lasso_ = glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=1,
  lambda=fit_lasso$lambda.1se
)

fit_ridge_ = glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=0,
  lambda=fit_ridge$lambda.1se
)

fit_enet_ = glmnet(
  x=zip_train %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname(),
  y=zip_train %>% pull(y),
  family="binomial",
  alpha=0.5,
  lambda=fit_enet$lambda.1se
)


yhat_lasso = predict(
  fit_lasso, 
  s=fit_lasso$lambda.1se,
  newx=zip_test %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname()
) %>%
  as.numeric() %>%
  sigmoid() %>%
  round()

yhat_ridge = predict(
  fit_ridge, 
  s=fit_ridge$lambda.1se,
  newx=zip_test %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname()
) %>%
  as.numeric() %>%
  sigmoid() %>%
  round()

yhat_enet = predict(
  fit_enet, 
  s=fit_enet$lambda.1se,
  newx=zip_test %>% dplyr::select(X1:X256) %>% as.matrix() %>% unname()
) %>%
  as.numeric() %>%
  sigmoid() %>%
  round()


mean(yhat_lasso == zip_test$y)
mean(yhat_ridge == zip_test$y)
mean(yhat_enet == zip_test$y)

