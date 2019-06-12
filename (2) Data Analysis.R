# STAT 359: Final project
# (2) Data analysis R script 
# John Lee

# list of the methods i will try:
# (1) logistic lasso
# (2) SVM - linear, polynomial, radial 
# (3) trees - bagging, RF, boosted 


# load packages
library(tidyverse)
library(forcats) # for factor vars
library(modelr)
library(skimr)
library(janitor)
library(glmnet) # ridge and lasso
library(glmnetUtils) # improves working with glmnet
library(tree) # for CART
library(randomForest) 
library(onehot)
library(keras) # for NNs in R
library(tensorflow)
library(nnet) # for multinomial 
library(e1071) # for SVM
library(caret) # for ML functions (e.g., confusion matrix)
library(missForest) # for prodNA() -- for seeding the MI 
library(xgboost) # for XG boost
library(mlr)
library(tuneRanger) # for tuning RF


# Set seed
set.seed(3)


# Load the files
train_set <- read_rds("train_set.rds") %>% as_tibble() %>% clean_names()
test_set <- read_rds("test_set.rds") %>% as_tibble() %>% clean_names()

# Inspect the training set
train_set %>% skim
train_set %>% class

# Inspect the test set
test_set %>% skim

# Create versions of the train and test sets in which the categorical vars are one-hot encoded (OHE)

# Create OHE train set ---------------

non_ohe_train <- train_set %>% select(-educ, -race)

# One-hot encode the categorical predictors
ohe_train <- train_set %>% 
  # Only select the cat predictors
  select(educ, race) %>%
  onehot::onehot(max_levels = 40) %>% # use onehot to encode variables
  predict(as_tibble(train_set)) %>% # get OHE matrix
  as_tibble() # Convert back to tibble

# Create another combined df -- using the OHE cat predictors 
ohe_train_set <- base::cbind(non_ohe_train, ohe_train) %>% as_tibble() %>% clean_names()

ohe_train_set %>% skim


# Create OHE test set ---------------

non_ohe_test <- test_set %>% select(-educ, -race)

# One-hot encode the categorical predictors
ohe_test <- test_set %>% 
  # Only select the cat predictors
  select(educ, race) %>%
  onehot::onehot(max_levels = 40) %>% # use onehot to encode variables
  predict(as_tibble(test_set)) %>% # get OHE matrix
  as_tibble() # Convert back to tibble

# Create another combined df -- using the OHE cat predictors 
ohe_test_set <- base::cbind(non_ohe_test, ohe_test) %>% as_tibble() %>% clean_names()

ohe_test_set %>% skim



# Part A: Use CV to (optimize tuning and) identify the candidate models ------------------------

# Candidate Model 1: lasso logistic regression 

# Set up the lambda grid (200 values)
lambda_grid <- 10^seq(-2, 10, length = 200)

# Function to compute glm error rate
error_rate_glm <- function(model, data){
  as_tibble(data) %>% 
    mutate(# Compute the pred prob of votedfortrump being = 1
      pred_prob = predict(model, newdata = data, type = "response"),
      # Set pred class = 1 if the pred prob of trump vote is > .05
      pred_votedfortrump = if_else(pred_prob > 0.5, 1, 0),
      # Compare the pred and actual class of votedfortrump
      error = votedfortrump != pred_votedfortrump) %>% 
    # Extract the errors as a vector and compute the error rate
    pull(error) %>% 
    mean()
}


# Find the optimal lambda by using K-fold CV (with k = 10)
lasso_glm_10cv <- train_set %>% 
  cv.glmnet(formula = votedfortrump ~ ., data = ., 
            # alpha = 1 for lasso
            alpha = 1, 
            # Specify that it's a logistic glm
            type.measure = "class", nfolds = 10, family = "binomial",
            lambda = lambda_grid)

# Check plot of cv error
plot(lasso_glm_10cv)

# Best lambda for the lasso mod (using 10-fold CV)
lasso_lambda_min <- lasso_glm_10cv$lambda.min

lasso_lambda_min

# Fit and save the best model per method in a tibble
wft_glm_lasso <- tibble(train = train_set %>% list(),
                        test  = test_set %>% list()) %>%
  mutate(glm_lasso = map(train, ~ glmnet(votedfortrump ~ ., data = .x, family = "binomial",
                                         alpha = 1, lambda = lasso_lambda_min))) %>% 
  gather(key = method, value = model_fit, -test, -train)

wft_glm_lasso

# Get the test error for the logistic lasso candidate models (w/ best lambdas)
glm_lasso_results <- wft_glm_lasso %>% 
  mutate(test_error = map2_dbl(model_fit, test, error_rate_glm)) %>% 
  unnest(test_error, .drop = TRUE)

# test error: 0.0808
glm_lasso_results %>% knitr::kable(digits = 3)


# Inspect/compare model coefficients -- from the final model (fit on the full training set)
final_lasso_coefs <- wft_glm_lasso %>% 
  pluck("model_fit") %>% 
  map( ~ coef(.x) %>% 
         as.matrix() %>% 
         as.data.frame() %>% 
         rownames_to_column("name")) %>%
  reduce(full_join, by = "name") %>% 
  mutate_if(is.double, ~ if_else(. == 0, NA_real_, .)) %>% 
  rename(coefficient = s0) %>% 
  arrange(desc(abs(coefficient))) 

# Save the results
write_rds(final_lasso_coefs, "final_lasso_coefs.rds")

final_lasso_coefs <- read_rds("final_lasso_coefs.rds")

final_lasso_coefs %>% View


# Candidate Model 2a: Linear SVC ---------------

# Create a workflow tibble for all the SVM models --------------
svm_wft <- tibble(train = train_set %>% list(), 
                        test = test_set %>% list())

# Function to extract the training/test error rate
svm_error_rate <- function(confusion_matrix) {
  
  # Compute and return the error rate
  return(1 - as.numeric(unlist(confusion_matrix)[["overall.Accuracy"]]))
}


# Create a sequence of possible values for cost: from 0.01 to 10
cost_options <- seq(0.01, 10, length = 100)

# Use 10-fold CV to find the optimal cost parameter
linear_svc_cv <- svm_wft %>%
  mutate(tune_svc_linear = map(.x = train, 
                               .f = function(x){ 
                                 return(tune(svm, votedfortrump ~., data = x, kernel = "linear", 
                                             
                                             # let cost range over several values
                                             ranges = list(cost = cost_options))
                                 )
                               }))

# The best cost parameter: 1.826364; best CV error rate: 0.09307813 
linear_svc_cv$tune_svc_linear[[1]] 

# Save the results of the linear SVC CV process
write_rds(linear_svc_cv$tune_svc_linear[[1]], "linear_svc_cv.rds")

linear_svc_cv <- read_rds("linear_svc_cv.rds")

linear_svc_cv

# Fit the candidate model (w best parameters) and store the results 
wft_linear_svm <- svm_wft %>%
  mutate(model_fit = map(.x = train, # fit the model
                         .f = function(x) svm(votedfortrump ~ ., data = x, cost = 1.826364, 
                                              kernel = "linear", probability = TRUE)), 
         # Generate the predicted class -- using the training and test sets 
         train_pred = map2(model_fit, train, predict, probability = TRUE), 
         test_pred = map2(model_fit, test, predict, probability = TRUE),
         # x$votedfortrump = actual outcomes, y = predicted outcomes
         train_conf_matrix = map2(.x = train, .y = train_pred,  # get confusion matrix for the training error
                                  .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)),
         # again, x$votedfortrump = actual outcomes, y = predicted outcomes
         test_conf_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix for the test error
                                 .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)))

# Generate and store the training/test error for the linear SVM
linear_svm_results <- wft_linear_svm %>%
  select(train_conf_matrix, test_conf_matrix) %>%
  mutate(method = "linear_svm",
         train_error = map_dbl(.x = train_conf_matrix, .f = svm_error_rate),
         test_error = map_dbl(.x = test_conf_matrix, .f = svm_error_rate)) %>%
  select(method, train_error, test_error)

# Display the test error for the linear SVM: 0.0836
linear_svm_results



# Candidate Model 2b: Radial SVM ---------------


# Use 10-fold CV to find the optimal parameter values for the radial kernel: cost, gamma 

# Create a sequence of possible values for cost and gamma
cost_options <- seq(0.01, 10, length = 10)
gamma_options <- seq(0.01, 4, length = 5)


# Set up a tibble with different combos of values for max depth and the learning rate 
cost_options <- tibble(cost = seq(0.01, 10, length = 10))

model_def <- tibble(gamma = seq(0.01, 4, length = 5)) %>%
  crossing(cost_options)

model_def

fit_radial_svm <- function(data, cost, gamma) {
  
  data <- as_tibble(data)
  
  svm_mod <- svm(votedfortrump ~ ., data = data, kernel = "radial", # Specify that it's radial
      cost = cost, gamma = gamma, probability = TRUE)
  
  return(svm_mod)
  
}

# Perform the 10-fold CV to find the optimal tuning parameters
radial_svm_cv <- train_set %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of eta
  crossing(model_def) %>%
  mutate(param_combo = paste0("cost: ", as.character(cost), ", ", 
                              "gamma: ", as.character(gamma)),
         # Fit a model to each fold (~500)
         model_fit = pmap(.l = list(data = train, cost = cost, gamma = gamma), .f = fit_radial_svm)
         )

# CV, part 2 -- gen pred class for test set, gen test error
radial_svm_cv <- radial_svm_cv %>%
  mutate(
    conv_test = map(.x = test, .f = as_tibble), # Convert the resamble obj into a tibble
    # Generate the predicted class -- using the test sets 
    test_pred = map2(model_fit, conv_test, predict, probability = TRUE),
    # again, x$votedfortrump = actual outcomes, y = predicted outcomes
    test_conf_matrix = map2(.x = conv_test, .y = test_pred,  # get confusion matrix for the test error
                            .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)),
    # Generate the test error
    test_error = map_dbl(.x = test_conf_matrix, .f = svm_error_rate)
    ) 

# Display the CV results: best parameters: cost: 1.12, gamma: 0.01 (CV error 0.0925)
radial_svm_cv %>% 
  group_by(param_combo) %>% 
  summarize(test_error = mean(test_error)) %>%
  arrange(test_error)

# Save the results of the radial SVM CV
write_rds(radial_svm_cv, "radial_svm_cv.rds")

radial_svm_cv <- read_rds("radial_svm_cv.rds")

# Fit the candidate model (w best parameters) and store the results 
wft_radial_svm <- svm_wft %>%
  mutate(model_fit = map(.x = train, 
                         .f = function(x) svm(votedfortrump ~ ., data = x, kernel = "radial", 
                                              cost = 1.12, gamma = 0.01, probability = TRUE)),
         # Generate the predicted class of trump vote -- using the training and test sets 
         train_pred = map2(model_fit, train, predict, probability = TRUE), 
         test_pred = map2(model_fit, test, predict, probability = TRUE),
         # x$votedfortrump = actual outcomes, y = predicted outcomes
         train_conf_matrix = map2(.x = train, .y = train_pred,  # get confusion matrix for the training error
                                  .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)),
         # again, x$votedfortrump = actual outcomes, y = predicted outcomes
         test_conf_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix for the test error
                                 .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)))


# Generate and store the training/test error for the radial SVM
radial_svm_results <- wft_radial_svm %>%
  select(train_conf_matrix, test_conf_matrix) %>%
  mutate(method = "radial_svm",
         train_error = map_dbl(.x = train_conf_matrix, .f = svm_error_rate),
         test_error = map_dbl(.x = test_conf_matrix, .f = svm_error_rate)) %>%
  select(method, train_error, test_error)

# Display the results for the radial SVM: test error of 0.0808
radial_svm_results




# Candidate Model 2c: Polynomial SVM ---------------



# Use 10-fold CV to find the optimal parameter values for the polynomial kernel: cost, degree 

# Create a sequence of possible values for cost and degree
cost_options <- seq(0.01, 10, length = 10)
polynomial_options <- 2:5

# Use CV to find the optimal values of cost and degree
polynom_svm_cv <- svm_wft %>%
  mutate(cv_results = map(.x = train, 
                        .f = function(x) tune(svm, votedfortrump ~ ., data = x, kernel = "polynomial", 
                                              ranges = list(cost = cost_options, 
                                                            degree = polynomial_options))))

# Optimal tuning parameters from the CV: 1.12 (cost), 3 (degree).
# Best performance: CV error rate of 0.1014293 
polynom_svm_cv$cv_results[[1]] 

# Save the results of the radial SVM CV
write_rds(polynom_svm_cv$cv_results[[1]], "polynom_svm_cv.rds")

polynom_svm_cv <- read_rds("polynom_svm_cv.rds")

polynom_svm_cv

# Fit the candidate model (w best parameters) and store the results 
wft_polynom_svm <- svm_wft %>%
  mutate(model_fit = map(.x = train, 
                         .f = function(x) svm(votedfortrump ~ ., data = x, kernel = "polynomial", 
                                              cost = 1.12, degree = 3, probability = TRUE)),
         # Generate the predicted class of trump -- using the training and test sets 
         train_pred = map2(model_fit, train, predict, probability = TRUE), 
         test_pred = map2(model_fit, test, predict, probability = TRUE),
         # x$votedfortrump = actual outcomes, y = predicted outcomes
         train_conf_matrix = map2(.x = train, .y = train_pred,  # get confusion matrix for the training error
                                  .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)),
         # again, x$votedfortrump = actual outcomes, y = predicted outcomes
         test_conf_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix for the test error
                                 .f = function(x, y) caret::confusionMatrix(x$votedfortrump, y)))


# Generate and store the training/test error for the polynom SVM
polynom_svm_results <- wft_polynom_svm %>%
  select(train_conf_matrix, test_conf_matrix) %>%
  mutate(method = "polynom_svm",
         train_error = map_dbl(.x = train_conf_matrix, .f = svm_error_rate),
         test_error = map_dbl(.x = test_conf_matrix, .f = svm_error_rate)) %>%
  select(method, train_error, test_error)

# Display the results for the polynomial SVM - test error rate: 0.0975
polynom_svm_results


# Now, let's compare the results of the SVM candidate models -- using test error as the eval metric 

# Compare the SVM candidate models 
svm_results <- linear_svm_results %>%
  bind_rows(radial_svm_results) %>%
  bind_rows(polynom_svm_results) %>%
  select(-train_error)

svm_results <- svm_results %>% 
  arrange(test_error) 

# Save the results: 
write_rds(svm_results, "svm_results.rds")

svm_results %>%
  arrange(test_error) %>%
  knitr::kable(digits = 3)


# (3) Tree-based Candidate models -------------------------------------------------


# (3a) Candidate model: bagging 

# for the bagged forest model, the tuning parameter: number of trees (ntree)

# Set up a tibble with different values for ntree
model_def <- tibble(ntree = 1:100)

# returns a bagged forest where Trump vote is the outcome
fitbagging_trumpvote <- function(data, ntree){
  return(randomForest(formula = votedfortrump ~ ., data = data, 
                      mtry = length(train_set)-1, # mtry = num of predictors since bagging
                      ntree = ntree, importance = TRUE))
}

# Function to calculate RF error rate (applies for both RF and bagging)
error_rate_rf <- function(model, data){
  # Convert the resample object into a tibble
  as_tibble(data) %>% 
    # Generate the pred class 
    mutate(pred_votedfortrump = predict(model, newdata = data, type = "class"), 
           # Create a var that compares the predicted class to the actual/observed class
           error = pred_votedfortrump != votedfortrump) %>% 
    # extract the error vector and compute the rate
    pull(error) %>% 
    mean()
}

# Perform the 10-fold CV to find the optimal ntree
bagging_10fold <- train_set %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of ntree
  crossing(model_def) %>%
  # Fit the models and compute the fold error rate
  mutate(model_fit = map2(train, ntree, fitbagging_trumpvote),
         fold_error_rate = map2_dbl(model_fit, test, error_rate_rf))

# Save the results of the CV
write_rds(bagging_10fold, "bagging_cv.rds")

# Load the results of the CV
bagging_10fold <- read_rds("bagging_cv.rds")

# Display the results 
bagging_10fold %>% 
  group_by(ntree) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  arrange(error_rate)

# Plot the results: it looks like there are diminishing returns after ntree = ~50 
bagging_10fold %>% 
  group_by(ntree) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  ggplot(aes(x = ntree, y = error_rate)) +
  geom_line() +
  geom_point() +
  geom_smooth()



# Get the test error for the bagging candidate model (w/ best ntree)
bagging_test <- tibble(train = train_set %>% list(), 
                       test = test_set %>% list(),
                       # Set the optimal ntree from CV
                       ntree = 50,
                       method = "bagging") %>%
  mutate(model_fit = map2(train, ntree, fitbagging_trumpvote),
         test_error = map2_dbl(model_fit, test, error_rate_rf)) %>% 
  select(-ntree) %>%
  unnest(test_error, .drop = TRUE)

bagging_test


# (3b) Candidate model: RF -------------------------------------------------------------

# for the RF model: tuning parameters -- mtry, min node size

# Try tuning w tuneranger 

status.task = makeClassifTask(data = train_set, target = "votedfortrump")

estimateTimeTuneRanger(status.task)

res = tuneRanger(status.task, measure = list(acc), num.trees = 50,
                 tune.parameters = c("mtry", "min.node.size"), iters = 70)

#results: best mtry: 10, min node size: 14;
# best acc: 0.9085621
res 


# Set up a tibble with different values for mtry 
model_def <- tibble(mtry = 1:(ncol(train_set) - 1))

# Returns a random forest where status is the outcome
fitRF_trumpvote <- function(data, mtry){
  return(randomForest(formula = votedfortrump ~ ., data = as_tibble(data), 
                      ntree = 50,
                      mtry = mtry, 
                      replace = FALSE, # sample without replacement (to reduce overfitting)
                      nodesize = 14, # min. node size (bigger it is, less overfitting)
                      importance = TRUE))
}

# Note to self: don't do the stratified sampling, b/c it reduces accuracy 

# Perform the 10-fold CV to find the optimal mtry
rf_10fold <- train_set %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of mtry
  crossing(model_def) %>%
  # Fit the models and compute the fold error rate
  mutate(model_fit = map2(train, mtry, fitRF_trumpvote),
         fold_error_rate = map2_dbl(model_fit, test, error_rate_rf),
         importance = map(model_fit, randomForest::importance))

# save the results of the RF CV
write_rds(rf_10fold, "rf_cv.rds")

# load the RF CV
rf_10fold <- read_rds("rf_cv.rds")


# Display the results: best mtry = 10, best CV error rate: 0.0907 
rf_10fold %>% 
  group_by(mtry) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  arrange(error_rate) 

# Display the results -- 
rf_10fold %>% 
  group_by(mtry) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  arrange(error_rate) %>%
  ggplot(aes(x = mtry, y = error_rate)) +
  geom_point() +
  geom_line() +
  geom_smooth()



# What are the important variables? --- let's check


# custom function that adds the row name (var) as the var to the importance object
add_imp_vars <- function(matrix_obj){
  
  matrix_obj <- as.data.frame(matrix_obj)
  names_vec <- row.names(matrix_obj)
  
  # Add the vec of var names then save as a tibble
  matrix_obj$names_vec = names_vec
  as_tibble(matrix_obj) # Then convert df to a tibble
}

# Compute the avg var importance across the 500 estimates (from 10-fold CV)
rf_10fold %>%
  # Run the function that adds the var name to the impportance matrix
  mutate(imp_vars_tbl = map(importance, add_imp_vars)) %>%
  # Just filter in the imp vars tibbles, then unnest (combine) the tibbles
  select(imp_vars_tbl) %>%
  unnest(imp_vars_tbl) %>%
  # Group by var name, compute the avg importance metric for each var
  group_by(names_vec) %>%
  summarize(mean_MeanDecAcc = mean(MeanDecreaseAccuracy)) %>%
  # Arrange vars in order of their importance
  arrange(desc(mean_MeanDecAcc)) %>% View



# Get the test error for the RF candidate model (w/ best mtry of 10)
rf_test <- tibble(train = train_set %>% list(), 
                  test = test_set %>% list(),
                  # Set the optimal mtry based on 10-fold CV
                  mtry = 10,
                  method = "random_forest") %>%
  mutate(model_fit = map2(train, mtry, fitRF_trumpvote),
         test_error = map2_dbl(model_fit, test, error_rate_rf)) %>% 
  select(-mtry) %>%
  unnest(test_error, .drop = TRUE)

rf_test 


# Inspect the probabilities of voting for Trump -- for the model fit on the full training set 

# Function to calculate probabilities
return_prob_trumpvote <- function(model, test_data){
  
  as_tibble(test_data) %>% 
    # Generate the pred prob 
    mutate(prob_trumpvote = predict(model, newdata = test_data, type = "vote")[,2] %>% 
             base::round(digits = 3)) %>% # round the pred prob to 3 decimal places
    # Just filter in the 5 key predictors and the pred prob
    select(dem_ft, repub_ft, oppose_wall, conserv_scale, support_obamacare, prob_trumpvote)
}

prob_trump_vote <- tibble(train = train_set %>% list(), 
                          test = test_set %>% list(),
                          # Set the optimal mtry based on 10-fold CV
                          mtry = 10, 
                          method = "random_forest") %>%
                    mutate(model_fit = map2(train, mtry, fitRF_trumpvote), # fit the mod on full train set
                           pred_probs = map2(model_fit, test, return_prob_trumpvote)) # gen pred probs

# Save results
write_rds(prob_trump_vote, "prob_trump_vote.rds")

# Load results
prob_trump_vote <- read_rds("prob_trump_vote.rds") 

prob_trump_vote %>%
  select(pred_probs) %>%
  unnest(pred_probs) %>%
  filter(between(prob_trumpvote, .02, .08))

?predict 


# Get the var importance from the full training set
final_train_impvars <- tibble(train = train_set %>% list(), 
                  test = test_set %>% list(),
                  # Set the optimal mtry based on 10-fold CV
                  mtry = 10,
                  method = "random_forest") %>%
  mutate(model_fit = map2(train, mtry, fitRF_trumpvote),
         test_error = map2_dbl(model_fit, test, error_rate_rf),
         importance = map(model_fit, randomForest::importance),
         imp_vars_tbl = map(importance, add_imp_vars)) %>% 
  # Just filter in the imp vars tibbles, then unnest (combine) the tibbles
  select(imp_vars_tbl) %>%
  unnest(imp_vars_tbl) %>%
  # Group by var name, compute the avg importance metric for each var
  group_by(names_vec) %>%
  summarize(MeanDecrease_Acc = mean(MeanDecreaseAccuracy)) %>%
  # Arrange vars in order of their importance
  arrange(desc(MeanDecrease_Acc)) 

# Save the results (so that they're reproducible)
write_rds(final_train_impvars, "final_train_impvars.rds")

final_train_impvars <- read_rds("final_train_impvars.rds")

final_train_impvars %>% View

# (3c) Candidate model: boosted tree -------------------------------------------------------------

# for the RF model: tuning parameters -- learning rate, tree depth

# steps: 
# (1) set up the helper functions
# (2) do the CV to find the ideal parameters 
# (3) fit the final mod on the full training set --> and then generate predictions on the test set 

# Per the online guides, I need to first convert the factor variables to numeric labels (starting from 0)

xgb_train <- ohe_train_set %>%
  mutate(actual_label = as.numeric(votedfortrump)-1)

xgb_test <- ohe_test_set %>%
  mutate(actual_label = as.numeric(votedfortrump)-1)

# Set up a tibble with different combos of values for max depth and the learning rate 
max_depths <- tibble(max_depth = c(4, 5, 6, 7))

model_def <- tibble(learning_rate = round(seq(0.01, .2, length = 10), digits = 2)) %>%
  crossing(max_depths)

# Perform the 10-fold CV to find the optimal tuning parameters
boost_10fold <- xgb_train %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of eta
  crossing(model_def)

# Inspect
boost_10fold

# helper function that converts a df or resample obj to a matrix (for when status is the DV)
# Note: for this to work, the resample object has to be converted to a tibble using as_tibble()
make_xgb_matrix <- function(dat){
  
  # mat = a matrix of predictors 
  mat = as_tibble(dat) %>% 
    dplyr::select(-votedfortrump, -actual_label) %>% # Convert resample obj to a tibble and drop DVs
    as.matrix() # convert to a matrix before feeding into the xgb function
  
  val_labels = as_tibble(dat) %>% 
    pull(actual_label) %>% # Just extract the actual labels as a vec
    as.numeric() # To ensure that it's numeric
  
  return(xgb.DMatrix(data = mat, 
                     label = val_labels))
  
}

# create the wf tibble
wft_xgboost <- boost_10fold %>%
  mutate(# Create a var that represents the combo of the two parameters (will aggregate/group later by param combo)
    learning_rate = round(learning_rate, digits = 2), # round LR to 2 decimal points 
    param_combo = paste0(as.character(learning_rate), ", ", as.character(max_depth)),
    train_dg = map(train, make_xgb_matrix), 
    test_dg = map(test, make_xgb_matrix)) 

# Check contents
wft_xgboost


# Helper function to fit the mods
fit_xg_boost <- function(train_data, learning_rate, depth, nrounds, silent = 1){
  
  return(
    xgb.train(params = list(eta = learning_rate, 
                            max_depth = depth, 
                            silent = silent), 
              train_data, 
              nrounds = nrounds,
              # Specify that this is a classification tree
              objective = "binary:logistic")
              
    )
}


# Helper function to compute the boosting error rate
xg_error_rate <- function(model, test){
  
  actual_votedfortrump = getinfo(test, "label")
  pred_prob <- predict(model, test)
  error_rate <- as.numeric(sum(as.integer(pred_prob > 0.5) != actual_votedfortrump))/length(actual_votedfortrump)
  return(error_rate)
  
}


# Now, we can fit the mods 
wft_xgboost <- wft_xgboost %>%
  mutate(model_fit = pmap(.l = list(train_data = train_dg, learning_rate = learning_rate, 
                                    depth = max_depth, nrounds = 100),
                          .f = fit_xg_boost), 
         test_error = map2_dbl(model_fit, test_dg, xg_error_rate))

# Display the results of CV
wft_xgboost %>% 
  group_by(param_combo) %>% 
  summarize(test_error= mean(test_error)) %>%
  arrange(test_error)

# Save the CV results 
write_rds(wft_xgboost, "xgboost_cv.rds")

# Load the CV results
wft_xgboost <- read_rds("xgboost_cv.rds")

# Get the test error for the boosting candidate model (w/ best eta and max_depth)
boosting_test <- tibble(train = xgb_train %>% list(), 
                        test = xgb_test %>% list(),
                        best_eta = 0.09,
                        best_maxdepth = 4, 
                        method = "boosting") %>%
  # Convert the tibbles to xgb matrices
  mutate(train_dg = map(train, make_xgb_matrix), 
         test_dg = map(test, make_xgb_matrix)) %>%
  # Finally, fit the boosting mod and compute test error
  mutate(model_fit = pmap(.l = list(train_data = train_dg, learning_rate = best_eta, 
                                    depth = best_maxdepth, nrounds = 100),
                          .f = fit_xg_boost), 
         test_error = map2_dbl(model_fit, test_dg, xg_error_rate)) %>% 
  select(-best_eta, -best_maxdepth) %>%
  unnest(test_error, .drop = TRUE)

boosting_test



# Test errors combined and organized for the tree-based methods 
tree_results <- bagging_test %>%
  bind_rows(rf_test) %>% 
  bind_rows(boosting_test) %>% 
  arrange(test_error) %>%
  mutate(rank = row_number()) %>% 
  dplyr::select(rank, method, test_error) #%>%
  #knitr::kable(digits = 3)

# Save the results
write_rds(tree_results, "tree_results.rds")

tree_results %>% select(-rank) %>%
  knitr::kable(digits = 3)




