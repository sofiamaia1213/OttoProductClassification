library(tidyverse)
library(patchwork)
library(ggplot2)
library(tidymodels)
library(vroom)
library(GGally)
library(rpart)
library(glmnet)
library(bonsai)
library(lightgbm)
library(agua)
library(h2o)
library(kknn)
library(discrim)
library(kernlab)
library(naivebayes)
library(themis)
library(beepr)



#########################################################################
############################## H2O ######################################
#########################################################################

# Sys.setenv(JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-25.0.0.36-hotspot")
# h2o::h2o.init()

# trainData <- vroom("GitHub/OttoProductClassification/train.csv")
# testData  <- vroom("GitHub/OttoProductClassification/test.csv")
# 
# # Save IDs and remove them
# train_ids <- trainData$id
# test_ids  <- testData$id
# trainData <- trainData %>% select(-id)
# testData  <- testData %>% select(-id)

# # Ensure proper factor levels
# trainData$target <- factor(trainData$target, levels = paste0("Class_", 1:9))

# otto_recipe <- recipe(target ~ ., data = trainData)

# auto_model <- auto_ml() %>%
#   set_engine(
#     "h2o",
#     max_runtime_secs = 300,
#     max_models = 50,
#     seed = 17,
#     stopping_metric = "logloss"  # correct for multi-class
#   ) %>%
#   set_mode("classification")

# automl_wf <- workflow() %>%
#   add_recipe(otto_recipe) %>%
#   add_model(auto_model)

# final_fit <- fit(automl_wf, data = trainData)

# pred_probs <- predict(final_fit, new_data = testData, type = "prob")
# 
# # Clean column names to match Kaggle format
# colnames(pred_probs) <- gsub("^\\.pred_", "", colnames(pred_probs))
# pred_probs <- pred_probs %>%
#   select(Class_1, Class_2, Class_3, Class_4, Class_5,
#          Class_6, Class_7, Class_8, Class_9)

# submission <- bind_cols(id = test_ids, pred_probs)

# # Write CSV
# vroom_write(submission,
#             "GitHub/OttoProductClassification/H2O.csv",
#             delim = ",")


#########################################################################
############################## RF #######################################
#########################################################################

# trainData <- vroom("GitHub/OttoProductClassification/train.csv")
# testData  <- vroom("GitHub/OttoProductClassification/test.csv")

# # Save IDs and remove them
# train_ids <- trainData$id
# test_ids  <- testData$id
# trainData <- trainData %>% select(-id)
# testData  <- testData %>% select(-id)

# # Factorize target
# trainData$target <- factor(trainData$target, levels = paste0("Class_", 1:9))

# # Recipe
# otto_recipe <- recipe(target ~ ., data = trainData)

# # Random Forest spec 
# rf_spec <- rand_forest(
#   mtry = 10,       
#   min_n = 5,       
#   trees = 500
# ) %>%
#   set_engine("ranger", importance = "impurity") %>%
#   set_mode("classification")

# # Workflow
# rf_wf <- workflow() %>%
#   add_recipe(otto_recipe) %>%
#   add_model(rf_spec)

# # Fit final model
# final_fit <- rf_wf %>% fit(data = trainData)

# # Predict probabilities
# pred_probs <- predict(final_fit, new_data = testData, type = "prob")

# # Fix column names and order
# colnames(pred_probs) <- gsub("^\\.pred_", "", colnames(pred_probs))
# pred_probs <- pred_probs %>% select(
#   Class_1, Class_2, Class_3, Class_4, Class_5,
#   Class_6, Class_7, Class_8, Class_9
# )

# # Build submission
# submission <- bind_cols(id = test_ids, pred_probs)

# vroom_write(submission, "GitHub/OttoProductClassification/rf_no_tune.csv", delim = ",")

#########################################################################
##############################Light GBM##################################
#########################################################################

 trainData <- vroom("GitHub/OttoProductClassification/train.csv")
 testData  <- vroom("GitHub/OttoProductClassification/test.csv")
 train_ids <- trainData$id
 test_ids  <- testData$id
#
 trainData <- trainData %>% select(-id)
 testData  <- testData %>% select(-id)

 trainData$target <- factor(
   trainData$target,
   levels = paste0("Class_", 1:9)
 )

 otto_recipe <- recipe(target ~ ., data = trainData)

 lgb_spec <- boost_tree(
   trees = 500,
   tree_depth = tune(),
 learn_rate = tune(),
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

  lgb_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(lgb_spec)

folds <- vfold_cv(trainData, v = 2)

lgb_grid <- grid_latin_hypercube(
   tree_depth(range = c(4L, 14L)),
  learn_rate(range = c(-3, -1)),   
   size = 5
 )

 lgb_tuned <- tune_grid(
   lgb_wf,
  resamples = folds,
   grid = lgb_grid,
   metrics = metric_set(mn_log_loss),
  control = control_grid(verbose = TRUE, save_pred = FALSE)
 )

 best_params <- select_best(lgb_tuned, metric = "mn_log_loss")

final_lgb <- finalize_workflow(lgb_wf, best_params)


 final_fit <- final_lgb %>% fit(data = trainData)

pred_probs <- predict(final_fit, new_data = testData, type = "prob")

 colnames(pred_probs) <- gsub("^\\.pred_", "", colnames(pred_probs))

 pred_probs <- pred_probs %>%
   select(Class_1, Class_2, Class_3, Class_4, Class_5,
          Class_6, Class_7, Class_8, Class_9)

submission <- bind_cols(id = test_ids, pred_probs)
 vroom_write(submission, "GitHub/OttoProductClassification/lgbm.csv", delim = ",")
