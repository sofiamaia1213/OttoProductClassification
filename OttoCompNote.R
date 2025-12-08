# =========================================
# Otto Product Classification - LightGBM
# =========================================

# --------------------------
# 1. Introduction 
# --------------------------
# This notebook aims to classify products into one of 9 classes 
# based on the Otto Group Product Classification Challenge.
# We will load the data, perform minimal feature engineering, 
# train a LightGBM model using tidymodels, tune hyperparameters,
# and generate predictions for submission.

# --------------------------
# 2. Load Libraries
# --------------------------
library(tidyverse)
library(tidymodels)
library(vroom)
library(lightgbm)
library(patchwork)
library(GGally)
library(glmnet)
library(bonsai)
library(h2o)
library(kknn)
library(discrim)
library(kernlab)
library(naivebayes)
library(themis)
library(beepr)

# --------------------------
# 3. Load Data
# --------------------------
trainData <- vroom("GitHub/OttoProductClassification/train.csv")
testData  <- vroom("GitHub/OttoProductClassification/test.csv")

train_ids <- trainData$id
test_ids  <- testData$id

# Remove ID columns for modeling
trainData <- trainData %>% select(-id)
testData  <- testData %>% select(-id)

# Convert target to factor with correct levels
trainData$target <- factor(
  trainData$target,
  levels = paste0("Class_", 1:9)
)

# --------------------------
# 4. Feature Engineering 
# --------------------------
# For this notebook, we perform minimal feature engineering:
# - No missing values in dataset
# - All features are numeric
# - Target is converted to a factor

# --------------------------
# 5. Model Description & Parameter Tuning 
# --------------------------
# We will use LightGBM with tidymodels:
# - Boosted decision tree model
# - Hyperparameters tuned: tree_depth, learn_rate
# - Metric: Multinomial log loss (mn_log_loss)

# Recipe
otto_recipe <- recipe(target ~ ., data = trainData)

# Model specification
lgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Workflow
lgb_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(lgb_spec)

# Cross-validation folds
# We use 2 fold cross-validation to estimate model performance.
# Using only 2 folds helps the model run faster
folds <- vfold_cv(trainData, v = 2)

# Hyperparameter grid
# - tree_depth: from 4 to 14
# - learn_rate: from 0.001 to 0.1 (log scale, expressed as -3 to -1 in log10)
# Latin hypercube sampling allows us to explore the parameter space efficiently
lgb_grid <- grid_latin_hypercube(
  tree_depth(range = c(4L, 14L)),
  learn_rate(range = c(-3, -1)), # log scale
  size = 5
)

# Tune model
lgb_tuned <- tune_grid(
  lgb_wf,
  resamples = folds,
  grid = lgb_grid,
  metrics = metric_set(mn_log_loss),
  control = control_grid(verbose = TRUE, save_pred = FALSE)
)

# Select best hyperparameters
# Lower values are better
best_params <- select_best(lgb_tuned, metric = "mn_log_loss")

# Finalize workflow
final_lgb <- finalize_workflow(lgb_wf, best_params)

# Fit final model on full training data
final_fit <- final_lgb %>% fit(data = trainData)

# --------------------------
# 6. Generating Predictions
# --------------------------
# Generate predicted probabilities for the test set
pred_probs <- predict(final_fit, new_data = testData, type = "prob")

# Fix column names
colnames(pred_probs) <- gsub("^\\.pred_", "", colnames(pred_probs))

# Ensure column order matches submission format
pred_probs <- pred_probs %>%
  select(Class_1, Class_2, Class_3, Class_4, Class_5,
         Class_6, Class_7, Class_8, Class_9)

# Prepare submission
submission <- bind_cols(id = test_ids, pred_probs)

# Write submission CSV
vroom_write(submission, "GitHub/OttoProductClassification/lgbm.csv", delim = ",")

# Play a sound to indicate completion
beepr::beep(2)
