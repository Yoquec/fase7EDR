#################################################
# Created on Fri Jun 17 11:00:29 AM CEST 2022

# @file: fillGravityWindPower.R

# @author: Yoquec
##################################################

library(dplyr)
library(caret)
library(class)
library(doParallel)
library(MASS)


##########################################
# LOAD THE DATASETS

ag_data <- read.csv("data/CheckLandingV3.csv")
ag_data_test <- read.csv("data/CheckLandingV3test.csv")

# head(ag_data)
# head(ag_data_test)

# Combine both datasets to get more observations
# for the training of the KNN models
comb_data <- rbind(
                   ag_data %>% dplyr::select(-efficiency),
                   ag_data_test
)

# head(comb_data)

##########################################
# SEPARATING OBSERVATIONS INTO TRAIN AND TEST

missing_grav <- which(comb_data$gravity == 0)
missing_windpw <- which(comb_data$wind_power == 0)
# length(missing_grav)
# length(missing_windpw)
# head(comb_data[missing_grav, ])

# which have both missing?
missing_both <- which(comb_data$gravity == 0 & comb_data$wind_power == 0)
# length(missing_both)
# head(comb_data[missing_both, ])

"
As we have less missing observations inside wind_power, we will try
to predict it first in order to use it to predict both

Although we will introduce some more bias to the model, but it will be
helpfull to reduce the variance
"
##########################################
# BUILDING THE FIRST KNN MODEL (WIND POWER)

# Separate data obs from prediction obs
data_samples <- comb_data[-missing_windpw, ]
predict_samples <- comb_data[missing_windpw, ]

#Separate into train and test
set.seed(69420)
train_idx <- createDataPartition(
    data_samples$wind_power,
    p = 0.8,
    list = F
)

train_data <- data_samples[train_idx, ]
test_data <- data_samples[-train_idx, ]

# Prepare data for the knn build
	# We need to remember to exclude `gravity` as it has missing data
knn_bd_data <- train_data %>% dplyr::select(-X, -filename, -gravity)

# validation: repeated cross-validation
ctrl <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     number = 5)

# Prepare doParallel
detectCores()
n_workers <- 3
cl <- makeCluster(n_workers)
registerDoParallel(cl)

# Train our knn model with caret
# bd stands for "build", as in the model is still being 
# built and is still in a prototype fase
knn1_bd_model <- train(
    wind_power ~.,
    method = "knn",
    data = knn_bd_data,
    preProcess = c("center", "scale"),
    trControl = ctrl,
    tuneLength = 20
)

stopCluster(cl)

# summary(knn1_bd_model)
# names(knn1_bd_model)
# print(knn1_bd_model$results)

#-----------------------------------
# Regression tree test (to replace)
# knn
# library(rpart)

# # validation: repeated cross-validation
# ctrl <- trainControl(method = "repeatedcv",
#                      repeats = 3,
#                      number = 5)

# # Prepare doParallel
# n_workers <- 3
# cl <- makeCluster(n_workers)
# registerDoParallel(cl)

# tree_bd_model <- train(
# 											 wind_power ~.,
# 											 data = knn_bd_data,
# 											 method = "rpart",
# 											 preProcess = c("center", "scale"),
# 											 control = rpart.control(maxdepth = 6),
# 											 tuneLength = 20
# )

# stopCluster(cl)

# # summary(tree_bd_model)
# # names(tree_bd_model)
# # print(tree_bd_model$results)
# tree_optimal_cp <- as.numeric(tree_bd_model$bestTune[1])
# print(tree_bd_model$bestTune)


# tree_pt_model <- rpart(
# 											 wind_power ~.,
# 											 data = knn_bd_data,
# 											 tuneLength = 20
# )
#-----------------------------------



##########################################
# CHECKING THE ACCURACY OF OUR PROTOTYPE

	# Retrieve the optimal k from the training
knn1_optimalK <- as.numeric(knn1_bd_model$bestTune[1])

# Function to scale the testing data the same way as in the training set
# prepareTestingData <- function(testdf, cnt = knn_bd_center, scl = knn_bd_scale) {
# 	return((testdf - cnt) / scl)
# }

# Retrieve the prototype model from caret
# (pt stands for 'prototype')
knn1_pt_model <- knn1_bd_model$finalModel

# Generate testing data from caret
knn1_pt_testdata <- test_data %>% dplyr::select(-X, -filename, -gravity)

# Get the predictions
knn1_pt_pred <- predict(
												knn1_pt_model,
												knn1_pt_testdata %>% dplyr::select(-wind_power)
)

# Get Absolute error of a prediction
computeABSerr <- function(true, pred) {
	return(sum(abs(true - pred)) / length(true))
}

# Mean error of the model:
"
Tenemos un error medio de 5,33 cosa que se puede mejorar significativamente.
A lo mejor implementando un regression tree sale mejor, pero al final del dia
hay que recordar que esto es solo para rellenar NA's y de por sí es una mejora
a lo que ya tenemos
"
head(knn1_pt_testdata$wind_power)
head(knn1_pt_pred)
print(computeABSerr(knn1_pt_testdata$wind_power, knn1_pt_pred))

#-----------------------------------
# Regression tree predictions
# tree_pt_model <- tree_bd_model$finalModel
# tree_pt_pred <- as.vector(predict(
# 												tree_pt_model,
# 												knn1_pt_testdata %>% select(-wind_power)
# ))

# head(knn1_pt_testdata$wind_power)
# head(tree_pt_pred)
# summary(tree_pt_pred)

# library(rpart.plot)
# prp(tree_pt_model)
# print(computeABSerr(knn1_pt_testdata$wind_power, tree_pt_pred))
#-----------------------------------

##########################################
# BUILDING THE FINAL KNN MODEL
print(knn1_optimalK)

ctrl <- trainControl(method = "none")

# Build the final "fn" model
knn1_crt <- train(
    wind_power ~.,
    method = "knn",
    data = data_samples %>% dplyr::select(-X, -filename, -gravity),
    preProcess = c("center", "scale"),
    trControl = ctrl,
    tuneGrid = data.frame(k = knn1_optimalK)
)

knn1_final <- knn1_crt$finalModel
(knn1_final)

# Save the KNN model
saveRDS(knn1_final, "wind_powerKnn.RDS")


##########################################
# PREDICTING MISSING VALUES
# using the Knn model

# Get missing indeces
idx_ms_wp <- which(ag_data$wind_power == 0)
idx_ms_wp_test <- which(ag_data_test$wind_power == 0)

# Predict missnig wp in the training set
miss_wind_wp <- predict(knn1_final, ag_data[idx_ms_wp, ] %>%
												dplyr::select(-X, -filename, -gravity, -efficiency, -wind_power))

# Predict missnig wp in the testing set
miss_wind_wp_test <- predict(knn1_final, ag_data_test[idx_ms_wp_test, ] %>%
														 dplyr::select(-X, -filename, -gravity, -wind_power))

# Record the new values into the datasets
ag_data$wind_power[idx_ms_wp] <- miss_wind_wp
ag_data_test$wind_power[idx_ms_wp_test] <- miss_wind_wp_test

##########################################
# PREPARE DATA FOR THE SECOND KNN

# update comb_data with the updated dataframes
comb_data <- rbind(
									 ag_data %>% dplyr::select(-efficiency),
									 ag_data_test
)

################################################
# BUILDING THE SECOND KNN (GRAVITY)

# Separate data observations from prediction ones
data_samples_2  <- comb_data[-missing_grav,]
predict_samples_2  <- comb_data[missing_grav,]

# Separate into train and test sets
set.seed(672)
train_idx_2  <- createDataPartition(
  data_samples_2$gravity,
  p = 0.8,
  list = F
)

train_data_2  <- data_samples_2[train_idx_2,]
test_data_2  <- data_samples_2[-train_idx_2,]

# Prepare the data for the KNN
# Try with the generated values for `wind_power`
knn_2_bd_data  <- train_data_2 %>% dplyr::select(-X, -filename)

# Repated Cross-Validation
ctrl_2  <- trainControl(method = "repeatedcv",
                        repeats = 3,
                        number = 5)

# Prepare doParallel
n_workers_2  <- 3
cl_2  <- makeCluster(n_workers_2)
registerDoParallel(cl_2)

# Train the KNN with the caret Package
knn2_bd_model  <- train(gravity ~ .,
                        method = "knn",
                        data = knn_2_bd_data,
                        preProcess = c("center", "scale"),
                        trControl = ctrl_2,
                        tuneLength = 20)

stopCluster(cl_2)

# CHECKING THE ACCURACY OF THE SECOND KNN PROTOTYPE
# Retrieve the optimal k from the training
knn2_optimal_K  <- as.numeric(knn2_bd_model$bestTune[1])

# Retrieve the KNN protoype model from caret
knn2_pt_model  <- knn2_bd_model$finalModel

# Generate the testing data from caret
knn2_pt_testdata  <- test_data_2 %>% dplyr::select(-X, -filename)

# Get the predictions
knn2_pt_pred  <- predict(knn2_pt_model,
                         knn2_pt_testdata %>% dplyr::select(-gravity))

head(knn2_pt_testdata$gravity)
head(knn2_pt_pred)
print(computeABSerr(knn2_pt_testdata$gravity, knn2_pt_pred))

###########################################
# BUILDING THE FINAL SECOND KNN MODEL
print(knn2_optimal_K)

ctrl_2 <- trainControl(method = "none")

# Build the second final model
knn2_crt <- train(
  gravity ~.,
  method = "knn",
  data = data_samples %>% dplyr::select(-X, -filename),
  preProcess = c("center", "scale"),
  trControl = ctrl_2,
  tuneGrid = data.frame(k = knn2_optimal_K)
)

knn2_final <- knn2_crt$finalModel
(knn2_final)

# Save the KNN model
saveRDS(knn2_final, "gravityKnn.RDS")


##########################################
# PREDICTING MISSING VALUES
# using the model

# Get missing indexes
idx_ms_gravity <- which(ag_data$gravity == 0)
idx_ms_gravity_test <- which(ag_data_test$gravity == 0)

# Predict missing gravity in the training set
miss_gravity <- predict(knn2_final, ag_data[idx_ms_gravity, ] %>%
                          dplyr::select(-X, -filename, -gravity, -efficiency))

# Predict missing gravity in the testing set
miss_gravity_test <- predict(knn2_final, ag_data_test[idx_ms_gravity_test, ] %>%
                               dplyr::select(-X, -filename, -gravity))

# Record the new values into the datasets
ag_data$gravity[idx_ms_gravity] <- miss_gravity
ag_data_test$gravity[idx_ms_gravity_test] <- miss_gravity_test


#mod_bic <- MASS::stepAIC(lm(Salary ~ ., data = Hitters), k = log(nrow(Hitters)), trace = 0)
#modLasso <- glmnet(y = Y, x = X, alpha = 1, lambda = 0.5)
#modRidge = cv.glmnet(x = X, y = Y, alpha = 0, nfolds = n)
#car::vif(mod)
#sort(car::vif(mod_bth), decreasing = TRUE)
#sort(car::vif(mod_bth_int), decreasing = TRUE)

################################################
# REGRESSION MODEL 
  # Using the BIC and VIF (Variance Inflation Factor)

n_workers_3  <- 3
cl_3 <- makeCluster(n_workers_3)
registerDoParallel(cl_3)

#mod_all <- lm(efficiency ~.^2, data = ag_data)
mod_zero <- lm(efficiency ~1, data = ag_data)
mod_bth_int <- MASS::stepAIC(object = mod_zero, k = log(nrow(ag_data)),
                             scope = list(lower = mod_zro, upper = mod_all),
                            direction = "both")

stopCluster(cl_3)
