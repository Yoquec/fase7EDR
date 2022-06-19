f#################################################
# Created on Fri Jun 17 11:00:29 AM CEST 2022

# @file: fillGravityWindPower.R

# @author: Yoquec
##################################################

library(dplyr)
library(caret)
library(class)
library(doParallel)

##########################################
# LOAD THE DATASETS

ag_data <- read.csv("data/CheckLandingV3.csv")
ag_data_test <- read.csv("data/CheckLandingV3test.csv")

# head(ag_data)
# head(ag_data_test)

# Combine both datasets to get more observations
# for the training of the KNN models
comb_data <- rbind(
									 ag_data %>% select(-efficiency),
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
knn_bd_data <- train_data %>% select(-X, -filename, -gravity)

# validation: repeated cross-validation
ctrl <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     number = 5)

# Prepare doParallel
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
knn1_pt_testdata <- test_data %>% select(-X, -filename, -gravity)

# Get the predictions
knn1_pt_pred <- predict(
												knn1_pt_model,
												knn1_pt_testdata %>% select(-wind_power)
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
