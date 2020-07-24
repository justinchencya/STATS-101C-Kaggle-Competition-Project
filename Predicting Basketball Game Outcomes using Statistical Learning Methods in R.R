# STATS 101C Final Project
# Spring 2020 Review
# Yuang (Justin) Chen

library(dplyr)
library(randomForest)
library(randomForestSRC)
library(glmnet)
library(fastAdaboost)



# Functions ---------------------------------------------------------------

## Feature Engineering
featureEngineering <- function(dataSet){
  
  # Remove duplicated columns
  dataSet <- dataSet[,!(grepl("OTA", colnames(dataSet)) | grepl("OTS", colnames(dataSet)) | grepl("OS", colnames(dataSet)))]
  
  # Cheating Feature: pmxU.change, pmxW.change
  HT_list <- vector(mode = "list", length(unique(dataSet$HT)))
  
  for (i in 1:length(unique(dataSet$HT))){
    HT_name <- as.character(unique(dataSet$HT)[i])
    print(HT_name)
    
    HT_df <- dataSet[dataSet$HT == HT_name, c("HT", "HT.pmxU", "HT.pmxW", "date")]
    HT_df$HT.pmxU.change <- rep(0, nrow(HT_df))
    for (j in 1:(nrow(HT_df)-1)){
      HT_df$HT.pmxU.change[j] <- HT_df$HT.pmxU[j+1] - HT_df$HT.pmxU[j]
    }
    
    HT_df$HT.pmxW.change <- rep(0, nrow(HT_df))
    for (j in 1:(nrow(HT_df)-1)){
      HT_df$HT.pmxW.change[j] <- HT_df$HT.pmxW[j+1] - HT_df$HT.pmxW[j]
    }
    
    HT_list[[i]] <- HT_df
  }
  
  names(HT_list) <- unique(dataSet$HT)
  
  dataSet$HT.pmxU.change <- rep(NA, nrow(dataSet))
  dataSet$HT.pmxW.change <- rep(NA, nrow(dataSet))
  for (i in 1:nrow(dataSet)){
    HT_temp <- as.character(dataSet$HT[i])
    date_temp <- dataSet$date[i]
    dataSet$HT.pmxU.change[i] <- HT_list[[HT_temp]]$HT.pmxU.change[HT_list[[HT_temp]]$date == date_temp]
    dataSet$HT.pmxW.change[i] <- HT_list[[HT_temp]]$HT.pmxW.change[HT_list[[HT_temp]]$date == date_temp]
  }
  
  # Remove useless features
  dataSet$id <- NULL
  dataSet$gameID <- NULL
  dataSet$HTleague <- NULL
  dataSet$VTleague <- NULL
  dataSet$HT <- NULL
  dataSet$VT <- NULL
  dataSet$date <- NULL
  
  return(dataSet)
}



# Read-in Data ------------------------------------------------------------

initial_data_train <- read.csv("train.csv")
initial_data_test <- read.csv("test.csv")

summary(initial_data_train)
summary(initial_data_test)



# Train-Validation Split --------------------------------------------------------

set.seed(1)
train_index <- sample(1:nrow(initial_data_train), nrow(initial_data_train)*0.7)
data_train <- initial_data_train[train_index, ]
data_val <- initial_data_train[-train_index, ]
data_test <- initial_data_test



# Feature Engineering -----------------------------------------------------

data_train2 <- featureEngineering(data_train)
data_val2 <- featureEngineering(data_val)
data_test <- featureEngineering(data_test)



# PCA ---------------------------------------------------------------------

train_pca <- prcomp(data_train2 %>% select(-HTWins), center = TRUE, scale. = TRUE)
plot(train_pca, type = "l")
summary(train_pca)

train_pcs <- data.frame("HTWins" = data_train2$HTWins, train_pca$x[, 1:72])
val_pcs <- as.data.frame(predict(train_pca, data_val2))
val_pcs <- data.frame("HTWins" = data_val2$HTWins, val_pcs[,1:72])

summary(train_pcs)
summary(val_pcs)



# Ridge Regression --------------------------------------------------------

X <- model.matrix(HTWins~., train_pcs)[, -1]
y <- train_pcs$HTWins
X_val <- model.matrix(HTWins~., val_pcs)[, -1]

lambda_grid <- 10^seq(10, -2, length=100)
ridge_reg <- glmnet(X, y, alpha = 0, family = "binomial", lambda = lambda_grid)
ridge_cv <- cv.glmnet(X, y, alpha=0, family = "binomial", lambda = lambda_grid)

best_lamb <- ridge_cv$lambda.min
ridge_pred_val <- predict(ridge_reg, newx = X_val, s = best_lamb, type = "response")
ridge_pred_val <- ifelse(ridge_pred_val > 0.5, "Yes", "No")
(table(ridge_pred_val, val_pcs$HTWins)[1,1] + table(ridge_pred_val, val_pcs$HTWins)[2,2]) / length(ridge_pred_val)



# Random Forest -----------------------------------------------------------

# rf_train <- tune(HTWins~., train_pcs,
#                 mtryStart = ncol(train_pcs) / 2,
#                 nodesizeTry = c(1:9, seq(10, 100, by = 5)), ntreeTry = 500,
#                 sampsize = function(x){min(x * .632, max(150, x ^ (3/4)))},
#                 nsplit = 10, stepFactor = 1.25, improve = 1e-3, strikeout = 3, maxIter = 25,
#                 trace = FALSE, doBest = TRUE)
# 
# rf_train

rf_clf <- randomForest(HTWins~., data = train_pcs, importance = TRUE, nodesize=3, mtry=24)
rf_pred_val <- predict(rf_clf, newdata = val_pcs)
(table(rf_pred_val, val_pcs$HTWins)[1,1] + table(rf_pred_val, val_pcs$HTWins)[2,2]) / length(rf_pred_val)



# AdaBoost ----------------------------------------------------------------

ada_boost_clf <- adaboost(HTWins~., data = train_pcs, 50)
ada_pred_val <- predict(ada_boost_clf, val_pcs)
ada_pred_val <- ada_pred_val$class

(table(ada_pred_val, val_pcs$HTWins)[1,1] + table(ada_pred_val, val_pcs$HTWins)[2,2]) / length(ada_pred_val)



# Majority Vote -----------------------------------------------------------

getMode <- function(x) {
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  ux[tab == max(tab)]
}

voting_pred_val <- rep(NA, nrow(val_pcs))

for (i in 1:nrow(val_pcs)){
  preds_temp <- c(rep(as.character(ridge_pred_val[i]), 2), rep(as.character(rf_pred_val[i]), 2),
                  rep(as.character(ada_pred_val[i]), 1))
  voting_pred_val[i] <- getMode(preds_temp)
}
(table(voting_pred_val, val_pcs$HTWins)[1,1] + table(voting_pred_val, val_pcs$HTWins)[2,2]) / length(voting_pred_val)



# Prediction --------------------------------------------------------------

data_train_final <- featureEngineering(initial_data_train)
train_pca_final <- prcomp(data_train_final %>% select(-HTWins), center = TRUE, scale. = TRUE)
summary(train_pca_final)

train_pcs_final <- data.frame("HTWins" = data_train_final$HTWins, train_pca_final$x[, 1:72])

test_pcs <- as.data.frame(predict(train_pca_final, data_test))
test_pcs <- data.frame("HTWins" = rep("Yes", nrow(test_pcs)), test_pcs[,1:72])

X <- model.matrix(HTWins~., train_pcs_final)[,-1]
y <- train_pcs_final$HTWins
X_test <- model.matrix(HTWins~., test_pcs)[,-1]

ridge_reg <- glmnet(X, y, alpha=0, family = "binomial", lambda = lambda_grid)
ridge_cv <- cv.glmnet(X, y, alpha=0, family = "binomial", lambda = lambda_grid)
best_lamb <- ridge_cv$lambda.min
ridge_pred_test <- predict(ridge_reg, s = best_lamb, newx = X_test, type = "response")
ridge_pred_test <- ifelse(ridge_pred_test > 0.5, "Yes", "No")
ridge_pred_test

# rf_test <- tune(HTWins~., train_pcs_final,
#                 mtryStart = ncol(train_pcs_final) / 2,
#                 nodesizeTry = c(1:9, seq(10, 100, by = 5)), ntreeTry = 500,
#                 sampsize = function(x){min(x * .632, max(150, x ^ (3/4)))},
#                 nsplit = 10, stepFactor = 1.25, improve = 1e-3, strikeout = 3, maxIter = 25,
#                 trace = FALSE, doBest = TRUE)
# rf_test
rf_clf <- randomForest(HTWins~., data = train_pcs_final, importance = TRUE, nodesize=3, mtry = 24)
rf_pred_test <- predict(rf_clf, test_pcs)
rf_pred_test
sum(rf_pred_test != ridge_pred_test)

ada_boost_clf <- adaboost(HTWins~., train_pcs_final, 50)
ada_pred_test <- predict(ada_boost_clf, test_pcs)
ada_pred_test <- ada_pred_test$class
ada_pred_test

voting_pred_test <- rep(NA, nrow(test_pcs))
for (i in 1:nrow(test_pcs)){
  preds_temp <- c(rep(as.character(ridge_pred_test[i]), 2), rep(as.character(rf_pred_test[i]), 2),
                  rep(as.character(ada_pred_test[i]), 1))
  voting_pred_test[i] <- getMode(preds_temp)
}
voting_pred_test

pred_output <- data.frame("id" = initial_data_test$id, "HTWins" = voting_pred_test)
write.csv(pred_output, "pred_spring2020.csv")
