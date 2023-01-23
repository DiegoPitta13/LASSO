library(data.table)
library(ggplot2)
library(glmnet)
library(randomForest)
library(Metrics)
library(MLmetrics)
library(e1071)
library(forecastML)

#Defini??o do diret?rio de trabalho
setwd("C:/Users/marcu/Desktop")
dir()

#Importanto base de dados

dados<-read.table("labimec_PIB_T.txt", head=T)

data_train <- forecastML::create_lagged_df(dados, type = "train", method = "direct",
                                           outcome_col = 1, lookback = 1:12, horizon = 1:12)

windows <- forecastML::create_windows(data_train, window_length = 0)

model_fun <- function(data) {
  x <- as.matrix(data[, -1, drop = FALSE])
  y <- as.matrix(data[, 1, drop = FALSE])
  model <- glmnet::cv.glmnet(x, y)
}

model_results <- forecastML::train_model(data_train, windows, model_name = "LASSO", model_function = model_fun)

prediction_fun <- function(model, data_features) {
  data_pred <- data.frame("y_pred" = predict(model, as.matrix(data_features)),
                          "y_pred_lower" = predict(model, as.matrix(data_features)) - 30,
                          "y_pred_upper" = predict(model, as.matrix(data_features)) + 30)
}

data_forecast <- forecastML::create_lagged_df(dados, type = "forecast", method = "direct",
                                              outcome_col = 1, lookback = 1:12, horizon = 1:12)

data_forecasts <- predict(model_results, prediction_function = list(prediction_fun), data = data_forecast)

data_forecasts <- forecastML::combine_forecasts(data_forecasts)

plot(data_forecasts, data_actual = dados[-(1:100), ], actual_indices = (1:nrow(dados))[-(1:100)])

data_forecasts