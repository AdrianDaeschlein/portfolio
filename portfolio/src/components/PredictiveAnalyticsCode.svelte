<script lang="ts">
  import { onMount } from 'svelte';

  let highlightedCode = '';

  // Prism.js for syntax highlighting
  onMount(async () => {
    const Prism = await import('prismjs');
    await import('prismjs/themes/prism-okaidia.css');
    await import('prismjs/components/prism-r.js'); // For R language
    highlightedCode = Prism.highlight(code, Prism.languages.r, 'r');
  });

  const code = `
  ## ----setup, include=FALSE----------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = FALSE)
library(readr)
library(forecast)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(tseries)
library(urca)
library(dplyr)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
data <- read_csv("american_weekly_avg_delay.csv")


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
missing_values <- sum(is.na(data$Avg_ARR_DELAY))
print(missing_values)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
data$Avg_ARR_DELAY <- zoo::na.approx(data$Avg_ARR_DELAY)

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
missing_values <- sum(is.na(data$Avg_ARR_DELAY))
print(missing_values)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
head(data)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
ggplot(data, aes(x = FL_DATE, y = Avg_ARR_DELAY)) +
  geom_line(color = "blue")
  labs(title = "Average Arrival Delay Over Time",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal()


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
filtered_data <- data %>% filter(FL_DATE >= as.Date("2021-01-01"))
head(filtered_data)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
ggplot(filtered_data, aes(x = FL_DATE, y = Avg_ARR_DELAY)) +
  geom_line(color = "blue") +
  labs(title = "Average Arrival Delay Over Time (from 2021-01-01)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal()

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
filtered_data_ts <- ts(filtered_data$Avg_ARR_DELAY, start = c(2021, 1), frequency = 52)

# Plot seasonal plot
ggseasonplot(filtered_data_ts, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("Average Arrival Delay (minutes)") +
  ggtitle("Seasonality of Average Arrival Delay (Last 3 Years)")

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Compute the mean of the Avg_ARR_DELAY column
mean_delay <- mean(data$Avg_ARR_DELAY, na.rm = TRUE)

# Compute the standard deviation of the Avg_ARR_DELAY column
sd_delay <- sd(data$Avg_ARR_DELAY, na.rm = TRUE)
print(mean_delay)
print(sd_delay)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Add a constant to make all values positive (e.g., shift by adding the absolute value of the minimum plus a small constant)
shift_value <- abs(min(filtered_data$Avg_ARR_DELAY)) + 2
filtered_data$Shifted_Log_Avg_ARR_DELAY <- log(filtered_data$Avg_ARR_DELAY + shift_value)

# Plot the shifted log-transformed data
ggplot(filtered_data, aes(x = FL_DATE, y = Shifted_Log_Avg_ARR_DELAY)) +
  geom_line(color = "blue") +
  labs(title = "Shifted Log-Transformed Average Arrival Delay Over Time (from 2021-09-05)",
       x = "Date",
       y = "Shifted Log of Average Arrival Delay (minutes)") +
  theme_minimal()


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# decompose
# Decomposition for Data
decomp <- decompose(filtered_data_ts) # -> Strong trend component, existing but minor seasonality component, significant random component
plot(decomp)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# KPSS Test for Data (H0: Series stationary (has no unit root) / H1: Series non-stationary (has a unit root))
kpss <- ur.kpss(filtered_data_ts, type = "tau", lags = "short")
summary(kpss) # -> Results show data is stationary with high confidence

# ADF Test for Data (H0: Series non-stationary (has a unit root) / H1: Series stationary (has no unit root))
adf <- ur.df(filtered_data_ts, type = "trend", lags = 0) 
summary(adf) # -> Results show data is stationary with high confidence


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the row index to stop for training
train_length <- floor(0.8 * length(filtered_data_ts))

# Split the data into training and test sets for ETS and ARIMA
train_data <- ts(filtered_data_ts[1:train_length], start = start(filtered_data_ts), frequency = frequency(filtered_data_ts))
test_data <- ts(filtered_data_ts[(train_length + 1):length(filtered_data_ts)], start = time(filtered_data_ts)[train_length + 1], frequency = frequency(filtered_data_ts))


## 3.1) Fitting the ETS
# Fitting the ETS Model on the log-transformed but not detrended series
ets_AAA <- ets(train_data, model = "AAN") # Additive error, additive trend, Additive seasonality
summary(ets_AAA)

# Fitting Auto ETS Model
best_ets_model <- ets(train_data)
summary(best_ets_model)
print("--")
## 3.2) Fitting the ARIMA
# Plot ACF and PACF for the detrended log-transformed BTC data
par(mfrow = c(2, 1))
acf(filtered_data_ts, main = "ACF")
pacf(filtered_data_ts, main = "PACF")

# Fit ARIMA(3, 0, 0)(1, 1, 0)[12] model
arima_model <- arima(train_data, order = c(1, 0, 0), seasonal = list(order = c(0, 1, 0), period = 52))
summary(arima_model)

# Automatically select the best ARIMA model
best_arima_model <- auto.arima(train_data)
summary(best_arima_model) # -> ARIMA(0,0,1)(0,1,0)[52] with drift


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## 4.1) ETS Model
# Extract residuals from the fitted ETS model
residuals_ets <- residuals(best_ets_model)

# Plot ACF of residuals
acf(residuals_ets, main = "ACF of ETS Model Residuals")

# Plot histogram of residuals
hist(residuals_ets, main = "Histogram of ETS Model Residuals", xlab = "Residuals")

# QQ-plot of residuals
qqnorm(residuals_ets)
qqline(residuals_ets, col = "red")

# Perform Ljung-Box test on residuals
ljung_box_test <- Box.test(residuals_ets, lag = 20, type = "Ljung-Box")
print(ljung_box_test)

# Perform Shapiro-Wilk test for normality on ETS residuals
shapiro_ets <- shapiro.test(residuals_ets)
print(shapiro_ets)

## 4.2) ARIMA Model
# Extract residuals from the fitted ARIMA model
residuals_arima <- residuals(best_arima_model)

# Plot ACF of residuals
acf(residuals_arima, main = "ACF of ARIMA Model Residuals")

# Plot histogram of residuals
hist(residuals_arima, main = "Histogram of ARIMA Model Residuals", xlab = "Residuals")

# QQ-plot of residuals
qqnorm(residuals_arima)
qqline(residuals_arima, col = "red")

# Perform Ljung-Box test on residuals
ljung_box_test <- Box.test(residuals_arima, lag = 20, type = "Ljung-Box")
print(ljung_box_test)

# Perform Shapiro-Wilk test for normality on ETS residuals
shapiro_arima <- shapiro.test(residuals_arima)
print(shapiro_arima)

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Combine the training and test data
full_data <- c(train_data, test_data)

# Forecast using the ETS model (extending to cover the full range)
ets_forecast <- forecast(ets_AAA, h = length(test_data))

# Combine the actual data (train + test) and the forecasted data
actual_vs_forecast <- data.frame(
  Date = time(full_data),
  Actual = as.numeric(full_data),
  Forecast = c(rep(NA, length(train_data)), as.numeric(ets_forecast$mean))
)

# Plot the actual vs. forecasted data
ggplot() +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Actual), color = "blue", linetype = "solid", size = 1) +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Forecast), color = "red", linetype = "solid", size = 1) +
  labs(title = "ETS Model: Actual vs. Forecasted Data (Including Training Period)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme(legend.title = element_blank())

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecast using the ETS model (extending to cover the full range)
ets_forecast <- forecast(best_ets_model, h = length(test_data))

# Combine the actual data (train + test) and the forecasted data
actual_vs_forecast <- data.frame(
  Date = time(full_data),
  Actual = as.numeric(full_data),
  Forecast = c(rep(NA, length(train_data)), as.numeric(ets_forecast$mean))
)

# Plot the actual vs. forecasted data
ggplot() +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Actual), color = "blue", linetype = "solid", size = 1) +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Forecast), color = "red", linetype = "solid", size = 1) +
  labs(title = "ETS Model: Actual vs. Forecasted Data (Including Training Period)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme(legend.title = element_blank())


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecast using the ETS model (extending to cover the full range)
arima_forecast <- forecast(arima_model, h = length(test_data))

# Combine the actual data (train + test) and the forecasted data
actual_vs_forecast <- data.frame(
  Date = time(full_data),
  Actual = as.numeric(full_data),
  Forecast = c(rep(NA, length(train_data)), as.numeric(arima_forecast$mean))
)

# Plot the actual vs. forecasted data
ggplot() +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Actual), color = "blue", linetype = "solid", size = 1) +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Forecast), color = "red", linetype = "solid", size = 1) +
  labs(title = "ARIMA Model: Actual vs. Forecasted Data (Including Training Period)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme(legend.title = element_blank())


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecast using the ETS model (extending to cover the full range)
arima_forecast <- forecast(best_arima_model, h = length(test_data))

# Combine the actual data (train + test) and the forecasted data
actual_vs_forecast <- data.frame(
  Date = time(full_data),
  Actual = as.numeric(full_data),
  Forecast = c(rep(NA, length(train_data)), as.numeric(arima_forecast$mean))
)

# Plot the actual vs. forecasted data
ggplot() +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Actual), color = "blue", linetype = "solid", size = 1) +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Forecast), color = "red", linetype = "solid", size = 1) +
  labs(title = "ARIMA Model: Actual vs. Forecasted Data (Including Training Period)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme(legend.title = element_blank())


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
library(prophet)
df <- data.frame(ds = filtered_data$FL_DATE, y = filtered_data$Avg_ARR_DELAY)
m <- prophet(df)
future <- make_future_dataframe(m, periods = length(test_data))
forecast <- predict(m, future)
plot(m, forecast)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecast using the selected ETS model
forecast_ets <- forecast(best_ets_model, h = length(test_data))
plot(forecast_ets, main = "Selected ETS Model Forecast", ylab = "Delay", xlab = "Weeks")
lines(test_data, col = "red")

# Forecast using the  selected ARIMA model
forecast_arima <- forecast(arima_model, h = length(test_data))
plot(forecast_arima, main = "ARIMA(1,0,0)(0,1,0)[52]", ylab = "Delay", xlab = "Weeks")
lines(test_data, col = "red")

# Forecast using the  selected ARIMA model
forecast_arima2 <- forecast(best_arima_model, h = length(test_data))
plot(forecast_arima2, main = "Selected ARIMA Model Forecast", ylab = "Delay", xlab = "Weeks")
lines(test_data, col = "red")

# Forecast using the selected ETS model
forecast_ets <- forecast(best_ets_model, h = length(test_data))
# Transform forecasts back to original scale
ets_forecast_original <- exp(forecast_ets$mean)
ets_forecast_lower <- exp(forecast_ets$lower)
ets_forecast_upper <- exp(forecast_ets$upper)
# Plot the forecasts in the original scale
plot(exp(test_data), main = "Selected ETS Model Forecast (Original Scale)", ylab = "Number of Sales", xlab = "Time", col = "red", type = "l", lwd = 2)
lines(ets_forecast_original, col = "blue", lwd = 2)
lines(ets_forecast_lower[,2], col = "blue", lty = 2)  # 95% lower prediction interval
lines(ets_forecast_upper[,2], col = "blue", lty = 2)  # 95% upper prediction interval

# Forecast using the selected ARIMA model
forecast_arima <- forecast(best_arima_model, h = length(test_data))
# Transform forecasts back to original scale
arima_forecast_original <- exp(forecast_arima$mean)
arima_forecast_lower <- exp(forecast_arima$lower)
arima_forecast_upper <- exp(forecast_arima$upper)
# Plot the forecasts in the original scale
plot(exp(test_data), main = "Selected ARIMA Model Forecast (Original Scale)", ylab = "Number of Sales", xlab = "Time", col = "red", type = "l", lwd = 2)
lines(arima_forecast_original, col = "blue", lwd = 2)
lines(arima_forecast_lower[,2], col = "blue", lty = 2)  # 95% lower prediction interval
lines(arima_forecast_upper[,2], col = "blue", lty = 2)  # 95% upper prediction interval

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
### ----- 6) Evaluation -----

# Actual values
actual_values <- test_data

# ETS Forecasts
ets_forecast <- forecast_ets$mean

# ARIMA Forecasts
arima_forecast <- forecast_arima$mean

# ARIMA Forecasts
arima_forecast2 <- forecast_arima2$mean

# Calculate naive forecast for MASE calculation
naive_forecast <- naive(train_data, h = length(test_data))$mean
naive_mae <- mean(abs(naive_forecast - actual_values))

# Calculate accuracy metrics for ETS
ets_accuracy <- accuracy(ets_forecast, actual_values)
ets_mae <- ets_accuracy[,"MAE"]
ets_rmse <- ets_accuracy[,"RMSE"]
ets_mape <- ets_accuracy[,"MAPE"]
ets_mase <- ets_mae / naive_mae

# Calculate accuracy metrics for ARIMA
arima_accuracy <- accuracy(arima_forecast, actual_values)
arima_mae <- arima_accuracy[,"MAE"]
arima_rmse <- arima_accuracy[,"RMSE"]
arima_mape <- arima_accuracy[,"MAPE"]
arima_mase <- arima_mae / naive_mae

# Calculate accuracy metrics for ARIMA
arima_accuracy2 <- accuracy(arima_forecast2, actual_values)
arima_mae2 <- arima_accuracy2[,"MAE"]
arima_rmse2 <- arima_accuracy2[,"RMSE"]
arima_mape2 <- arima_accuracy2[,"MAPE"]
arima_mase2 <- arima_mae / naive_mae

# Print accuracy metrics
print("ETS Model Accuracy:")
print(paste("MAE:", ets_mae))
print(paste("RMSE:", ets_rmse))
print(paste("MAPE:", ets_mape))
print(paste("MASE:", ets_mase))

print("ARIMA Model Accuracy:")
print(paste("MAE:", arima_mae))
print(paste("RMSE:", arima_rmse))
print(paste("MAPE:", arima_mape))
print(paste("MASE:", arima_mase))

print("ARIMA Model Accuracy:")
print(paste("MAE:", arima_mae2))
print(paste("RMSE:", arima_rmse2))
print(paste("MAPE:", arima_mape2))
print(paste("MASE:", arima_mase2))

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate accuracy metrics for the first ARIMA model
arima_accuracy1 <- accuracy(forecast(arima_model, h = length(test_data))$mean, actual_values)
arima_mae1 <- arima_accuracy1[,"MAE"]
arima_rmse1 <- arima_accuracy1[,"RMSE"]
arima_mape1 <- arima_accuracy1[,"MAPE"]
arima_mase1 <- arima_mae1 / naive_mae

# Calculate accuracy metrics for the best ARIMA model
arima_accuracy2 <- accuracy(forecast(best_arima_model, h = length(test_data))$mean, actual_values)
arima_mae2 <- arima_accuracy2[,"MAE"]
arima_rmse2 <- arima_accuracy2[,"RMSE"]
arima_mape2 <- arima_accuracy2[,"MAPE"]
arima_mase2 <- arima_mae2 / naive_mae

# Print accuracy metrics for the first ARIMA model
print("ARIMA Model (1,0,0)(0,1,0)[52] Accuracy:")
print(paste("MAE:", arima_mae1))
print(paste("RMSE:", arima_rmse1))
print(paste("MAPE:", arima_mape1))
print(paste("MASE:", arima_mase1))

# Print accuracy metrics for the best ARIMA model
print("Best ARIMA Model Accuracy:")
print(paste("MAE:", arima_mae2))
print(paste("RMSE:", arima_rmse2))
print(paste("MAPE:", arima_mape2))
print(paste("MASE:", arima_mase2))




## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Convert weekly data to monthly data by aggregating the average delays
monthly_data <- filtered_data %>%
  mutate(Month = floor_date(FL_DATE, "month")) %>%
  group_by(Month) %>%
  summarise(Avg_ARR_DELAY = mean(Avg_ARR_DELAY, na.rm = TRUE))

# Display the first few rows of the monthly data
head(monthly_data)

# Create a time series object from the monthly data
monthly_data_ts <- ts(monthly_data$Avg_ARR_DELAY, start = c(2021, 1), frequency = 12)

# Plot the monthly data
ggplot(monthly_data, aes(x = Month, y = Avg_ARR_DELAY)) +
  geom_line(color = "blue") +
  labs(title = "Average Arrival Delay Over Time (Monthly Aggregation)",
       x = "Month",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal()

# Split the monthly data into training and test sets (80% training, 20% test)
train_length <- floor(0.8 * length(monthly_data_ts))
train_data <- ts(monthly_data_ts[1:train_length], start = start(monthly_data_ts), frequency = frequency(monthly_data_ts))
test_data <- ts(monthly_data_ts[(train_length + 1):length(monthly_data_ts)], start = time(monthly_data_ts)[train_length + 1], frequency = frequency(monthly_data_ts))

# Fit the ETS model on the training data
ets_model <- ets(train_data, model = "AAA")
summary(ets_model)

# Forecast using the ETS model
ets_forecast <- forecast(ets_model, h = length(test_data))

# Combine the actual data (train + test) and the forecasted data
full_data <- c(train_data, test_data)
actual_vs_forecast <- data.frame(
  Date = time(full_data),
  Actual = as.numeric(full_data),
  Forecast = c(rep(NA, length(train_data)), as.numeric(ets_forecast$mean))
)

# Plot the actual vs. forecasted data
ggplot() +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Actual), color = "blue", linetype = "solid", size = 1) +
  geom_line(data = actual_vs_forecast, aes(x = Date, y = Forecast), color = "red", linetype = "solid", size = 1) +
  labs(title = "ETS Model: Actual vs. Forecasted Monthly Data (Including Training Period)",
       x = "Date",
       y = "Average Arrival Delay (minutes)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme(legend.title = element_blank())

# Evaluate the residuals of the ETS model
residuals_ets <- residuals(ets_model)

# Plot ACF of residuals
acf(residuals_ets, main = "ACF of ETS Model Residuals")

# Plot histogram of residuals
hist(residuals_ets, main = "Histogram of ETS Model Residuals", xlab = "Residuals")

# QQ-plot of residuals
qqnorm(residuals_ets)
qqline(residuals_ets, col = "red")

# Perform Ljung-Box test on residuals
ljung_box_test <- Box.test(residuals_ets, lag = 20, type = "Ljung-Box")
print(ljung_box_test)

# Perform Shapiro-Wilk test for normality on ETS residuals
shapiro_ets <- shapiro.test(residuals_ets)
print(shapiro_ets)



    `;
</script>

<div class="code-container">
  <h3>Code</h3>
  <div class="code-editor">
    <pre>{@html highlightedCode}</pre>
  </div>
</div>

<style>
  .code-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    height: 100%;
    margin: 40px;
    margin-left: 80px;
  }

  .code-editor {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 16px;
    border-radius: 8px;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    overflow: auto;
    height: 80%;
    max-height: 800px;
    width: 100%;
    max-width: 400px;
  }

  pre {
    margin: 0;
  }
</style>
