'''
Imports
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import DateOffset

'''
Data Preparation
'''
data_path = "Quant_Research/Nat_Gas.csv"
nat_gas_prices = pd.read_csv(data_path)
nat_gas_prices['Dates'] = pd.to_datetime(nat_gas_prices['Dates'], format='%m/%d/%y')
nat_gas_prices.set_index('Dates', inplace=True)

'''
Decomposition using STL
'''
stl = STL(nat_gas_prices['Prices'], seasonal=13, robust=True)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal
residual = result.resid

'''
Data Modeling
'''
# Fitting a linear regression model to the trend component
time_index = np.arange(len(trend.dropna()))[:, np.newaxis]
lin_reg = LinearRegression()
lin_reg.fit(time_index, trend.dropna())
extended_time_index = np.arange(len(trend.dropna()) + 12)[:, np.newaxis]
predicted_trend = lin_reg.predict(extended_time_index)

'''
Data Visualization - 2x2 Grid of Plots
'''
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Decomposed Components
axs[0, 0].plot(nat_gas_prices.index, nat_gas_prices['Prices'], label='Historical Prices')
axs[0, 0].plot(nat_gas_prices.index, trend, label='Trend', linestyle=':')
axs[0, 0].plot(nat_gas_prices.index, seasonal, label='Seasonal', linestyle='-.')
axs[0, 0].plot(nat_gas_prices.index, residual, label='Residuals', linestyle='--')
axs[0, 0].set_title('Decomposed Components')
axs[0, 0].legend()

# Plot 2: Actual and Predicted Trend
future_dates = pd.date_range(start=nat_gas_prices.index[-1] + DateOffset(months=1), periods=12, freq='ME')
axs[0, 1].plot(nat_gas_prices.index, trend, label='Actual Trend')
axs[0, 1].plot(future_dates, predicted_trend[-12:], label='Predicted Trend', linestyle='--')
axs[0, 1].set_title('Actual and Predicted Trend')
axs[0, 1].legend()

# Plot 3: Historical Prices, Trend, and Total Forecast
axs[1, 0].plot(nat_gas_prices.index, nat_gas_prices['Prices'], label='Historical Prices')
axs[1, 0].plot(nat_gas_prices.index, trend, label='Trend', linestyle=':')
axs[1, 0].plot(future_dates, predicted_trend[-12:] + seasonal[:12], label='Total Forecast', linestyle='--')  # Simplified total forecast
axs[1, 0].set_title('Natural Gas Prices Forecast')
axs[1, 0].legend()

# Plot 4: Residuals with Significant Spikes
significant_spikes = residual[(residual > 0.2) | (residual < -0.2)]
axs[1, 1].plot(residual, label='Residuals', color='gray')
axs[1, 1].scatter(significant_spikes.index, significant_spikes, color='red', label='Significant Spikes')
axs[1, 1].set_title('Residuals with Significant Spikes')
axs[1, 1].legend()

plt.tight_layout()
plt.show()


'''
Forecasting Function
'''

def estimate_gas_price(start_date, end_date=None, predict_months=13):
    """
    Forecast prices starting from start_date.
    If end_date is None, predicts for predict_months ahead.
    If end_date is provided, predicts from start_date to end_date.
    """
    input_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)
        periods = (end_date.year - input_date.year) * 12 + end_date.month - input_date.month + 1
    else:
        periods = predict_months
    
    # Assuming the model has been fit globally
    delta_years = input_date.year - nat_gas_prices.index[0].year
    delta_months = input_date.month - nat_gas_prices.index[0].month
    months_since_start = delta_years * 12 + delta_months
    future_index = np.arange(months_since_start, months_since_start + periods)[:, np.newaxis]
    predicted_trend = lin_reg.predict(future_index)
    
    future_months = [((input_date.month - 1 + i) % 12) + 1 for i in range(periods)]
    predicted_seasonal = [seasonal[seasonal.index.month == month].mean() for month in future_months]
    
    predicted_prices = predicted_trend + predicted_seasonal
    forecast_dates = pd.date_range(start=input_date, periods=periods, freq='ME')
    forecast_prices = pd.Series(predicted_prices, index=forecast_dates)
    
    return forecast_prices

'''
EXAMPLE USAGE
'''
# Predicting for the entire historical period and for next 12 months from a specific date
historical_predictions = estimate_gas_price(nat_gas_prices.index[0], nat_gas_prices.index[-1])
input_date = '2023-09-30'
specific_date_predictions = estimate_gas_price(input_date)

'''
Forecasting Visualization
'''

# Visualization for historical predictions
historical_start_date = nat_gas_prices.index[0]
historical_end_date = nat_gas_prices.index[-1]
historical_actual_prices = nat_gas_prices[historical_start_date:historical_end_date]['Prices']

plt.figure(figsize=(10, 5))
plt.plot(historical_predictions.index, historical_predictions, label='Predicted Prices (Historical)', marker='o')
plt.plot(historical_actual_prices.index, historical_actual_prices, label='Actual Prices (Historical)', marker='x')
plt.title('Comparison of Predicted and Actual Natural Gas Prices for Historical Period')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Visualization for predictions starting from a specific date
specific_start_date = pd.to_datetime(input_date)
specific_end_date = specific_start_date + pd.DateOffset(months=12)
specific_actual_prices = nat_gas_prices[specific_start_date:specific_end_date]['Prices']

plt.figure(figsize=(10, 5))
plt.plot(specific_date_predictions.index, specific_date_predictions, label='Predicted Prices (Next 12 Months)', marker='o')
plt.plot(specific_actual_prices.index, specific_actual_prices, label='Actual Prices (Next 12 Months)', marker='x')
plt.title(f'Comparison of Predicted and Actual Natural Gas Prices from {input_date}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

'''
Performance Measurement
'''

# Performance measurement for historical predictions
historical_mae = mean_absolute_error(historical_actual_prices, historical_predictions[:len(historical_actual_prices)])
historical_rmse = np.sqrt(mean_squared_error(historical_actual_prices, historical_predictions[:len(historical_actual_prices)]))
historical_mape = np.mean(np.abs((historical_actual_prices - historical_predictions[:len(historical_actual_prices)]) / historical_actual_prices)) * 100

print(f"Historical - Mean Absolute Error (MAE): {historical_mae}")
print(f"Historical - Root Mean Squared Error (RMSE): {historical_rmse}")
print(f"Historical - Mean Absolute Percentage Error (MAPE): {historical_mape}%")

# Performance measurement for next 12 months from specific date
specific_mae = mean_absolute_error(specific_actual_prices, specific_date_predictions[:len(specific_actual_prices)])
specific_rmse = np.sqrt(mean_squared_error(specific_actual_prices, specific_date_predictions[:len(specific_actual_prices)]))
specific_mape = np.mean(np.abs((specific_actual_prices - specific_date_predictions[:len(specific_actual_prices)]) / specific_actual_prices)) * 100

print(f"Specific Date - Mean Absolute Error (MAE): {specific_mae}")
print(f"Specific Date - Root Mean Squared Error (RMSE): {specific_rmse}")
print(f"Specific Date - Mean Absolute Percentage Error (MAPE): {specific_mape}%")

