
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pmdarima as pm

# Load dataset and preprocess
data = pd.read_csv('XAUUSD_2010-2023.csv')
data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
data.set_index('time', inplace=True)
daily_data = data['close'].resample('D').mean().interpolate()

# Use only the last 3 years of data, or adjust if needed
filtered_data = daily_data[daily_data.index >= '2020-01-01']
if filtered_data.empty:
    print("Warning: No data found for the specified date range. Using all available data instead.")
    filtered_data = daily_data

# Plot the filtered time series if data is available
if not filtered_data.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Daily Closing Price')
    plt.title('XAUUSD Daily Closing Price (Last 3 Years)')
    plt.show()
else:
    raise ValueError("No data available in the specified range. Please check the date range or dataset.")

# Check stationarity with ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

check_stationarity(filtered_data)

# Auto ARIMA with reduced search space and smaller seasonal period
auto_arima_model = pm.auto_arima(
    filtered_data,
    start_p=1, start_q=1, max_p=2, max_q=2,
    seasonal=True, m=7,
    start_P=0, start_Q=0, max_P=1, max_Q=1,
    d=1, D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(auto_arima_model.summary())

# Split data into train and test sets (20% test)
train_size = int(len(filtered_data) * 0.8)
train, test = filtered_data[:train_size], filtered_data[train_size:]

# Fit the SARIMA model with optimal parameters
order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order
sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
sarima_result = sarima_model.fit(disp=False)

# Forecasting
predictions = sarima_result.predict(start=len(train), end=len(filtered_data)-1, dynamic=False)

# Evaluate with RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print('RMSE:', rmse)

# Plot Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('SARIMA Model Predictions for XAUUSD Daily Closing Price (Last 3 Years)')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/c467c15f-12a5-42ab-938e-1e6f696195f7)

![image](https://github.com/user-attachments/assets/bb14d21d-82bc-48bd-8236-2c4a7db43392)

### RESULT:
Thus the program run successfully based on the SARIMA model.
