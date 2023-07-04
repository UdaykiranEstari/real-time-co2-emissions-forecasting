import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product


# Load the preprocessed data
data = pd.read_csv('data/processed/CO2_Emissions.csv')
data = data[['year', 'China']]

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Define evaluation metrics
def evaluate_forecast(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

# AR Model
best_ar_mse = float('inf')
best_ar_model = None
best_ar_p = None

for p in range(1, 6):
    ar_model = AutoReg(train_data['China'], lags=p)
    ar_results = ar_model.fit()
    ar_predictions = ar_results.predict(start=train_size, end=len(data)-1)

    ar_mse, ar_mae = evaluate_forecast(test_data['China'], ar_predictions)

    if ar_mse < best_ar_mse:
        best_ar_mse = ar_mse
        best_ar_model = ar_results
        best_ar_p = p

print('AR Model:')
print('Best p:', best_ar_p)
print('MSE:', best_ar_mse)
print('MAE:', evaluate_forecast(test_data['China'], best_ar_model.predict(start=train_size, end=len(data)-1))[1])

# ARIMA Model
best_arima_mse = float('inf')
best_arima_model = None
best_arima_params = None

for params in ParameterGrid({'p': range(1, 6), 'd': range(2), 'q': range(2)}):
    arima_model = ARIMA(train_data['China'], order=(params['p'], params['d'], params['q']))
    arima_results = arima_model.fit()
    arima_predictions = arima_results.predict(start=train_size, end=len(data)-1)

    arima_mse, arima_mae = evaluate_forecast(test_data['China'], arima_predictions)

    if arima_mse < best_arima_mse:
        best_arima_mse = arima_mse
        best_arima_model = arima_results
        best_arima_params = params

print('ARIMA Model:')
print('Best Parameters:', best_arima_params)
print('MSE:', best_arima_mse)
print('MAE:', evaluate_forecast(test_data['China'], best_arima_model.predict(start=train_size, end=len(data)-1))[1])

# SARIMA Model with Parameter Tuning
best_sarima_mse = float('inf')
best_sarima_model = None
best_sarima_params = None

p_values = range(1, 6)
d_values = range(2)
q_values = range(2)
P_values = range(2)
D_values = range(1)
Q_values = range(2)
m = 12  # Seasonal period (assuming monthly data)

for params in product(p_values, d_values, q_values, P_values, D_values, Q_values):
    order = (params[0], params[1], params[2])
    seasonal_order = (params[3], params[4], params[5], m)
    sarima_model = SARIMAX(train_data['China'], order=order, seasonal_order=seasonal_order)
    sarima_results = sarima_model.fit()
    sarima_predictions = sarima_results.predict(start=train_size, end=len(data)-1)

    sarima_mse, sarima_mae = evaluate_forecast(test_data['China'], sarima_predictions)

    if sarima_mse < best_sarima_mse:
        best_sarima_mse = sarima_mse
        best_sarima_model = sarima_results
        best_sarima_params = params

print('SARIMA Model:')
print('Best Parameters:', best_sarima_params)
print('MSE:', best_sarima_mse)
print('MAE:', evaluate_forecast(test_data['China'], best_sarima_model.predict(start=train_size, end=len(data)-1))[1])

# LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data['China'].values.reshape(-1, 1))

def create_lstm_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 1
train_X, train_y = create_lstm_dataset(scaled_train_data, look_back)

best_lstm_mse = float('inf')
best_lstm_model = None
best_lstm_units = None

for units in range(10, 101, 10):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=0)

    lstm_predictions = model.predict(test_data['China'].values.reshape(-1, 1))
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    lstm_mse, lstm_mae = evaluate_forecast(test_data['China'], lstm_predictions)

    if lstm_mse < best_lstm_mse:
        best_lstm_mse = lstm_mse
        best_lstm_model = model
        best_lstm_units = units

print('LSTM Model:')
print('Best Units:', best_lstm_units)
print('MSE:', best_lstm_mse)
print('MAE:', evaluate_forecast(test_data['China'], best_lstm_model.predict(test_data['China'].values.reshape(-1, 1)))[1])

# Compare the performance of the models
models = ['AR', 'ARIMA', 'SARIMA', 'LSTM']
mse_scores = [best_ar_mse, best_arima_mse, 0, best_lstm_mse]
mae_scores = [evaluate_forecast(test_data['China'], best_ar_model.predict(start=train_size, end=len(data)-1))[1],
              evaluate_forecast(test_data['China'], best_arima_model.predict(start=train_size, end=len(data)-1))[1],
              0,
              evaluate_forecast(test_data['China'], best_lstm_model.predict(test_data['China'].values.reshape(-1, 1)))[1]]

plt.bar(models, mse_scores, color='blue', label='MSE')
plt.bar(models, mae_scores, color='orange', label='MAE')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.legend()
plt.show()

# Select the best-performing model
best_model_index = np.argmin(mse_scores)
best_model = models[best_model_index]
print('Best Model:', best_model)

# Save the best model
if best_model == 'AR':
    best_ar_model.save('models/AR.pkl')
elif best_model == 'ARIMA':
    best_arima_model.save('models/ARIMA.pkl')
elif best_model == 'SARIMA':
    best_sarima_model.save('models/SARIMA.pkl')
elif best_model == 'LSTM':
    best_lstm_model.save('models/LSTM.h5')

# Make predictions for the next 10 years using the best model
if best_model == 'AR':
    predictions = best_ar_model.predict(start=len(data), end=len(data)+9)
elif best_model == 'ARIMA':
    predictions = best_arima_model.predict(start=len(data), end=len(data)+9)
elif best_model == 'SARIMA':
    predictions = best_sarima_model.predict(start=len(data), end=len(data)+9)
elif best_model == 'LSTM':
    predictions = best_lstm_model.predict(test_data['China'].values.reshape(-1, 1))
    predictions = scaler.inverse_transform(predictions)

# Show the best model's predictions
print('Predictions:', predictions)

# save the predictions to list
predictions = predictions.tolist()

# now make a dataframe using the above predictions list and the years 2021 to 2030
predictions = pd.DataFrame({'year': range(2021, 2031), 'China': predictions})


# Plot the predictions
plt.plot(data['year'], data['China'], label='Actual')
plt.plot(predictions['year'],predictions['China'], label='Predicted')
