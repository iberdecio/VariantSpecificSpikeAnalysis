import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from scipy.stats import sem, t

print(os.getcwd())
file_path = 'C:/Users/iberd/Bioinformatics/MonthlyTotalCounts.csv'
test_takens = pd.read_csv(file_path).transpose()
test_takens.columns = ['weekly_cases']
transposedData = test_takens[1:]
transposedData.index = pd.to_datetime(transposedData.index)

transposedData['weekly_cases'] = pd.to_numeric(transposedData['weekly_cases'], errors='coerce')
transposedData['smoothed_cases'] = transposedData['weekly_cases'].rolling(window=3).mean()
transposedData['normalized_cases'] = transposedData['smoothed_cases'] / transposedData['smoothed_cases'].max()
print("Original shape:", test_takens.shape)
print("Transposed shape:",transposedData.shape)

def create_time_delay_embedding(series, D, T):
    lagged_data = pd.DataFrame(index=series.index)
    for i in range(D):
        lagged_data[f'lag_{T*i}'] = series.shift(T * i)
    lagged_data.dropna(inplace=True)
    return lagged_data

normalized_series = transposedData['normalized_cases']
embedded_data = create_time_delay_embedding(normalized_series, D=6, T=2)
X = embedded_data.values
n_future_steps = 4
y = [transposedData['normalized_cases'].iloc[i + n_future_steps:i + n_future_steps + 4].values for i in range(len(embedded_data))]
min_length = min(len(X), len(y))
X, y = X[:min_length], y[:min_length]
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = np.array([np.array(yi) for yi in y])

# Initialize last_model
last_model = None

# Perform multiple training and evaluation runs
n_runs = 30
mse_scores = []
rmse_scores = [] 

for run in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.2))  # Adding dropout here
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    rmse_scores.append(np.sqrt(mse))
    last_model = model  # Update last_model

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
confidence_interval = t.interval(0.95, len(rmse_scores)-1, loc=mean_rmse, scale=sem(rmse_scores))
print("Mean RMSE over 30 runs:", mean_rmse)
print("Standard Deviation of RMSE over 30 runs:", std_rmse)

# Predict using the last trained model
latest_data = normalized_series[-6*2:].values  # Adjust based on D and T
latest_embedded = create_time_delay_embedding(pd.Series(latest_data), D=6, T=2)
latest_embedded_values = latest_embedded.values[-1].reshape((1, 1, latest_embedded.shape[1]))
future_normalized_cases = last_model.predict(latest_embedded_values)
print("Predicted Values:", future_normalized_cases)

max_smoothed_cases = transposedData['smoothed_cases'].max()
future_cases = future_normalized_cases * max_smoothed_cases
print("Unnormalized future cases:", future_cases)