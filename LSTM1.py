
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('AirPassengers.csv')
df.columns = ['Month', 'Passengers']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Passengers']])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Prepare sequences
seq_length = 12
X, y = create_sequences(scaled_data, seq_length)

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=75, verbose=1)

# Predict on last known sequences
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
future_preds = []

for _ in range(12):
    pred = model.predict(last_sequence, verbose=0)[0][0]
    future_preds.append(pred)
    new_input = np.append(last_sequence[0, 1:], [[pred]], axis=0)
    last_sequence = new_input.reshape(1, seq_length, 1)

# Inverse transform the predictions
future_passengers = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Prepare future date range
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# Create future DataFrame
future_df = pd.DataFrame({'Month': future_dates, 'Predicted Passengers': future_passengers})
print(future_df)

# Plot historical and future predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Passengers'], label='Historical Data')
plt.plot(future_df['Month'], future_df['Predicted Passengers'], label='Future Forecast', linestyle='--', marker='o')
plt.title('Air Passengers Forecast (Next 12 Months)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('future_forecast.png')
plt.show()

