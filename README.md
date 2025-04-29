# Air Passengers Forecasting with LSTM

This project demonstrates time series forecasting using an LSTM (Long Short-Term Memory) neural network on the classic **Air Passengers** dataset, which contains monthly totals of international airline passengers from 1949 to 1960.

## 📂 Dataset

- **Source**: Provided as `AirPassengers.csv`
- **Columns**:
  - `Month`: Date in `YYYY-MM` format.
  - `#Passengers`: Monthly total number of international airline passengers.

## 📈 Objective

Use past 12 months of passenger data to predict the next month's value using an LSTM model implemented in Keras (TensorFlow backend).

## 🛠️ Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

```

## 🧪 How to Run
Place AirPassengers.csv in the project folder.

Run the lstm_airpassengers_forecast.py script or Jupyter Notebook.

## 🧠 Model Overview
Preprocessing:

Normalizes passenger counts using MinMaxScaler.

Transforms data into sequences of 12 months for training.

Model:

LSTM with 50 units + Dense output layer.

Optimizer: Adam

Loss: Mean Squared Error (MSE)

Training:

Trained for 100 epochs on 80% of the dataset.

## 📊 Output
The model generates a plot comparing the predicted passenger counts to the actual values from the test set.

(replace with your output if needed)

## 📌 Notes
LSTM models are sensitive to scaling and sequence length.

This example does not include hyperparameter tuning or more advanced techniques like seasonality decomposition.
