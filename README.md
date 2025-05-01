# Air Passengers Time Series Forecasting Using LSTM

This project implements a time series forecasting model using Long Short-Term Memory (LSTM) neural networks. The model is trained on the Air Passengers dataset, which contains monthly totals of international airline passengers from 1949 to 1960.

## Dataset

The dataset file `AirPassengers.csv` includes:

- `Month`: Date of observation (monthly format)
- `Passengers`: Number of international airline passengers

## Objective

The objective of this project is to build and train an LSTM model that can predict future passenger counts, specifically forecasting the next 12 months based on historical data.

## Requirements

The following Python libraries are required to run the script:

- pandas  
- numpy  
- matplotlib  
- scikit-learn  
- tensorflow  

You can install them using:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```
## Usage

Place AirPassengers.csv in the same directory as the script.
Run the Python script:
```bash
python LSTM1.py
```
## Output

The script will:

Train an LSTM model using the historical dataset.

Forecast the number of passengers for the next 12 months.

Display a plot comparing historical data with future predictions.

Save the plot as future_forecast.png.

Print a table of forecasted passenger counts for the next year.

## Model Summary

Type: LSTM (Long Short-Term Memory) Neural Network

Sequence Length: 12 months (one year used to predict the next month)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Epochs: 100

## Example Output Table

| Month      | Predicted Passengers |
|:----------:|:--------------------:|
| 1961-01-01 | 440.12               |
| 1961-02-01 | 423.55               |
| 1961-03-01 | ...                  |
| 1961-12-01 | 500.92               |


## License
This project is intended for academic and educational use.
