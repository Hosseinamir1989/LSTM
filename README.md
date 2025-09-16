# 📈 LSTM Stock Price Prediction Pipeline

This project provides an end-to-end pipeline for forecasting next-day end-of-day stock prices using LSTM (Long Short-Term Memory) neural networks. It includes data loading, preprocessing, technical indicator enrichment, normalization, model training, evaluation, and feature importance analysis.

## Features

- Loads historical stock data and enriches it with technical indicators (MACD, RSI, Bollinger Bands, Momentum, etc.)
- Preprocesses and normalizes features for LSTM input
- Trains an LSTM model to predict next-day closing prices
- Evaluates model performance (RMSE, MAE, Directional Accuracy)
- Analyzes feature importance via feature drop analysis
- Outputs evaluation tables and plots for further analysis

## Getting Started

### 1. Install Requirements

```sh
pip install -r requirements.txt
```

## Project Structure

```
.
├── data
│   ├── raw
│   ├── processed
│   └── external
├── notebooks
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   └── feature_engineering.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   └── model_evaluator.py
│   └── utils
│       ├── __init__.py
│       ├── logger.py
│       └── plotter.py
├── requirements.txt
└── README.md
```

- `data/`: Contains folders for raw, processed, and external data.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `src/`: Source code for the project.
  - `data/`: Data loading and preprocessing scripts.
  - `models/`: LSTM model definition and evaluation scripts.
  - `utils/`: Utility scripts for logging and plotting.
- `requirements.txt`: Python package dependencies.
- `README.md`: This README file.



