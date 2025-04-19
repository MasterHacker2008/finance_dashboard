import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

trading_days = 252


def readData(file_path):
    # Load the data into a pandas DataFrame (assuming the data is stored in a CSV)
    data = pd.read_csv(file_path)
    # Convert the 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    # Set the Date column as the index
    data.set_index('Date', inplace=True)

    return data


# Function to fill NaN with average of previous two values
def fill_nan(series):
    series = series.copy()
    for i in range(len(series)):
        if pd.isna(series[i]) and i >= 2:
            prev1 = series[i - 1]
            prev2 = series[i - 2]
            if pd.notna(prev1) and pd.notna(prev2):
                series[i] = (prev1 + prev2) / 2
    return series


def daily_returns(data):
    df = pd.DataFrame()
    # Calculating Returns
    for i in data.head():
        df[i + "_return"] = data[i][::-1].pct_change()
    return df


def compute_portfolio(portfolio_returns, risk_free_rate):
    # Annualize stats

    mean_return = portfolio_returns.mean() * trading_days
    volatility = portfolio_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else np.nan

    return {
        "annual_return": mean_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe_ratio
    }

# data = data.sort_index()

# Calculate 30-day Rolling Volatility
# columns = [x for x in data.columns if "return" in x]  # Daily return coloumns
# for i in columns:
#     data[i[:-7] + '_Volatility'] = data[i].rolling(window=30).std()
#
# returns = data[columns]
# mean_daily_returns = returns.mean()
# cov_matrix = returns.cov()
#
# weights = [1 / len(columns) for _ in range(len(columns))]  # Equal weights for each commodity
#
# portfolio = compute_portfolio(data, columns, weights)
#
# print("Annual Return:", round(portfolio['annual_return'] * 100, 2), "%")
# print("Annual Volatility:", round(portfolio['annual_volatility'] * 100, 2), "%")
# print("Sharpe Ratio:", round(portfolio['sharpe_ratio'], 2))
