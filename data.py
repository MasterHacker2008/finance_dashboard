import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(layout="wide")
# Load the data into a pandas DataFrame (assuming the data is stored in a CSV)
data = pd.read_csv('commodities_data.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Set the Date column as the index
data.set_index('Date', inplace=True)


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


def compute_portfolio(returns_df, asset_columns, weights, risk_free_rate=0.0):
    """
    Computes portfolio return, volatility, and Sharpe ratio.

    Parameters:
    - returns_df: DataFrame containing daily return columns
    - asset_columns: list of column names for individual asset returns
    - weights: list or np.array of weights (must sum to 1)
    - risk_free_rate: daily risk-free rate (e.g., 0.01 / 252 for 1% annual)

    Returns:
    - Dictionary with portfolio return series and summary stats
    """

    # Ensure weights are a numpy array
    weights = np.array(weights)

    # Check if weights match the number of assets
    if len(weights) != len(asset_columns):
        raise ValueError("Weights must match the number of selected asset columns.")

    # Calculate daily portfolio returns
    portfolio_returns = (returns_df[asset_columns] * weights).sum(axis=1)

    # Annualize stats
    trading_days = 252
    mean_return = portfolio_returns.mean() * trading_days
    volatility = portfolio_returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (mean_return - risk_free_rate * trading_days) / volatility if volatility != 0 else np.nan

    return {
        "daily_returns": portfolio_returns,
        "annual_return": mean_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe_ratio
    }


# Apply the fill_nan function to each of the columns
for col in data.head():
    data[col] = fill_nan(data[col])

# Calculating Returns
for i in data.head():
    data[i + "_return"] = data[i][::-1].pct_change()

# data = data.sort_index()

# Calculate 30-day Rolling Volatility
columns = [x for x in data.columns if "return" in x]  # Daily return coloumns
for i in columns:
    data[i[:-7] + '_Volatility'] = data[i].rolling(window=30).std()

returns = data[columns]
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

weights = [1 / len(columns) for _ in range(len(columns))] #Equal weights for each commodity

portfolio = compute_portfolio(data, columns, weights)

print("Annual Return:", round(portfolio['annual_return'] * 100, 2), "%")
print("Annual Volatility:", round(portfolio['annual_volatility'] * 100, 2), "%")
print("Sharpe Ratio:", round(portfolio['sharpe_ratio'], 2))

# View the data in a csv file (Testing)
data.to_csv("test.csv")

st.title("Financial Analysis")

st.dataframe(data)

st.line_chart(data['Oil_return'])

plt.figure(figsize=(12, 6))

# plt.plot(data.index, data['Gold_Volatility'], label='Gold Volatility')

for i in columns:
    plt.plot(data.index, data[i], label=i)


plt.xlabel('Date')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
