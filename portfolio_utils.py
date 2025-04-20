import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
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
def fill_nan(column):
    """
        Replaces NaN values in a pandas column with the average of the previous two values (if available).
    """
    column = column.copy()
    for i in range(len(column)):
        # Checks if each item in the column is nan
        if pd.isna(column[i]) and i >= 2:  # Assumes the first two values are not nan
            prev1 = column[i - 1]
            prev2 = column[i - 2]
            if pd.notna(prev1) and pd.notna(prev2):  # Makes sure that the previous 2 values are not nan
                column[i] = (prev1 + prev2) / 2  # Average
    return column


def daily_returns(data):
    """
      Calculates daily percentage returns for each commodity in the DataFrame.
      """
    df = pd.DataFrame()
    # Calculating Returns
    for i in data.head():  # Iterate through each column (commodity)
        # [::-1] - Reverses the column and .pct_change() -calculates percent change
        df[i + "_return"] = data[i][::-1].pct_change()
    return df


def compute_portfolio(returns_df, risk_free_rate, weights):
    weights = np.array(weights)

    # Daily mean return for each asset
    mean_returns = returns_df.mean()

    # Daily portfolio return
    daily_portfolio_return = np.sum(mean_returns * weights)

    # Annualized return
    annual_return = daily_portfolio_return * trading_days

    # Covariance matrix of daily returns
    cov_matrix = returns_df.cov()

    # Daily portfolio volatility
    daily_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Annualized volatility
    annual_volatility = daily_volatility * np.sqrt(trading_days)

    # Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio
    }



def commodities_performance(returns_df):
    commodity_names = []
    sharpe_ratios = []
    annual_returns = []
    annual_vols = []
    for i in list(returns_df.columns):
        print(returns_df[i])
        commodity = compute_portfolio(returns_df[[i]], risk_free_rate=0.0, weights=[1])
        commodity_names.append(i.replace("_return", ""))
        sharpe_ratios.append(commodity["sharpe_ratio"])
        annual_returns.append(commodity["annual_return"])
        annual_vols.append(commodity["annual_volatility"])

    commodity_df = pd.DataFrame({
        "Commodity": commodity_names,
        "Sharpe Ratio": sharpe_ratios,
        "Annual Return": annual_returns,
        "Annual Volatility": annual_vols
    })
    return commodity_df


def objective_function(weights, returns_df, risk_free_rate=0.0):
    # Compute portfolio daily returns
    portfolio_returns = (returns_df * weights).sum(axis=1)

    # Annualized stats
    mean_return = portfolio_returns.mean() * trading_days
    volatility = portfolio_returns.std() * np.sqrt(trading_days)

    sharpe = (mean_return - risk_free_rate) / volatility if volatility != 0 else np.nan

    return -sharpe  # We minimize negative Sharpe


def optimizing_weights(returns_df, risk_free_rate=0.0):
    n_assets = len(returns_df.columns)

    # Initial guess: equal weights
    init_guess = [1.0 / n_assets] * n_assets

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Optimization
    result = minimize(
        objective_function,
        init_guess,
        args=(returns_df, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x  # Optimized weights


def compare(commodity, portfolio):
    # Create comparison DataFrame
    df = pd.DataFrame({
        "Type": ["Portfolio", commodity["Commodity"]],
        "Annual Return": [portfolio["annual_return"], commodity["Annual Return"]],
        "Annual Volatility": [portfolio["annual_volatility"], commodity["Annual Volatility"]],
        "Sharpe Ratio": [portfolio["sharpe_ratio"], commodity["Sharpe Ratio"]],
    })

    # Set index for plotting
    df.set_index("Type", inplace=True)

    return df
