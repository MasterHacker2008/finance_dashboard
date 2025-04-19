import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio_utils import fill_nan, readData, daily_returns, compute_portfolio


st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

data = readData('commodities_data.csv')

# Fill NaN values in data
for col in data.columns:
    data[col] = fill_nan(data[col])

# Slider to filter years
start_year, end_year = st.slider(
    "Select year range",
    min_value=int(data.index.year.min()),
    max_value=int(data.index.year.max()),
    value=(2010, 2020),
    step=1
)

# Multiselect to filter commodities
commodities = st.multiselect(
    "Select commodities",
    options=data.columns,
    default=["Oil", "Gold", "Silver"]
)

# Filter data based on user input
filtered_data = data[
    (data.index.year >= start_year) & (data.index.year <= end_year)
][commodities]

st.dataframe(filtered_data, use_container_width=True)


# Resample to year-end prices
yearly_prices = filtered_data.resample('Y').last()

# Calculate annual returns
annual_returns = yearly_prices.pct_change().dropna()
annual_returns.index = annual_returns.index.year  # Replace full date with just year

# Plotting annual returns

st.subheader("Annual Returns of Selected Commodities")
st.line_chart(annual_returns)

st.subheader("Specify Portfolio Weights for Each Commodity")

# Toggle for optimization
optimize_weights = st.toggle("🔄 Automatically optimize weights for maximum Sharpe Ratio")

weights = {}

if optimize_weights:
    st.info("Weights will be optimized to maximize Sharpe Ratio.")

    # Placeholder: set equal weights (you'll replace this with actual optimization logic)
    weights = {commodity: 1 / len(commodities) for commodity in commodities}

else:
    st.info("Manually set weights for each selected commodity.")

    total_weight = 0
    for commodity in commodities:
        weight = st.number_input(
            f"Weight for {commodity} (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0 / len(commodities),
            step=1.0,
            format="%.1f"
        )
        weights[commodity] = weight
        total_weight += weight

    if total_weight != 100.0:
        st.warning(f"⚠️ Total weight is {total_weight}%. Please adjust to total 100%.")
    else:
        st.success("✅ Weights look good!")

# Normalize weights to sum to 1
normalized_weights = {k: v / 100 for k, v in weights.items()}

print(filtered_data)
returns_df = daily_returns(filtered_data)

print(returns_df)

weights = list(normalized_weights.values())
portfolio_returns = (returns_df * weights).sum(axis=1)


portfolio = compute_portfolio(portfolio_returns, risk_free_rate=0.0)

asset_names = list(filtered_data.columns)
# Display the chosen weights

col1, col2 = st.columns([0.4,0.6])
with col1:

    weights_df = pd.DataFrame(weights, index=asset_names, columns=["Weight"])
    st.dataframe(weights_df, use_container_width=True)


fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')  # no background
ax.pie(
    weights,
    labels=asset_names,
    autopct="%1.1f%%",
    startangle=90,
    textprops={'fontsize': 10, "color":"#D3D3D3"}
)
ax.axis("equal")  # Ensure it's a circle

# Remove white frame from the plot (optional)
fig.patch.set_alpha(0.0)

with col2:
    st.pyplot(fig)



st.header("Portfolio Performance")
st.write(f"**Annual Return:** {portfolio["annual_return"] * 100:.2f}%")
st.write(f"**Annual Volatility:** {portfolio["annual_volatility"] * 100:.2f}%")
st.write(f"**Sharpe Ratio:** {portfolio["sharpe_ratio"]:.2f}")



