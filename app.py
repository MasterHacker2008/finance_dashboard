import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio_utils import fill_nan, readData, daily_returns, compute_portfolio, commodities_performance, compare, \
    optimizing_weights

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")
st.write("")
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
    default=["Gold", "Paladium", "Wheat"]
)

# Filter data based on user input
filtered_data = data[
    (data.index.year >= start_year) & (data.index.year <= end_year)
    ][commodities]

st.dataframe(filtered_data, use_container_width=True)

# Resample to year-end prices
yearly_prices = filtered_data.resample('YE').last()

# Calculate annual returns
annual_returns = yearly_prices.pct_change()
annual_returns.index = annual_returns.index.year # Replace full date with just year

# Plotting annual returns
st.subheader("Annual Returns of Selected Commodities")
st.write("")
st.line_chart(annual_returns)

returns_df = daily_returns(filtered_data)
df = commodities_performance(returns_df)
top_commodity = df.sort_values(by="Sharpe Ratio", ascending=False).iloc[0]

st.subheader("Sharpe ratio of individual commodities")
st.write("")
col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.write(f"The sharpe ratio is greater for {top_commodity["Commodity"]}")
    st.dataframe(df)

with col2:
    df.set_index("Commodity", inplace=True)  # st.bar_chart needs index for x-axis
    st.bar_chart(df["Sharpe Ratio"])

st.subheader("Specify Portfolio Weights for Each Commodity")
st.write("")
# Toggle for optimization
optimize_weights = st.toggle("Optimize Weights", value=True)

weights = {}

if optimize_weights:
    st.info("Weights will be optimized to maximize Sharpe Ratio.")
    weights = optimizing_weights(returns_df)
    # Placeholder: set equal weights (you'll replace this with actual optimization logic)
    # weights = {commodity: 1 / len(commodities) for commodity in commodities}

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
        weights = list(normalized_weights.values())

portfolio = compute_portfolio(returns_df, risk_free_rate=0.0, weights=weights)

asset_names = list(filtered_data.columns)

# Display the chosen weights
col1, col2 = st.columns([0.4, 0.6])
weights_df = pd.DataFrame(weights, index=asset_names, columns=["Weight"])
weights_df = weights_df[weights_df["Weight"] > 0]

with col1:
    st.dataframe(weights_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')  # no background
ax.pie(
    list(weights_df["Weight"]),
    labels=list(weights_df.index),
    autopct="%1.1f%%",
    startangle=90,
    textprops={'fontsize': 10, "color": "#D3D3D3"}
)
ax.axis("equal")  # Ensure it's a circle

# Remove white frame from the plot (optional)
fig.patch.set_alpha(0.0)

with col2:
    st.pyplot(fig)

st.header("Performance Metrics")
st.write("")

st.subheader("Portfolio")
st.write("")

col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
with col1:
    st.metric(label="Annual Return", value=f"{portfolio['annual_return'] * 100:.2f}%")
with col2:
    st.metric(label="Annual Volatility", value=f"{portfolio['annual_volatility'] * 100:.2f}%")
with col3:
    st.metric(label="Sharpe Ratio", value=f"{portfolio['sharpe_ratio']:.2f}")

st.subheader(top_commodity["Commodity"])
st.write("")

col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
with col1:
    st.metric(label="Annual Return", value=f"{top_commodity['Annual Return'] * 100:.2f}%")
with col2:
    st.metric(label="Annual Volatility", value=f"{top_commodity['Annual Volatility'] * 100:.2f}%")
with col3:
    st.metric(label="Sharpe Ratio", value=f"{top_commodity['Sharpe Ratio']:.2f}")

df = compare(top_commodity, portfolio)

st.write("")
st.subheader("Comparison of Portfolio vs Top Commodity")
st.write("")
st.bar_chart(df, stack=False)
