# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_utils import fill_nan, readData, daily_returns, compute_portfolio, commodities_performance, compare, optimizing_weights
import altair as alt

# Page setup for Streamlit
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# Load commodity data from CSV
data = readData('commodities_data.csv')

# Fill missing values in each column
for col in data.columns:
    data[col] = fill_nan(data[col])

# === Sidebar controls ===

# Year range slider for filtering data
start_year, end_year = st.slider(
    "Select year range",
    min_value=int(data.index.year.min()),
    max_value=int(data.index.year.max()),
    value=(2010, 2024),
    step=1
)

# Multiselect widget to choose commodities
commodities = st.multiselect(
    "Select commodities",
    options=data.columns,
    default=["Gold", "Paladium", "Wheat"]
)

# Filter data based on selected years and commodities
filtered_data = data[
    (data.index.year >= start_year) & (data.index.year <= end_year)
][commodities]

# Show filtered data
st.dataframe(filtered_data, use_container_width=True)

# === Returns & Volatility ===

# Resample to year-end prices for annual return calculation
yearly_prices = filtered_data.resample('YE').last()

# Calculate annual returns
annual_returns = yearly_prices.pct_change()
annual_returns.index = annual_returns.index.year  # Use year only for x-axis

# Line chart of annual returns
st.subheader("Annual Returns of Selected Commodities")
st.line_chart(annual_returns)

# === Sharpe Ratio Inputs ===

# User inputs the risk-free rate (e.g., 3%)
risk_free_rate = st.number_input(
    label="Enter the Risk-Free Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.5,
    format="%.2f"
) / 100  # Convert % to decimal

# Calculate daily returns and performance
returns_df = daily_returns(filtered_data)
commodities_df = commodities_performance(returns_df, risk_free_rate)

# Identify top-performing commodity
top_commodity = commodities_df.sort_values(by="Sharpe Ratio", ascending=False).iloc[0]

# === Display Sharpe Ratios ===

st.subheader("Sharpe ratio of individual commodities")
col1, col2 = st.columns([0.5, 0.5])
with col1:
    st.write(f"The Sharpe Ratio is greater for {top_commodity['Commodity']}")
    st.dataframe(commodities_df)

with col2:
    # Plot bar chart of Sharpe ratios
    commodities_df.set_index("Commodity", inplace=True)
    st.bar_chart(commodities_df["Sharpe Ratio"])

# === Portfolio Weights ===

st.subheader("Specify Portfolio Weights for Each Commodity")
optimize_weights = st.toggle("Optimize Weights", value=True)
weights = {}

if optimize_weights:
    # Automatically calculate weights for best Sharpe ratio
    st.info("Weights will be optimized to maximize Sharpe Ratio.")
    weights = optimizing_weights(returns_df, risk_free_rate)
else:
    # Let user manually assign weights
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

    # Show warning if weights don't add up to 100%
    if total_weight != 100.0:
        st.warning(f"⚠️ Total weight is {total_weight}%. Please adjust to total 100%.")
    else:
        st.success("✅ Weights look good!")
        # Normalize weights to decimals (0 to 1)
        normalized_weights = {k: v / 100 for k, v in weights.items()}
        weights = list(normalized_weights.values())

# === Portfolio Analysis ===

# Compute portfolio performance using user-selected or optimized weights
portfolio = compute_portfolio(returns_df, risk_free_rate=risk_free_rate, weights=weights)
asset_names = list(filtered_data.columns)

# Display weights
col1, col2 = st.columns([0.4, 0.6])
weights_df = pd.DataFrame(weights, index=asset_names, columns=["Weight"])
weights_df = weights_df[weights_df["Weight"] > 0]

with col1:
    st.dataframe(weights_df, use_container_width=True)

# Pie chart for portfolio weights
fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
ax.pie(
    list(weights_df["Weight"]),
    labels=list(weights_df.index),
    autopct="%1.1f%%",
    startangle=90,
    textprops={'fontsize': 10, "color": "#D3D3D3"}
)
ax.axis("equal")
fig.patch.set_alpha(0.0)

with col2:
    st.pyplot(fig)

# === Bubble Chart: Risk vs Return ===

st.header("Performance Metrics")

# Combine commodity and portfolio metrics
data = {
    'Asset': commodities_df.index.tolist() + ["Portfolio"],
    'Return': commodities_df["Annual Return"].tolist() + [portfolio['annual_return']],
    'Risk': commodities_df["Annual Volatility"].tolist() + [portfolio['annual_volatility']],
    'Sharpe': commodities_df["Sharpe Ratio"].tolist() + [portfolio['sharpe_ratio']]
}
df = pd.DataFrame(data)

# Bubble chart using Altair
chart = alt.Chart(df).mark_circle().encode(
    x=alt.X('Risk', title='Volatility (Risk)'),
    y=alt.Y('Return', title='Expected Return'),
    size=alt.Size('Sharpe', title='Sharpe Ratio', scale=alt.Scale(range=[100, 1000])),
    color='Asset',
    tooltip=['Asset', 'Return', 'Risk', 'Sharpe']
).properties(
    title='Risk vs Return (Bubble Size = Sharpe Ratio)',
    width=700,
    height=500
)
st.altair_chart(chart, use_container_width=True)

# === Portfolio and Top Commodity Metrics ===

st.subheader("Portfolio")
col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
with col1:
    st.metric(label="Annual Return", value=f"{portfolio['annual_return'] * 100:.2f}%")
with col2:
    st.metric(label="Annual Volatility", value=f"{portfolio['annual_volatility'] * 100:.2f}%")
with col3:
    st.metric(label="Sharpe Ratio", value=f"{portfolio['sharpe_ratio']:.2f}")

st.subheader(top_commodity["Commodity"])
col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
with col1:
    st.metric(label="Annual Return", value=f"{top_commodity['Annual Return'] * 100:.2f}%")
with col2:
    st.metric(label="Annual Volatility", value=f"{top_commodity['Annual Volatility'] * 100:.2f}%")
with col3:
    st.metric(label="Sharpe Ratio", value=f"{top_commodity['Sharpe Ratio']:.2f}")

# === Comparison Bar Chart ===

df = compare(top_commodity, portfolio)
st.subheader("Comparison of Portfolio vs Top Commodity")
st.bar_chart(df, stack=False)
