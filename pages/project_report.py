import streamlit as st

st.set_page_config(page_title="Project Report")

st.title("Project Report")

# --- Project Write-up Section ---
st.markdown("---")
st.markdown("## 📘 Project Write-up")

st.download_button(
    label="Download CSV",
    file_name="commodities_data.csv",
    mime="text/csv",
    data="./commodities_data.csv"
)
st.markdown("""
### 🎯 Project Title
**Commodity Portfolio Analysis: Testing the Impact of Diversification on Volatility and Risk-Adjusted Returns**

### 🧠 Hypothesis
A portfolio consisting of a mix of commodities (e.g. Crude Oil, Gold, and Silver) will have lower volatility and higher risk-adjusted returns compared to holding a single commodity.

---

### 🔍 Dataset & Analysis

- **Source**: Historical daily commodity prices (e.g., Gold, Oil).
- **Tools Used**: Python, Pandas, NumPy, Streamlit, Matplotlib, Altair, Scipy.
- **Techniques**:
  - Data cleaning (handling NaN values, sorting)
  - Calculating daily returns, volatility, and Sharpe ratios
  - Constructing weighted portfolios
  - Comparing individual vs diversified performance
  - Visualizing using charts and tables

---

### 🧮 Key Statistics

- **Mean**, **Median**, **Mode**, and **Frequency** were calculated for each asset's return series.
- You can view these under the **Basic Statistics** section above.

---

### ✅ Findings

- The **optimized portfolio** achieved:
  - **Higher Sharpe Ratio** than any individual commodity
  - **Lower volatility** than more volatile single assets (like Oil)
  - **Slightly better returns** than holding Gold alone

This **supports the hypothesis** that diversification improves performance.

---

### 🚫 Limitations

- Focused on a specific time period — may not reflect all market conditions.
- Ignores macroeconomic events (e.g. inflation, war, interest rates).
- Assumes zero risk-free rate unless specified.

---

### 👤 End Users

- Students studying finance or statistics
- Individual investors exploring risk-adjusted returns
- Teachers using the dashboard for demonstration

---

### 🧰 Roles

- **Developer**: Programmed the data analysis and app interface
- **Researcher**: Formulated hypothesis and sourced data
- **Designer**: Designed charts and layout
- **Presenter**: Summarized findings and built dashboard experience

---

### ✅ Conclusion

The analysis clearly shows that combining commodities into a portfolio leads to better risk-adjusted returns and reduced volatility, **proving the hypothesis**.

Thank you for exploring this dashboard!
""")