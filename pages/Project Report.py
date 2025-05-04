import streamlit as st

# --- Project Write-up Section ---
# -------------------------------------
# Header
# -------------------------------------
st.set_page_config(page_title="Risk-Adjusted Commodities Portfolio", layout="wide")

st.title("📊 Risk-Adjusted Commodities Portfolio")
st.markdown("**By Matthew Reynolds and Arjun Kunder**")

st.markdown("🔗 [View Full Report PDF](https://github.com/MasterHacker2008/finance_dashboard/blob/main/ALT2_Commodities_Analysis_Report.pdf)")

st.markdown(
    """
    <style>
    .custom-text {
        font-size: 14px;
        line-height: 1;
        color: grey;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="custom-text">Candidate Names:</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Matthew Reynolds and Arjun Kunder</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Candidate Number: 2</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Class Group: ComScience</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="custom-text">Teacher Name: Mr McEneaney</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">St. Josephs CBS</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Project Title: Risk Adjusted Commodities Portfolio</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">Date Submitted: 01/05/2025</p>', unsafe_allow_html=True)



st.divider()





# -------------------------------------
# 1. Problem Statement
# -------------------------------------
with st.expander("📌 1. Problem Statement"):
    st.markdown("""
    The purpose of this investigation is to analyse a dataset to develop the most effective **risk-adjusted commodities portfolio**.

    **Hypothesis**:  
    A portfolio weighted for **Sharpe Ratio** will have a higher risk-adjusted return than any single commodity.
    """)

# -------------------------------------
# 2. Dataset Description
# -------------------------------------
with st.expander("📁 2. Dataset Description"):
    st.markdown("""
    - Filename: `commodities_data.csv`  
    - Source: Public finance dataset (kaggle.com)  
    - Records: ~6,500  
    - Fields: Name, Daily Prices over 25 years
    """)
    st.download_button(
        label="Download CSV",
        file_name="commodities_data.csv",
        mime="text/csv",
        data="./commodities_data.csv"
    )

# -------------------------------------
# 3. Design and Planning
# -------------------------------------
with st.expander("🧪 3. Design and Planning"):
    st.markdown("""
    **Language:** Python  
    **Libraries Used:**
    - `pandas`: Data manipulation  
    - `numpy`: Numerical calculations  
    - `matplotlib`: Plotting  
    - `streamlit`: Interface  
    - `scipy.optimize`: Portfolio optimisation
    """)

    st.markdown("**Data Cleaning:** Missing values filled using average of two surrounding values.")

# Section 4: Implementation
st.subheader('4. Implementation')

st.write("""
The Python program performs the following actions:
1. Reads the .csv file.
2. Cleans the data.
3. Calculates:
    - Daily mean returns
    - Daily portfolio return
    - Annualized return
    - Covariance matrix of daily returns
    - Daily portfolio volatility
    - Annualized volatility
    - Sharpe ratio for each commodity
4. Optimizes the portfolio based on the Sharpe ratio of all the commodities.
5. Compares the portfolio to gold (our top commodity).
6. Visualizes the mean values in a bar chart.
""")

# -------------------------------------
# 5. Sharpe Ratio and Optimisation
# -------------------------------------
st.subheader("5. Sharpe Ratio & Optimisation Logic")

st.markdown("**Sharpe Ratio Formula**")
st.latex(r"\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}")

st.markdown("""
Where:  
- **Rp** = Annual return of the portfolio  
- **Rf** = Risk-free rate (e.g., 0.02)  
- **σp** = Portfolio volatility (standard deviation)
""")

st.markdown(r"""
**Optimisation Logic using `scipy.optimize.minimize`:**
- **Goal**: Maximise Sharpe Ratio (i.e., minimise its negative)
- **Constraints**: Weights sum to 1
- **Bounds**: Each weight between 0 and 1
""")
# -------------------------------------
# 5. Results
# -------------------------------------
with st.expander("📊 5. Analysis of Results"):
    st.markdown("""
    - **Natural Gas** had the highest raw return  
    - **Oil** had the highest volatility  
    - **Gold** had the best Sharpe Ratio  
    - **Optimised Portfolio**: Balanced returns with lower risk

    Graphs include:
    - Annual return line chart
    - Sharpe Ratio bar chart
    - Portfolio vs Commodity comparison
    - Pie chart of weights
    """)

# -------------------------------------
# 6. Evaluation
# -------------------------------------
with st.expander("🧠 6. Evaluation and Reflection"):
    st.markdown("""
    **Strengths:**
    - Accurate metrics
    - Effective use of optimisation
    - Interactive dashboard for easier interpretation

    **Limitations:**
    - Ignores macroeconomic factors (inflation, war)
    - Historical data may not reflect future conditions

    **Future Improvements:**
    - Include asset correlations
    - Use machine learning to detect regime changes
    - Add portfolio rebalancing logic
    """)

# -------------------------------------
# 7. Sources
# -------------------------------------
with st.expander("📚 7. References and Sources"):
    st.markdown("""
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SciPy Minimize Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [Streamlit Docs](https://docs.streamlit.io/)
    """)

# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.markdown("© 2025 Matthew Reynolds & Arjun Kunder")
