# Exploratory Data Analysis (EDA) & Machine Learning Overview

This document summarizes the **analytical and modeling decisions** behind the IBOVESPA time series forecasting project.  
The focus is on **methodological correctness**, **iterative reasoning**, and **engineering trade-offs**, rather than predictive performance claims.

---

## 🎯 Objective

The objective of this project is to evaluate whether historical IBOVESPA data contains **predictive signal beyond simple persistence**, using a properly designed time series pipeline.

The emphasis is on:
- Correct handling of temporal data
- Robust exploratory analysis
- Transparent modeling decisions

---

## 🧹 Data Preparation (Pre-EDA)

Before analysis, the dataset was reviewed to ensure consistency and usability:

- **Duplicate records** were identified and removed after validating that no trading occurred on the affected dates.
- **Missing values** were handled conservatively to avoid introducing artificial patterns.
- Initial feature selection favored simplicity to reduce noise and leakage risk.

These steps ensured a clean and reliable foundation for analysis.

---

## 🔍 Exploratory Data Analysis (EDA)

### Structural Analysis

Several initial checks were performed to validate the integrity of the time series:

- Record counts per year confirmed continuity and absence of large gaps.
- Visual inspection of closing prices revealed:
  - A long-term upward trend
  - High volatility
  - No strong or stable seasonality

These characteristics informed later modeling choices.

---

### Correlation Analysis

- Open, high, low, and close prices showed very strong correlation.
- Trading volume exhibited weak correlation with price-based features.

Given this redundancy, only the **closing price** was retained for modeling to reduce dimensionality and complexity.

---

### Outlier & Volatility Analysis

Traditional outlier detection techniques were avoided due to the strong upward trend, which would incorrectly flag early values as anomalies.

Instead:
- Daily **percentage variation** was analyzed to identify extreme movements.
- This approach provided a more meaningful view of volatility independent of absolute price levels.

Extreme variations linked to external shocks (e.g., financial crises) were identified and treated cautiously to avoid misleading the model in the absence of external context.

---

## 🕒 Time Series Properties

### Stationarity

Statistical testing confirmed that the closing price series is **non-stationary**, reinforcing the need for models capable of handling trends and temporal dependence.

### Seasonality

Time series decomposition showed that residual components dominate seasonal effects, indicating **weak or inconsistent seasonality**.

This further justified avoiding seasonality-heavy assumptions in modeling.

---

## 🔧 Post-EDA Preprocessing

### Feature Selection

To maintain clarity and reduce noise:
- Only the closing price was retained
- Volume and other correlated price features were excluded

This choice favors interpretability and minimizes leakage risk.

---

### Inflation Adjustment (Key Engineering Decision)

An increase in volatility over time was observed.  
To test whether this was driven by macroeconomic effects rather than market behavior alone, prices were **adjusted for accumulated inflation**.

Key observations:
- Inflation adjustment reduced artificial volatility growth across years
- Percentage variations remained structurally consistent
- The inflation-adjusted series provided a more stable input for modeling

The **inflation-adjusted closing price** was therefore used as the primary model input.

This step reflects a real-world consideration often overlooked in financial ML projects.

---

## ✂️ Data Splitting Strategy

Because time series data is highly order-dependent:

- Data was split **chronologically**
- No shuffling was performed
- Validation windows always followed training windows

This preserves causality and prevents lookahead bias.

---

## 🤖 Modeling Approach

A **Long Short-Term Memory (LSTM)** model was selected due to:

- Non-stationary behavior
- Temporal dependency
- Lack of strong seasonality

Model tuning focused on balancing:
- Trend learning
- Volatility sensitivity
- Computational efficiency

The model was evaluated against a **naive persistence baseline**, which is known to be strong in financial time series.

---

## 🧠 Key Takeaways

- Financial time series are highly autocorrelated, making simple baselines difficult to outperform
- Increased model complexity does not guarantee improved predictive power
- Proper evaluation and honest interpretation are more valuable than optimistic results
- Inflation adjustment can materially improve data quality and interpretability
- Methodological rigor is essential when working with noisy, real-world data

---

## ✅ Conclusion

This project demonstrates an end-to-end analytical workflow for time series forecasting, emphasizing:

- Careful exploratory analysis
- Iterative decision-making
- Correct temporal validation
- Engineering discipline over performance claims

The results reinforce a well-known reality in financial modeling:  
**robust pipelines and honest baselines matter more than complex models**.
