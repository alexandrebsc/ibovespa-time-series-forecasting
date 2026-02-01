# IBOVESPA Time Series Forecasting

This project explores **time series forecasting of the IBOVESPA index** (Brazilian stock market index) using machine learning techniques, with a strong emphasis on **correct time-series methodology**, **reproducibility**, and **clean ML engineering practices**.

Rather than aiming to outperform the market, the primary goal is to demonstrate how to design and evaluate a **realistic forecasting pipeline**, avoiding common pitfalls such as data leakage, improper validation, and misleading performance claims.

---

## 📌 Project Objectives

- Build a reusable **data ingestion and preprocessing pipeline**
- Perform **exploratory data analysis (EDA)** to uncover statistical and temporal patterns
- Train and evaluate **time series forecasting models**
- Compare predictive models against a **strong baseline**
- Emphasize engineering discipline over raw predictive performance

---

## 📊 Dataset

- **Source**: 
  - [Historical IBOVESPA index data](https://br.investing.com/indices/bovespa-historical-data)
  - [IPCA Brazil (IBGE)](https://www.ibge.gov.br/)
- **Frequency**: Daily
- **Target variable**: Closing price
- **Characteristics**:
  - Non-stationary
  - High volatility
  - Strong temporal dependency

---

## 🔍 Methodology

### Time Series Strategy

- Strict chronological splitting (no shuffling)
- Train and validation separation respecting time order
- Feature generation using only past information
- Explicit baseline comparison

### Models

- **LSTM (Long Short-Term Memory)** neural network
- **Naive persistence baseline** (next value equals last observed value)

The baseline is intentionally simple but very strong for financial time series and serves as a **reference point to assess whether model complexity adds real value**.

---

## 🚀 How to Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis
```bash
python src/run_eda.py
```

### 3. Train the Model
```bash
python src/run_train.py
```

### 4. Validate the Model
```bash
python src/run_validation.py
```

---

## 📈 Evaluation Metrics

Models are evaluated on **unseen future data** using:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE**
- **R²** Score

This ensures a realistic assessment of generalization performance.

---

## 🧠 Results

### LSTM Model Performance

- **MAE**: 1242.07  
- **MSE**: 2,278,960.50  
- **RMSE**: 1509.62  
- **R²**: 0.9468  

### Naive Persistence Baseline Performance

- **MAE**: 918.31  
- **MSE**: 1,280,635.00  
- **RMSE**: 1131.65  
- **R²**: 0.9701  

### Interpretation

The naive persistence baseline outperformed the LSTM across all evaluated metrics.

This result indicates that, for this dataset and forecasting horizon, **autocorrelation dominates predictive performance**, and increased model complexity does not yield meaningful improvements without the inclusion of exogenous variables.

This outcome is expected in financial time series and confirms that the forecasting pipeline, data splits, and evaluation strategy are **correct and free from data leakage**.

> **NOTE**: For a detailed discussion on exploratory data analysis (EDA) and machine learning experiments, see  
> **[ead_and_ml.md](ead_and_ml.md)**

---

## ⚠️ Limitations

- No external signals (macroeconomic indicators, interest rates, news, or sentiment)
- Target variable is the price level, which amplifies autocorrelation effects
- Models are sensitive to structural breaks and regime changes
- Financial markets are inherently noisy and partially efficient

This project prioritizes **methodological correctness and transparency** over aggressive optimization.

---

## 🔮 Possible Improvements

- Forecast returns or directional movement instead of prices
- Incorporate exogenous variables (e.g., inflation, global indices)
- Add probabilistic forecasting (prediction intervals)
- Compare against additional statistical models (ARIMA / SARIMA)
- Introduce experiment tracking (e.g., MLflow)

---

## 🧪 Engineering Highlights

- Modular and reproducible forecasting pipeline
- Clear separation between EDA, training, and validation
- Configuration-driven execution
- Deterministic behavior for reproducibility
- Explicit safeguards against data leakage

---

## 📂 Project Structure

```
├── data/
│ ├── brazil_inflation_index_ipca.csv
│ └── ibovespa_history.csv
├── graphs/
│ ├── *.png
├── src/
│ ├── run_eda.py
│ ├── run_train.py
│ ├── run_validation.py
│ └── utils/
│   ├── config.py
│   ├── constants.py
│   ├── cross_validator.py
│   ├── evaluator.py
│   ├── forecasting_pipeline.py
│   ├── ibovespa_loader.py
│   ├── ibovespa_preprocessor.py
│   ├── lstm_model.py
│   └── time_series_splitter.py
├── ead_and_ml.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

--- 

## 👤 Author 

Developed by **Alexandre** as part of a portfolio focused on **data engineering, machine learning, and applied time series analysis**. This project is intended to demonstrate **real-world ML discipline**, not trading advice. 

> ⚠️ Disclaimer: This project is for educational purposes only and does not constitute financial or investment advice.