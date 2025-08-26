# Enhanced Stock Price Prediction Using LSTM with Technical Indicators: A Comparative Study

## Project Overview

This repository contains the code and resources for a dissertation project that investigates the effectiveness of an LSTM-based deep learning model for stock price forecasting. The project conducts a rigorous comparative study, evaluating the LSTM model against two powerful and widely-used alternatives: the statistical ARIMAX model and the machine learning-based XGBoost model. The core objective is to determine which of these approaches provides the most robust and accurate predictions when applied to real-world financial data.

---

##  Methodology

The project's methodology is a multi-stage process designed for a fair and comprehensive comparison.

* **Data Collection:** Daily stock data for 12 major US companies (AAPL, MSFT, GOOGL, AMZN, etc.) was obtained from Yahoo Finance via the `yfinance` library, covering the period from 2015 to 2024.
* **Feature Engineering:** A comprehensive suite of technical indicators (e.g., RSI, MACD, Bollinger Bands) was calculated using the `pandas_ta` library.
* **Feature Selection:** An XGBoost model was used to calculate and aggregate feature importances across all stocks, identifying the most influential indicators to serve as input variables.
* **Model Implementation:**
    * **ARIMAX:** An `auto_arima` model was used to automatically determine the optimal parameters for each time series.
    * **XGBoost:** Hyperparameter tuning was performed using the `Optuna` framework with a time-series cross-validation strategy.
    * **LSTM:** A deep learning model was constructed using TensorFlow/Keras, with a `KerasTuner` `RandomSearch` for hyperparameter optimization.
* **Performance Evaluation:** All models were evaluated on an unseen test set using three key metrics:
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)
    * Mean Absolute Percentage Error (MAPE)

---

##  Key Findings

* **ARIMAX's Superiority:** The ARIMAX model emerged as the most consistently accurate and reliable predictor. It achieved the lowest absolute error (RMSE and MAE) for the majority of stocks, demonstrating that a well-structured traditional statistical approach can outperform modern machine learning models for this specific task.
* **XGBoost's Inconsistency:** Despite its reputation, the XGBoost model showed highly inconsistent performance. While it delivered excellent relative accuracy (low MAPE) on a few tickers, its high absolute error on others indicates its limitations in capturing long-term trends in time-series data.
* **LSTM's Potential:** The LSTM model was competitive and demonstrated strong potential, particularly on highly volatile stocks. However, its performance did not consistently surpass that of the ARIMAX model.

For a detailed breakdown of the results and a direct comparison with other studies, please refer to the `Final Model Performance Summary` table in the [analysis notebook](#usage).

---

##  Getting Started

To reproduce the results and run the project locally, follow these steps:

### Prerequisites

* Python 3.8+
* The following libraries, which can be installed via `pip`:
    `yfinance`, `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `statsmodels`, `pmdarima`, `xgboost`, `optuna`, `keras-tuner`, `pandas-ta`, `matplotlib`, `seaborn`, `joblib`

### Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The entire analysis pipeline is contained within a single Python script or Jupyter Notebook.

1.  Open the main script/notebook (e.g., `main_analysis.ipynb` or `run_project.py`).
2.  Ensure your Google Drive is mounted as specified in the code if you are running in a Google Colab environment.
3.  Run the code sequentially to download data, preprocess it, train the models, and generate the results and plots.

---

##  Contributions

This project was developed as part of a dissertation for a master's degree in Data Science at University of Hertfordshire.

Authored by:
- [Oreva Otiede]


