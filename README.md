# Lumina Analytics Suite 🔮

> **The Next-Generation AI-Powered Data Analysis Platform.**
> *From raw data to enterprise-grade insights in seconds.*

![Lumina Banner](https://img.shields.io/badge/Lumina-Analytics%20Suite-00ADB5?style=for-the-badge&logo=streamlit) 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Lumina** is a comprehensive, local-first data analytics application built on **Streamlit**. It bridges the gap between simple spreadsheets and complex coding environments, providing a GUI-based experience for advanced Machine Learning, Statistical Analysis, and Business Intelligence.

---

## 📊 Core Analytics Features

Lumina provides a robust foundation for everyday data tasks.

### 🔍 Data Monitor & Quality
*   **Data Health Check**: Automatically detect missing values, duplicates, and inconsistent types.
*   **Outlier Detection**: Statistical flagging of anomalies.
*   **Type Inference**: Smart detection of Categorical vs Numerical vs DateTime columns.

### 📈 Exploratory Data Analysis (EDA)
*   **Interactive Visuals**: Drag-and-drop plotting (Scatter, Line, Bar, Histograms).
*   **Correlation Matrix**: Heatmaps to identify feature relationships.
*   **Distribution Analysis**: Skewness and Kurtosis checks.

### 🤖 Predictive Modeling (Classic ML)
*   **Regression**: Linear, Ridge, Lasso, and ElasticNet.
*   **Classification**: Logistic Regression, Random Forest, and XGBoost.
*   **Model Comparison**: Automated leaderboard of model performance (Accuracy, R², etc.).
*   **Confusion Matrices**: Visual performance evaluation.

---

## ✨ Advanced Analytics 2.0 (New!)

Upgrade your insights with 6 research-grade professional suites.

### 1. 🧠 Explainability Suite
*Demystify "Black Box" models.*
*   **SHAP Analysis**: Global feature importance and individual prediction explanations (Waterfall plots).
*   **LIME**: Local interpretable model-agnostic explanations.
*   **Partial Dependence Plots (PDP)**: Visualize how specific features drive predictions.

### 2. 🤖 Deep Learning Suite
*Accessible Neural Networks.*
*   **Neural Network Builder**: Configure MLP Architectures (Layers, Activations).
*   **Real-time Training**: Watch Loss/Accuracy curves converge in real-time.
*   **AutoML**: Optional automated hyperparameter tuning.

### 3. 📝 NLP Suite
*Unlocking insights from text.*
*   **Topic Modeling (LDA)**: Uncover hidden themes in document collections.
*   **Sentiment Analysis**: Time-series sentiment tracking.
*   **Named Entity Recognition (NER)**: Extract people, organizations, and locations.
*   **Word Clouds**: Dynamic text summarization.

### 4. 📈 Advanced Time Series
*Forecasting the future.*
*   **Facebook Prophet**: Robust forecasting with seasonality support.
*   **ARIMA**: Classic statistical forecasting.
*   **Decomposition**: Trend, Seasonality, and Residual breakdown.
*   **Monte Carlo Simulation**: Stochastic risk assessment.

### 5. 📉 Advanced Statistics & Research
*Dissertation-level rigor.*
*   **Post-Hoc Testing**: Tukey's HSD for ANOVA.
*   **Mixed Effects Models**: Hierarchical linear modeling.
*   **Survival Analysis**: Kaplan-Meier Estimators and Log-Rank Tests.
*   **Multivariate Analysis**: MANOVA and Principal Component Analysis (PCA).

### 6. 💼 Business Intelligence (BI)
*Metrics that matter.*
*   **CLV (Customer Lifetime Value)**: Predictive BG/NBD models.
*   **Churn Prediction**: Risk scoring pipelines.
*   **Retention Heatmaps**: Cohort analysis.
*   **Price Elasticity**: Econometric modeling.

---

## 🏭 Enterprise Features

*   **Smart Narrative Engine**: Auto-generates text summaries of your data findings using NLP.
*   **Report Generation**: Export full analysis as **Microsoft Word (.docx)** or **PDF** reports.
*   **A/B Test Simulator**: Bayesian "Impact Phase" for simulating intervention outcomes.
*   **Security**: Local-first processing (data never leaves your machine).

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.9 or higher
*   pip (Python Package Manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/lumina-analytics.git
    cd lumina-analytics
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This strictly installs tested versions of `scikit-learn`, `statsmodels`, `shap`, `prophet`, etc.*

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Usage Workflow

1.  **Upload Data**: Drag & Drop CSV, Excel, JSON, or Parquet files.
2.  **Monitor**: Inspect data quality and fix issues.
3.  **Choose a Phase**: use the Sidebar to navigate between Core EDA, Predictive Modeling, or Advanced Suites.
4.  **Configure & Run**: Select variables, tune parameters, and visualize.
5.  **Export**: Generate a "Smart Report" or download charts.

---

## 🧪 Testing & Reliability

Lumina is strictly tested to ensure stability.

*   **Unit Tests**: Validate statistical formulas and utilities.
*   **Smoke Tests**: End-to-end verification of all Analytics Suites.
*   **Safety Wrappers**: `safe_plot` and `safe_dataframe` prevent UI crashes.

**Run the test suite:**
```bash
python3 -m unittest discover tests
# Or specifically run smoke tests:
python3 tests/test_smoke_suites.py
```

---

## 🤝 Contributing

We welcome contributions! Please see the `tests/` directory for guidance.

---

**Built with ❤️ by the Lumina Team.**
