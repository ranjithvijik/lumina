# Lumina Analytics Suite 🔮

> **The Next-Generation AI-Powered Data Analysis Platform.**
> *From raw data to enterprise-grade insights in seconds.*

![Lumina Banner](https://img.shields.io/badge/Lumina-Analytics%20Suite-00ADB5?style=for-the-badge&logo=streamlit) 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Lumina** is a comprehensive, local-first data analytics application built on **Streamlit**. It bridges the gap between simple spreadsheets and complex coding environments, providing a GUI-based experience for advanced Machine Learning, Statistical Analysis, and Business Intelligence.

---

## ✨ Features: Advanced Analytics 2.0

Lumina has been upgraded with **6 Professional Suites**, replacing disparate modules with cohesive, task-oriented workflows.

### 1. 🧠 Explainability Suite
*Demystify your "Black Box" models models.*
*   **SHAP Analysis**: Global feature importance and individual prediction explanations (Waterfall plots).
*   **LIME**: Local interpretable model-agnostic explanations.
*   **Partial Dependence Plots (PDP)**: Visualize how specific features drive predictions.
*   **Feature Interactions**: 2D interaction maps.

### 2. 🤖 Deep Learning Suite
*Accessible Neural Networks for everyone.*
*   **Neural Network Builder**: Configure MLP Architectures (Layers, Activation: ReLU/Tanh, Solvers).
*   **Training & Visuals**: Real-time Loss Curve visualization and R²/Accuracy metrics.
*   **AutoML Integration**: (Optional) Automated hyperparameter tuning.

### 3. 📝 NLP Suite
*Unlocking insights from text.*
*   **Topic Modeling (LDA)**: Uncover hidden themes in document collections.
*   **Sentiment Analysis**: Time-series sentiment tracking and polarity scoring.
*   **Named Entity Recognition (NER)**: Extract people, organizations, and locations.
*   **Word Clouds**: Dynamic, visually stunning text summarization.

### 4. 📈 Advanced Time Series
*Forecasting the future.*
*   **Facebook Prophet**: Robust forecasting for business time series (with seasonality).
*   **ARIMA**: Classic statistical forecasting.
*   **Decomposition**: Trend, Seasonality, and Residual breakdown.
*   **Monte Carlo Simulation**: Stochastic forecasting for risk assessment.

### 5. 📉 Advanced Statistics
*Research-grade statistical rigor.*
*   **Post-Hoc Testing**: Tukey's HSD for ANOVA follow-ups.
*   **Repeated Measures ANOVA**: For longitudinal studies.
*   **Mixed Effects Models**: Hierarchical linear modeling.
*   **Bayesian A/B Testing**: Probabilistic conversion rate comparison.

### 6. 💼 Business Intelligence (BI)
*Metrics that matter.*
*   **CLV (Customer Lifetime Value)**: Predict future customer value using **Lifetimes** (BG/NBD & Gamma-Gamma).
*   **Churn Prediction**: Random Forest-based risk scoring.
*   **Retention Heatmaps**: Cohort analysis for user retention.
*   **Price Elasticity**: Log-log regression models to optimize pricing.

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
2.  **Monitor & Clean**: Use the **Data Quality** phase to inspect missing values and outliers.
3.  **Choose a Suite**: Correctly navigate via the sidebar to your desired analysis (e.g., "Explainability").
4.  **Configure & Run**:
    *   Select your **Target** (y) and **Features** (X).
    *   Adjust parameters (epochs, layers, forecast horizon).
    *   Click "Run" to generate interactive Plotly visualizations.
5.  **Export**: Download charts as PNG or full reports as Word Docs.

---

## 🧪 Testing & Reliability

Lumina is strictly tested to ensure stability for enterprise use cases.

*   **Unit Tests**: Validate statistical formulas and data cleaning utilities.
*   **Smoke Tests**: End-to-end verification of all 6 Analytics Suites.
*   **Safety Wrappers**: The `safe_plot` and `safe_dataframe` utilities prevent UI crashes even with malformed data.

**Run the test suite:**
```bash
python3 -m unittest discover tests
# Or specifically run smoke tests:
python3 tests/test_smoke_suites.py
```

---

## 🤝 Contributing

We welcome contributions! Please see the `tests/` directory for guidance on adding new modules. Ensure all new features are accompanied by a smoke test in `test_smoke_suites.py`.

---

**Built with ❤️ by the Lumina Team.**
