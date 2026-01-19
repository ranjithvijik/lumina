# Lumina - Post-Graduate Statistics Upgrade

A research-grade statistical suite capable of handling complex dissertation-level analysis, with performance optimizations for large datasets.

## 🚀 Performance & Stability
- **Big Data Guards**: Automatic sampling (50k rows) for ML and GLM phases prevents memory crashes with massive datasets.
- **Monte Carlo Caching**: Simulation results are now cached, making parameter tuning instant.
- **Dendrogram Fix**: Resolved a critical missing import (`plotly.figure_factory`) that would have crashed hierarchial clustering.
- **Robust Datetime**: Enhanced logic correctly identifies time-series columns even without explicit Pandas types.

## 🛡️ Application Hardening (v8.0 Updates)
- **Anti-Crash UI Wrappers**: Implemented `safe_dataframe` and `safe_plot` to wrap all UI components.
    - **Fix 1 (Arrow Serialization)**: Automatically handles mixed-type columns (numbers/strings) which previously caused `ArrowTypeError` crashes on Streamlit Cloud.
    - **Fix 2 (Deprecations)**: Migrated 80+ plot calls to use `width="stretch"`, enforcing compatibility with Streamlit v1.53+.
- **Centralized Data Cleaning**: Implemented `clean_xy` helper to eliminate NaN propagation risks.
- **Defensive Modeling**: Added min-sample checks (N>10 for Regression, N>20 for ML).

## 🧪 Comprehensive Testing Suite
A new `tests/` directory ensures robustness against regressions:
- **Unit Tests**: Verify utility logic (`get_column_types`, `clean_xy`).
- **UI Safety Tests**: Verify that wrappers correctly strip deprecated arguments and sanitize data.
- **Integration Tests**: Verify application imports and syntax integrity.
- **Functional Tests**: Verify parsing of CSV, Excel, JSON, and Parquet files.

Run tests: 
```bash
python -m unittest discover tests
```

## 🎓 Refined Research Modules
### 1. ⏳ Survival Analysis (Log-Rank Test)
- **Automatic Statistical Comparison**: Stratified analysis automatically runs a **Log-Rank Test** (p-value reported).
- **Engine**: Uses `lifelines` (industry standard) with manual fallback.

### 2. 🔋 Power Analysis (Improved UX)
- **Dynamic Inputs**: Parameters like "Number of Groups (k)" appear logically *before* calculation.

## Full Workflow
1. **Upload**: Select multiple CSVs.
2. **Merge**: Join them on a common ID column.
3. **Analyze**: Run GLM (Log link) or Survival Analysis.
4. **Report**: Download the comprehensive Word report.

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
