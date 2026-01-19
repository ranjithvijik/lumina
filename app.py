import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, shapiro, levene, mannwhitneyu, kruskal, spearmanr, kendalltau
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
import io
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics Imports (Professional Phases)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score
try:
    import shap
except ImportError:
    shap = None
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from itertools import combinations
# Post-Grad Stats Imports
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
import statsmodels.genmod.families as families
import math

# Survival Analysis (Optional)
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False



# ============================================================================
# 1. CONFIGURATION & THEME
# ============================================================================
st.set_page_config(
    page_title="Lumina Analytics Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* GLOBAL THEME OVERRIDE (Forces Light Mode) */
        :root {
            --primary-color: #00ADB5;
            --background-color: #f4f6f9;
            --secondary-background-color: #ffffff;
            --text-color: #262730;
            --font: "Inter", sans-serif;
        }

        /* 1. Main Application */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* 2. Text & Headings */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, span {
            color: var(--text-color) !important;
        }

        /* 3. Inputs & Widgets (Force Dark Text on Light Background) */
        .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox, .stMultiSelect {
            color: var(--text-color) !important;
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
        }
        /* Fix Selectbox/MultiSelect rendering */
        div[data-baseweb="select"] {
            background-color: #ffffff !important;
        }
        div[data-baseweb="select"] span { 
            color: var(--text-color) !important; 
        }
        /* Input placeholder color */
        ::placeholder {
            color: #888 !important;
            opacity: 1;
        }

        /* 4. Sidebar (Optional Contrast Enhancement) */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] span {
             color: var(--text-color) !important;
        }

        /* 5. Phase Container (Dark Gradient) - Force White Text */
        .phase-container {
            background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: white !important;
        }
        .phase-container h1, .phase-container div, .phase-container span { 
            color: white !important; 
        }

        /* 6. Metrics (White Card) */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e1e4e8;
        }
        div[data-testid="stMetric"] label { color: #555 !important; } /* Label is medium grey */
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text-color) !important; }

        /* 7. Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: var(--text-color) !important;
        }
        .stTabs [aria-selected="true"] {
             color: #00ADB5 !important;
             font-weight: bold;
        }

        /* 8. Summary Boxes */
        .stat-summary {
            background-color: #f0f4f8; 
            padding: 12px; 
            border-left: 4px solid #00ADB5;
            border-radius: 6px; 
            margin-bottom: 12px;
            color: var(--text-color) !important;
        }
        .stat-summary strong { color: #00ADB5 !important; }
        
        /* 9. File Uploader Fix */
        [data-testid="stFileUploader"] {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            border: 1px dashed #e0e0e0;
        }
        [data-testid="stFileUploader"] section {
            background-color: #ffffff !important;
        }
        [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
             color: var(--text-color) !important;
        }
        /* Icon color */
        [data-testid="stFileUploader"] svg {
             fill: var(--text-color) !important;
        }
        /* Browse Button Fix */
        [data-testid="stFileUploader"] button {
             color: var(--text-color) !important;
             background-color: #ffffff !important;
             border: 1px solid #d0d7de !important;
        }
        [data-testid="stFileUploader"] button:hover {
             border-color: #00ADB5 !important;
             color: #00ADB5 !important;
        }
        
        /* 10. Buttons (Force standard text color & White Background) */
        div.stButton > button {
            color: var(--text-color) !important;
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
        }
        div.stButton > button:hover {
            border-color: #00ADB5 !important;
            color: #00ADB5 !important;
        }
        div.stButton > button p {
            color: inherit !important;
        }
        
        /* 11. Alerts (Success, Info, Warning, Error) */
        div[data-baseweb="notification"] p {
            color: var(--text-color) !important;
        }
        
        /* 12. Expander */
        .streamlit-expanderHeader {
            background-color: #ffffff !important;
            color: var(--text-color) !important;
        }
        .streamlit-expanderContent {
            background-color: #ffffff !important;
            color: var(--text-color) !important;
        }
        
        /* 13. Dataframes (Header Fix) */
        [data-testid="stDataFrame"] th {
            color: var(--text-color) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CORE ENGINES (INGESTION & UTILS)
# ============================================================================
# ============================================================================
# 2. FILE INGESTION ENGINE
# ============================================================================

@st.cache_data(show_spinner=False)
def parse_uploaded_file(uploaded_file):
    """Universal file parser handling multiple formats and multi-sheet Excel files."""
    try:
        filename = uploaded_file.name.lower()
        
        if filename.endswith(('.xlsx', '.xls')):
            xls_dict = pd.read_excel(uploaded_file, sheet_name=None)
            if len(xls_dict) > 1:
                return xls_dict, "multi"
            else:
                return list(xls_dict.values())[0], "single"
        elif filename.endswith('.csv'):
            return pd.read_csv(uploaded_file), "single"
        elif filename.endswith('.tsv') or filename.endswith('.txt'):
            return pd.read_csv(uploaded_file, sep='\t'), "single"
        elif filename.endswith('.json'):
            return pd.read_json(uploaded_file), "single"
        elif filename.endswith('.parquet'):
            return pd.read_parquet(uploaded_file), "single"
        else:
            return f"Unsupported file extension: {filename}", "error"
            
    except Exception as e:
        return f"Error parsing file: {str(e)}", "error"

def smart_date_converter(df):
    """Intelligent datetime conversion."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                if 'date' in col.lower() or 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
    return df

from contextlib import contextmanager

@contextmanager
def stat_box(title=None):
    st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
    if title:
        st.write(f"**{title}**")
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

def safe_dataframe(data, **kwargs):
    """Wrapper for safe_dataframe that handles Arrow compatibility and deprecations."""
    if data is None: return
    
    # Fix Arrow handling of mixed types in Object columns
    # We copy to display-only version to not affect analysis
    df_disp = data.copy()
    for col in df_disp.select_dtypes(include=['object']):
        try:
            df_disp[col] = df_disp[col].astype(str)
        except: pass
        
    # Handle deprecation: explicit width vs use_container_width
    if 'use_container_width' in kwargs:
        del kwargs['use_container_width'] # Remove old arg
    
    # Default to stretch if not specified (legacy behavior)
    if 'width' not in kwargs:
        kwargs['width'] = "stretch" # Streamlit 1.53+ syntax
        
    st.dataframe(df_disp, **kwargs)

def safe_plot(fig, **kwargs):
    """Wrapper for safe_plot that handles deprecations."""
    if fig is None: return
    
    if 'use_container_width' in kwargs:
        del kwargs['use_container_width']
        
    # Default to stretch
    st.plotly_chart(fig, width="stretch", **kwargs)

def get_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    return numeric, categorical, datetime_cols

def clean_xy(df, target, features):
    """Helper to ensure X and y are perfectly aligned and clean."""
    data = df[[target] + features].dropna()
    X = data[features]
    y = data[target]
    return X, y

# ============================================================================
# 6. CHART RECOMMENDATION ENGINE
# ============================================================================

def recommend_chart_type(x_col, y_col, df):
    if x_col is None: return "None"
    
    x_type = df[x_col].dtype
    y_type = df[y_col].dtype if y_col else None
    
    is_x_num = pd.api.types.is_numeric_dtype(x_type)
    is_x_cat = pd.api.types.is_object_dtype(x_type) or isinstance(x_type, pd.CategoricalDtype)
    is_x_date = pd.api.types.is_datetime64_any_dtype(x_type)
    
    is_y_num = pd.api.types.is_numeric_dtype(y_type) if y_type else False
    is_y_cat = pd.api.types.is_object_dtype(y_type) if y_type else False
    
    if not y_col:
        if is_x_num: return "Histogram"
        if is_x_cat: return "Bar Chart (Count)"
        if is_x_date: return "Line Chart (Frequency)"
    
    if is_x_date and is_y_num: return "Line Chart"
    if is_x_num and is_y_num: return "Scatter Plot"
    if is_x_cat and is_y_num: return "Box Plot"
    if is_x_num and is_y_cat: return "Box Plot (Horiz)"
    if is_x_cat and is_y_cat: return "Heatmap"
    
    return "Scatter Plot"

# ============================================================================
# 3. HELPER FUNCTIONS: FILE MERGING & STATS
# ============================================================================

def merge_datasets(dfs__param):
    """
    Merging UI for multiple dataframes
    """
    if len(dfs__param) == 1:
        return list(dfs__param.values())[0]
    
    st.subheader("🔗 Dataset Merger")
    st.info(f"Detected {len(dfs__param)} files. Choose how to verify them.")
    
    merge_mode = st.radio("Merge Logic", ["Concatenate (Stack)", "Join (Merge by Key)"], horizontal=True)
    
    if merge_mode == "Concatenate (Stack)":
        try:
            return pd.concat(dfs__param.values(), ignore_index=True)
        except Exception as e:
            st.error(f"Concat failed: {e}")
            return None
            
    else:
        # Simple iterative merge
        df_keys = list(dfs__param.keys())
        base_key = st.selectbox("Base Dataset", df_keys)
        base_df = dfs__param[base_key]
        
        secondary_keys = [k for k in df_keys if k != base_key]
        
        for k in secondary_keys:
            st.markdown(f"**Merging '{k}' into '{base_key}'...**")
            cols1 = base_df.columns.tolist()
            cols2 = dfs__param[k].columns.tolist()
            
            shared_cols = list(set(cols1) & set(cols2))
            if not shared_cols:
                st.error(f"No common columns between {base_key} and {k}")
                return base_df
                
            on_key = st.selectbox(f"Join Key for {k}", shared_cols)
            how = st.selectbox(f"Join Type for {k}", ["inner", "left", "right", "outer"], index=1)
            
            base_df = pd.merge(base_df, dfs__param[k], on=on_key, how=how)
            
        return base_df

# ============================================================================
# 3. ANALYSIS PHASES
# ============================================================================

# --- MONITOR PHASE ---
def render_monitor(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🔍 Monitor Phase</div>
            <div class="phase-subtitle">Data Health Check</div>
        </div>
    """, unsafe_allow_html=True)
    if df is None: return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Cells", df.isna().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Missing Data Pattern")
        fig = px.imshow(df.isnull(), aspect="auto", color_continuous_scale=[[0,"#2c3e50"],[1,"#FF2E63"]])
        fig.update_layout(yaxis_visible=False, coloraxis_showscale=False)
        safe_plot(fig, use_container_width=True)
    
    with c2:
        st.subheader("Data Types")
        dtypes = df.dtypes.value_counts().reset_index()
        dtypes.columns = ['Type', 'Count']
        dtypes['Type'] = dtypes['Type'].astype(str)
        fig_pie = px.pie(dtypes, values='Count', names='Type', hole=0.4)
        safe_plot(fig_pie, use_container_width=True)

    with st.expander("📋 Data Preview"):
        safe_dataframe(df.head(50))

# --- EXPLORE PHASE ---
def render_explore(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📊 Smart Explore</div>
            <div class="phase-subtitle">Auto-Adaptive Visualization Engine</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return

    num_cols, cat_cols, date_cols = get_column_types(df)
    all_cols = df.columns.tolist()

    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        
        with c1:
            x_col = st.selectbox("X Axis (Required)", all_cols, index=0)
        
        with c2:
            def_y_idx = 0
            if len(num_cols) > 0 and x_col not in num_cols:
                def_y_idx = all_cols.index(num_cols[0])
            y_col = st.selectbox("Y Axis (Optional)", ["None"] + all_cols, index=def_y_idx + 1 if "None" in ["None"] + all_cols else 0)
            if y_col == "None": y_col = None

        rec_chart = recommend_chart_type(x_col, y_col, df)
        
        with c3:
            color_col = st.selectbox("Color / Group", ["None"] + cat_cols + num_cols)
            if color_col == "None": color_col = None
            
        with c4:
            chart_types = [
                "Auto (Recommended)", "Scatter Plot", "Line Chart", "Area Chart", 
                "Bar Chart", "Bar Chart (Count)", "Histogram", "Box Plot", 
                "Violin Plot", "Density Heatmap", "Pie Chart", "Funnel Chart", "ECDF Plot"
            ]
            selected_type = st.selectbox("Chart Type", chart_types)

    final_chart = rec_chart if selected_type == "Auto (Recommended)" else selected_type

    try:
        fig = None
        
        if "Scatter" in final_chart:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols" if y_col and pd.api.types.is_numeric_dtype(df[y_col]) else None, 
                           title=f"{x_col} vs {y_col}")
        
        elif "Line" in final_chart or "Area" in final_chart:
            if y_col:
                if len(df) > 1000 and not pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    st.caption("ℹ️ Data aggregated by mean for readability.")
                    agg_df = df.groupby([x_col, color_col] if color_col else x_col)[y_col].mean().reset_index()
                else:
                    agg_df = df
                    
                func = px.area if "Area" in final_chart else px.line
                fig = func(agg_df, x=x_col, y=y_col, color=color_col, title=f"Trend of {y_col} by {x_col}")
            else:
                counts = df[x_col].value_counts().sort_index().reset_index()
                counts.columns = [x_col, 'Count']
                fig = px.line(counts, x=x_col, y='Count', title=f"Frequency of {x_col}")

        elif "Bar" in final_chart:
            if "Count" in final_chart or not y_col:
                fig = px.histogram(df, x=x_col, color=color_col, barmode="group", title=f"Count Distribution of {x_col}")
            else:
                agg_df = df.groupby([x_col, color_col] if color_col else x_col)[y_col].mean().reset_index()
                fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col, barmode="group", title=f"Mean {y_col} by {x_col}")

        elif final_chart == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, marginal="box", nbins=30, title=f"Distribution of {x_col}")
            
        elif "Box" in final_chart:
            orient = 'h' if "Horiz" in final_chart else 'v'
            x_ax = x_col if orient == 'v' else y_col
            y_ax = y_col if orient == 'v' else x_col
            fig = px.box(df, x=x_ax, y=y_ax, color=color_col, title=f"Box Plot")
                
        elif "Violin" in final_chart:
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True, title=f"Density of {y_col} by {x_col}")
            
        elif "ECDF" in final_chart:
            fig = px.ecdf(df, x=x_col, color=color_col, title=f"Cumulative Distribution")

        elif final_chart == "Heatmap":
            if y_col:
                ct = pd.crosstab(df[x_col], df[y_col])
                fig = px.imshow(ct, aspect="auto", color_continuous_scale="Viridis", title=f"Heatmap: {x_col} vs {y_col}")
            else:
                st.warning("Heatmap requires Y axis.")
                
        elif final_chart == "Density Heatmap":
            if y_col:
                fig = px.density_heatmap(df, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")

        elif final_chart == "Pie Chart":
            if y_col:
                fig = px.pie(df, values=y_col, names=x_col, title=f"Proportion")
            else:
                fig = px.pie(df, names=x_col, title=f"Count Proportion")
                
        elif final_chart == "Funnel Chart":
            if y_col:
                agg_df = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False)
                fig = px.funnel(agg_df, x=y_col, y=x_col, title=f"Funnel")

        if fig:
            fig.update_layout(template="plotly_white", height=600, font=dict(family="Inter, sans-serif"), hovermode="closest")
            safe_plot(fig, use_container_width=True)
        else:
            st.warning(f"Could not generate {final_chart} with selected columns.")
            
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")

# ============================================================================
# 8. DATA QUALITY & PROFILING PHASE (NEW)
# ============================================================================

def render_data_quality(df):
    """Professional data profiling & quality assessment"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🏥 Data Quality & Profiling</div>
            <div class="phase-subtitle">Comprehensive Data Health & Validation Report</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    # --- OVERALL QUALITY SCORE ---
    num_cols, cat_cols, date_cols = get_column_types(df)
    
    missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    duplicate_pct = df.duplicated().sum() / len(df) * 100
    quality_score = 100 - (missing_pct + duplicate_pct/2)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Quality Score", f"{quality_score:.1f}%", delta="Good" if quality_score > 80 else "Fair" if quality_score > 60 else "Poor")
    col2.metric("Completeness", f"{100-missing_pct:.1f}%")
    col3.metric("Uniqueness", f"{100-duplicate_pct:.1f}%")
    col4.metric("Columns Analyzed", len(df.columns))
    
    tabs = st.tabs(["Column Profile", "Missing Data", "Outliers", "Duplicates", "Data Types"])
    
    # --- TAB 1: COLUMN PROFILING ---
    with tabs[0]:
        st.subheader("Column-by-Column Analysis")
        
        profile_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            
            if col in num_cols:
                profile_data.append({
                    'Column': col,
                    'Type': 'Numeric',
                    'Missing': f"{missing} ({missing_pct:.1f}%)",
                    'Mean': f"{df[col].mean():.2f}",
                    'Std': f"{df[col].std():.2f}",
                    'Min': f"{df[col].min():.2f}",
                    'Max': f"{df[col].max():.2f}",
                    'Unique': df[col].nunique()
                })
            else:
                profile_data.append({
                    'Column': col,
                    'Type': 'Categorical',
                    'Missing': f"{missing} ({missing_pct:.1f}%)",
                    'Unique': df[col].nunique(),
                    'Top Value': df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else 'N/A',
                    'Top Freq': df[col].value_counts().values[0] if len(df[col].value_counts()) > 0 else 0
                })
        
        profile_df = pd.DataFrame(profile_data)
        safe_dataframe(profile_df, use_container_width=True)
    
    # --- TAB 2: MISSING DATA ANALYSIS ---
    with tabs[1]:
        st.subheader("Missing Data Patterns")
        
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isna().sum(),
            'Missing %': (df.isna().sum() / len(df)) * 100
        }).sort_values('Missing %', ascending=False)
        
        if missing_summary['Missing Count'].sum() > 0:
            fig = px.bar(missing_summary, x='Column', y='Missing %', title="Missing Data by Column",
                        color='Missing %', color_continuous_scale='Reds')
            safe_plot(fig, use_container_width=True)
            
            st.subheader("Missing Data Correlations")
            # Show which columns have missing values together
            missing_corr = df.isnull().corr()
            if missing_corr.shape[0] > 1:
                fig = go.Figure(data=go.Heatmap(z=missing_corr.values, x=missing_corr.columns, 
                                                 y=missing_corr.columns, colorscale='Blues'))
                safe_plot(fig, use_container_width=True)
            
            safe_dataframe(missing_summary, use_container_width=True)
        else:
            st.success("✓ No missing values detected!")
    
    # --- TAB 3: OUTLIER DETECTION ---
    with tabs[2]:
        st.subheader("Outlier Detection & Analysis")
        
        if num_cols:
            outlier_method = st.radio("Detection Method", ["IQR (Tukey)", "Z-Score", "Isolation Forest"])
            col_select = st.selectbox("Select Column", num_cols)
            
            data_clean = df[col_select].dropna()
            outliers = pd.Series([False] * len(data_clean), index=data_clean.index)
            
            if outlier_method == "IQR (Tukey)":
                Q1 = data_clean.quantile(0.25)
                Q3 = data_clean.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (data_clean < Q1 - 1.5*IQR) | (data_clean > Q3 + 1.5*IQR)
                threshold_info = f"IQR Range: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]"
            
            elif outlier_method == "Z-Score":
                z_scores = np.abs((data_clean - data_clean.mean()) / data_clean.std())
                outliers = z_scores > 3
                threshold_info = "Z-Score Threshold: |Z| > 3"
            
            else:  # Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(data_clean.values.reshape(-1, 1)) == -1
                threshold_info = "Isolation Forest (10% contamination)"
            
            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(data_clean)) * 100
            
            st.metric("Outliers Detected", f"{outlier_count} ({outlier_pct:.2f}%)")
            st.caption(threshold_info)
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(data_clean))), y=data_clean, 
                                    mode='markers', name='Normal',
                                    marker=dict(color='blue', size=6)))
            fig.add_trace(go.Scatter(x=data_clean[outliers].index, y=data_clean[outliers],
                                    mode='markers', name='Outliers',
                                    marker=dict(color='red', size=10)))
            fig.update_layout(title=f"Outlier Detection: {col_select}", height=500)
            safe_plot(fig, use_container_width=True)
            
            # Outlier statistics
            if outlier_count > 0:
                st.subheader("Outlier Statistics")
                outlier_vals = data_clean[outliers]
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean (Outliers)", f"{outlier_vals.mean():.2f}")
                col2.metric("Median (Outliers)", f"{outlier_vals.median():.2f}")
                col3.metric("Std (Outliers)", f"{outlier_vals.std():.2f}")
        else:
            st.warning("No numeric columns available.")
    
    # --- TAB 4: DUPLICATE ANALYSIS ---
    with tabs[3]:
        st.subheader("Duplicate Row Detection")
        
        subset_cols = st.multiselect("Check Duplicates in Columns", df.columns, 
                                     default=df.columns.tolist()[:5])
        
        if subset_cols:
            dup_counts = df.duplicated(subset=subset_cols).sum()
            st.metric("Duplicate Rows", dup_counts)
            
            if dup_counts > 0:
                duplicates = df[df.duplicated(subset=subset_cols, keep=False)].sort_values(subset_cols)
                safe_dataframe(duplicates.head(20), use_container_width=True)
                st.caption(f"Showing first 20 of {len(duplicates)} duplicate rows")
            else:
                st.success("✓ No duplicates found!")
    
    # --- TAB 5: DATA TYPE ANALYSIS ---
    with tabs[4]:
        st.subheader("Data Type Distribution & Format Validation")
        
        type_summary = pd.DataFrame({
            'Data Type': df.dtypes.value_counts().index.astype(str),
            'Count': df.dtypes.value_counts().values
        })
        
        fig = px.pie(type_summary, values='Count', names='Data Type', title="Data Type Distribution")
        safe_plot(fig, use_container_width=True)
        
        st.subheader("Format Validation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Date Columns Found:**")
            if date_cols:
                for col in date_cols:
                    st.write(f"  • {col}")
            else:
                st.write("  None detected")
        
        with col2:
            st.write("**Categorical Columns:**")
            if cat_cols:
                for col in cat_cols[:5]:
                    unique_count = df[col].nunique()
                    st.write(f"  • {col} ({unique_count} unique)")
                if len(cat_cols) > 5:
                    st.write(f"  ... and {len(cat_cols)-5} more")
            else:
                st.write("  None detected")

# ============================================================================
# 9. ANOMALY DETECTION & OUTLIER TREATMENT (NEW)
# ============================================================================

def render_anomaly_detection(df):
    """Advanced anomaly detection with multiple algorithms"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🔎 Anomaly Detection</div>
            <div class="phase-subtitle">Identify & Analyze Unusual Patterns & Behaviors</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, date_cols = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    
    # Multivariate anomaly detection
    features = st.multiselect("Select Features for Anomaly Detection", num_cols, 
                             default=num_cols[:min(3, len(num_cols))])
    
    if not features:
        st.error("Select at least one feature.")
        return
    
    contamination = st.slider("Contamination Rate (expected % of anomalies)", 0.01, 0.2, 0.05)
    
    data_clean = df[features].dropna()
    try:
        if data_clean.empty:
            st.warning("No data available for clustering after cleaning.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean)
    except ValueError as e:
        st.error(f"Scaling Error: {e}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # --- ISOLATION FOREST ---
    with col1:
        st.subheader("Isolation Forest")
        iso_model = IsolationForest(contamination=contamination, random_state=42)
        iso_pred = iso_model.fit_predict(X_scaled)
        iso_anomalies = iso_pred == -1
        
        st.metric("Anomalies Found", iso_anomalies.sum())
        
        # Feature importance via decision path depth
        if len(features) == 1:
            anomaly_scores = iso_model.score_samples(X_scaled)
            fig = px.histogram(x=anomaly_scores, nbins=30, 
                             title="Anomaly Scores (Isolation Forest)")
            safe_plot(fig, use_container_width=True)
    
    # --- LOCAL OUTLIER FACTOR ---
    with col2:
        st.subheader("Local Outlier Factor")
        lof_model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_pred = lof_model.fit_predict(X_scaled)
        lof_anomalies = lof_pred == -1
        
        st.metric("Anomalies Found", lof_anomalies.sum())
        
        lof_scores = lof_model.negative_outlier_factor_
        fig = px.histogram(x=lof_scores, nbins=30,
                          title="LOF Scores (Local Outlier Factor)")
        safe_plot(fig, use_container_width=True)
    
    # --- ENSEMBLE CONSENSUS ---
    with col3:
        st.subheader("Ensemble Consensus")
        consensus = iso_anomalies.astype(int) + lof_anomalies.astype(int)
        ensemble_anomalies = consensus >= 1  # Flagged by at least one method
        
        st.metric("Consensus Anomalies", ensemble_anomalies.sum())
        
        agreement_df = pd.DataFrame({
            'Both Methods': (iso_anomalies & lof_anomalies).sum(),
            'Isolation Only': (iso_anomalies & ~lof_anomalies).sum(),
            'LOF Only': (~iso_anomalies & lof_anomalies).sum(),
            'Neither': (~iso_anomalies & ~lof_anomalies).sum()
        }, index=[0]).T
        
        fig = px.bar(agreement_df, title="Method Agreement")
        safe_plot(fig, use_container_width=True)
    
    # --- VISUALIZATION ---
    st.subheader("2D Projection (PCA)")
    if len(features) > 1:
        pca = PCA(2)
        X_pca = pca.fit_transform(X_scaled)
        
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Ensemble': ensemble_anomalies.astype(str)
        })
        
        fig = px.scatter(viz_df, x='PC1', y='PC2', color='Ensemble',
                        title="Anomalies in PCA Space",
                        color_discrete_map={'True': 'red', 'False': 'blue'})
        safe_plot(fig, use_container_width=True)
    
    # --- ANOMALY DETAILS ---
    st.subheader("Anomalous Records")
    anomaly_indices = np.where(ensemble_anomalies)[0]
    
    if len(anomaly_indices) > 0:
        n_show = min(10, len(anomaly_indices))
        anomaly_records = data_clean.iloc[anomaly_indices[:n_show]]
        safe_dataframe(anomaly_records, use_container_width=True)
        st.caption(f"Showing {n_show} of {len(anomaly_indices)} anomalous records")
    else:
        st.success("✓ No anomalies detected!")

# ============================================================================
# 10. FEATURE ENGINEERING & OPTIMIZATION (NEW)
# ============================================================================

def render_feature_engineering(df):
    """Advanced feature engineering & transformation"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">⚙️ Feature Engineering</div>
            <div class="phase-subtitle">Create & Transform Features for Modeling</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, date_cols = get_column_types(df)
    
    st.subheader("Feature Transformation Options")
    
    with st.expander("1️⃣ Polynomial Features", expanded=False):
        if len(num_cols) > 0:
            poly_cols = st.multiselect("Select Columns for Polynomial Features", num_cols)
            poly_degree = st.slider("Polynomial Degree", 2, 4, 2)
            
            if poly_cols and st.button("Generate Polynomial Features"):
                data_poly = df[poly_cols].dropna()
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                poly_features = poly.fit_transform(data_poly)
                
                feature_names = poly.get_feature_names_out(poly_cols)
                poly_df = pd.DataFrame(poly_features, columns=feature_names)
                
                st.success(f"✓ Generated {len(feature_names)} polynomial features")
                safe_dataframe(poly_df.head(), use_container_width=True)
                
                csv = poly_df.to_csv(index=False)
                st.download_button("Download Polynomial Features", csv, 
                                 "poly_features.csv", "text/csv")
    
    with st.expander("2️⃣ Interaction Terms", expanded=False):
        if len(num_cols) > 1:
            inter_cols = st.multiselect("Select Columns for Interactions", num_cols)
            
            if len(inter_cols) >= 2 and st.button("Generate Interactions"):
                interaction_features = {}
                
                for col1, col2 in combinations(inter_cols, 2):
                    interaction_features[f"{col1}_×_{col2}"] = df[col1] * df[col2]
                
                inter_df = pd.DataFrame(interaction_features)
                st.success(f"✓ Generated {len(interaction_features)} interaction terms")
                safe_dataframe(inter_df.head(), use_container_width=True)
    
    with st.expander("3️⃣ Feature Scaling & Normalization", expanded=False):
        if len(num_cols) > 0:
            scale_cols = st.multiselect("Select Columns to Scale", num_cols)
            scale_method = st.radio("Scaling Method", 
                                   ["StandardScaler (Z-score)", 
                                    "RobustScaler (Outlier-resistant)",
                                    "MinMax Scaling (0-1)"])
            
            if scale_cols and st.button("Apply Scaling"):
                data_scale = df[scale_cols].copy()
                
                if scale_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                elif scale_method == "RobustScaler (Outlier-resistant)":
                    scaler = RobustScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                
                scaled_data = scaler.fit_transform(data_scale)
                scaled_df = pd.DataFrame(scaled_data, columns=[f"{c}_scaled" for c in scale_cols])
                
                st.success(f"✓ Applied {scale_method}")
                safe_dataframe(scaled_df.head(), use_container_width=True)
    
    with st.expander("4️⃣ Categorical Encoding", expanded=False):
        if len(cat_cols) > 0:
            encode_col = st.selectbox("Select Categorical Column", cat_cols)
            encode_method = st.radio("Encoding Method",
                                    ["One-Hot Encoding", "Label Encoding", "Frequency Encoding"])
            
            if st.button("Apply Encoding"):
                if encode_method == "One-Hot Encoding":
                    encoded = pd.get_dummies(df[[encode_col]], prefix=encode_col)
                    st.success("✓ One-Hot Encoding Applied")
                
                elif encode_method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    encoded = pd.DataFrame({
                        f"{encode_col}_encoded": le.fit_transform(df[encode_col].astype(str))
                    })
                    st.success("✓ Label Encoding Applied")
                
                else:  # Frequency Encoding
                    freq_map = df[encode_col].value_counts().to_dict()
                    encoded = pd.DataFrame({
                        f"{encode_col}_freq": df[encode_col].map(freq_map)
                    })
                    st.success("✓ Frequency Encoding Applied")
                
                safe_dataframe(encoded.head(), use_container_width=True)

# ============================================================================
# 11. ADVANCED VISUALIZATION: CORRELATION HEATMAP WITH SIGNIFICANCE
# ============================================================================

def render_correlation_analysis(df):
    """Professional correlation analysis with dendrograms & significance"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📊 Correlation & Association Analysis</div>
            <div class="phase-subtitle">Relationship Strength & Statistical Significance</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    
    # --- PEARSON CORRELATION WITH P-VALUES ---
    st.subheader("Correlation Matrix with Statistical Significance")
    
    corr_matrix = df[num_cols].corr()
    
    # Calculate p-values
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                           index=corr_matrix.index, 
                           columns=corr_matrix.columns)
    
    for i, col1 in enumerate(num_cols):
        for j, col2 in enumerate(num_cols):
            if i != j:
                valid_data = df[[col1, col2]].dropna()
                if len(valid_data) > 2:
                    _, p_val = stats.pearsonr(valid_data[col1], valid_data[col2])
                    p_values.iloc[i, j] = p_val
    
    # Create annotated heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(title="Pearson Correlation Matrix", height=600)
    safe_plot(fig, use_container_width=True)
    
    # --- CORRELATION STRENGTH INTERPRETATION ---
    st.subheader("Strong Correlations (|r| > 0.7)")
    
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_values.iloc[i, j]
            
            if abs(corr_val) > 0.7 and p_val < 0.05:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                strong_corr.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': f"{corr_val:.4f}",
                    'P-Value': f"{p_val:.4e}",
                    'Significance': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                })
    
    if strong_corr:
        safe_dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    else:
        st.info("No strong correlations detected.")
    
    # --- HIERARCHICAL CLUSTERING DENDROGRAM ---
    st.subheader("Correlation Dendrogram (Hierarchical Clustering)")
    
    if len(num_cols) > 2:
        # Distance matrix from correlation
        distance_matrix = 1 - np.abs(corr_matrix)
        # Handle NaNs in distance matrix (e.g., perfect correlation or constant)
        distance_matrix = distance_matrix.fillna(0)
        
        # Use values for pdist to treat rows as vectors
        linkage_matrix = linkage(pdist(distance_matrix.values, metric='euclidean'), method='ward')
        
        fig = ff.create_dendrogram(distance_matrix, labels=distance_matrix.columns, linkagefun=lambda x: linkage_matrix)
        
        # Create dendrogram plot using scipy
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        dendro = scipy_dendrogram(linkage_matrix, labels=corr_matrix.columns, 
                                 no_plot=True)
        
        icoord = np.array(dendro['icoord'])
        dcoord = np.array(dendro['dcoord'])
        
        for i in range(len(icoord)):
            fig.add_trace(go.Scatter(x=icoord[i], y=dcoord[i], 
                                    mode='lines', line_color='grey',
                                    hoverinfo='none', showlegend=False))
        
        fig.update_xaxes(ticktext=dendro['ivl'], tickvals=list(range(10, 10*len(dendro['ivl']), 10)))
        fig.update_layout(title="Correlation Dendrogram", 
                         xaxis_title="Features", yaxis_title="Distance",
                         height=500)
        safe_plot(fig, use_container_width=True)
    else:
        st.info("Need more than 2 variables for dendrogram.")

# --- STATISTICAL TEST PHASE ---
def render_statistical_test(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📉 Statistical Testing</div>
            <div class="phase-subtitle">Hypothesis Testing & Distribution Analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    if not num_cols:
        st.warning("No numeric columns available for statistical testing.")
        return
    
    test_type = st.selectbox(
        "Select Test Type",
        [
            "Normality Test (Shapiro-Wilk)",
            "Equal Variance Test (Levene)",
            "T-Test (Independent Samples)",
            "Mann-Whitney U Test",
            "One-Way ANOVA",
            "Kruskal-Wallis Test",
            "Correlation Analysis",
            "Chi-Square Test of Independence"
        ]
    )
    
    col1, col2 = st.columns(2)
    
    # --- NORMALITY TEST ---
    if test_type == "Normality Test (Shapiro-Wilk)":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        
        data_clean = df[var].dropna()
        if len(data_clean) < 3:
            st.error("Need at least 3 observations.")
            return
        
        stat, p_value = shapiro(data_clean)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**Shapiro-Wilk Test for Normality**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Statistic", f"{stat:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}")
        c3.metric("Result", "✓ Normal" if p_value > 0.05 else "✗ Not Normal")
        st.write("**Null Hypothesis**: Data is normally distributed")
        st.write(f"**Conclusion**: {'Fail to reject H0 (data appears normal)' if p_value > 0.05 else 'Reject H0 (data is not normal)'}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "Q-Q Plot"))
        
        fig.add_trace(go.Histogram(x=data_clean, name="Distribution", nbinsx=30), row=1, col=1)
        
        from scipy.stats import probplot
        qq = probplot(data_clean)
        # qq[0][0] = theoretical quantiles (x)
        # qq[0][1] = ordered values (y)
        # qq[1] = (slope, intercept, _)
        slope, intercept, _ = qq[1]
        
        # Calculate regression line points
        x_trend = np.array([min(qq[0][0]), max(qq[0][0])])
        y_trend = slope * x_trend + intercept
        
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data Points'), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name='Reference Line'), row=1, col=2)
        
        safe_plot(fig, use_container_width=True)

    # --- EQUAL VARIANCE TEST ---
    elif test_type == "Equal Variance Test (Levene)":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        with col2:
            group = st.selectbox("Select Grouping Variable", cat_cols)
        
        groups = [group_data[var].dropna().values for name, group_data in df.groupby(group)]
        
        if len(groups) < 2:
            st.error("Need at least 2 groups.")
            return
        
        stat, p_value = levene(*groups)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**Levene's Test for Equal Variances**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Statistic", f"{stat:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}")
        c3.metric("Result", "✓ Equal" if p_value > 0.05 else "✗ Unequal")
        st.write("**Null Hypothesis**: All groups have equal variance")
        st.write(f"**Conclusion**: {'Fail to reject H0 (equal variances)' if p_value > 0.05 else 'Reject H0 (unequal variances)'}")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- T-TEST ---
    elif test_type == "T-Test (Independent Samples)":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        with col2:
            group = st.selectbox("Select Grouping Variable", cat_cols)
        
        unique_groups = df[group].unique()
        if len(unique_groups) != 2:
            st.error("This test requires exactly 2 groups.")
            return
        
        group1, group2 = unique_groups[0], unique_groups[1]
        group1_data = df[df[group] == group1][var].dropna()
        group2_data = df[df[group] == group2][var].dropna()
        
        if len(group1_data) < 3 or len(group2_data) < 3:
            st.warning("Need at least 3 observations per group.")
            return

        stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        
        # Calculate Cohen's d
        n1, n2 = len(group1_data), len(group2_data)
        s1, s2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / s_pooled
        
        with stat_box(f"T-Test: {group1} vs {group2}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T-Statistic", f"{stat:.4f}")
            c2.metric("P-Value", f"{p_val:.4f}")
            c3.metric("Cohen's d", f"{cohens_d:.4f}")
            c4.metric("Result", "✓ Significant" if p_val < 0.05 else "✗ Not Sig.")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=group1_data, name=str(group1)))
        fig.add_trace(go.Box(y=group2_data, name=str(group2)))
        fig.update_layout(title="Group Comparison Boxplot")
        safe_plot(fig, use_container_width=True)

    # --- MANN-WHITNEY U TEST ---
    elif test_type == "Mann-Whitney U Test":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        with col2:
            group = st.selectbox("Select Grouping Variable", cat_cols)
        
        groups = df[group].unique()
        if len(groups) != 2:
            st.error("This test requires exactly 2 groups.")
            return
        
        group1_data = df[df[group] == groups[0]][var].dropna()
        group2_data = df[df[group] == groups[1]][var].dropna()
        
        stat, p_value = mannwhitneyu(group1_data, group2_data)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**Mann-Whitney U Test (Non-parametric)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("U-Statistic", f"{stat:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}")
        c3.metric("Result", "✓ Significant" if p_value < 0.05 else "✗ Not Sig.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- ONE-WAY ANOVA ---
    elif test_type == "One-Way ANOVA":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        with col2:
            group = st.selectbox("Select Grouping Variable", cat_cols)
        
        groups = [group_data[var].dropna().values for name, group_data in df.groupby(group)]
        
        if len(groups) < 2:
            st.error("Need at least 2 groups.")
            return
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**One-Way ANOVA**")
        c1, c2, c3 = st.columns(3)
        c1.metric("F-Statistic", f"{f_stat:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}")
        c3.metric("Result", "✓ Significant" if p_value < 0.05 else "✗ Not Sig.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- KRUSKAL-WALLIS ---
    elif test_type == "Kruskal-Wallis Test":
        with col1:
            var = st.selectbox("Select Variable", num_cols)
        with col2:
            group = st.selectbox("Select Grouping Variable", cat_cols)
        
        # Validation
        if len(df[group].unique()) < 2:
            st.error("Need at least 2 groups.")
            return

        groups = [group_data[var].dropna().values for name, group_data in df.groupby(group)]
        
        h_stat, p_value = kruskal(*groups)
        
        with stat_box("Kruskal-Wallis Test (Non-parametric)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("H-Statistic", f"{h_stat:.4f}")
            c2.metric("P-Value", f"{p_value:.4f}")
            c3.metric("Result", "✓ Significant" if p_value < 0.05 else "✗ Not Sig.")

    # --- CORRELATION ANALYSIS ---
    elif test_type == "Correlation Analysis":
        with col1:
            var1 = st.selectbox("Select First Variable", num_cols)
        with col2:
            var2 = st.selectbox("Select Second Variable", num_cols)
        
        if var1 == var2:
            st.error("Select different variables.")
            return
        
        data_clean = df[[var1, var2]].dropna()
        
        pearson_r, pearson_p = stats.pearsonr(data_clean[var1], data_clean[var2])
        spearman_r, spearman_p = spearmanr(data_clean[var1], data_clean[var2])
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**Correlation: {var1} vs {var2}**")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson r", f"{pearson_r:.4f}")
        c2.metric("Pearson p", f"{pearson_p:.4f}")
        c3.metric("Spearman ρ", f"{spearman_r:.4f}")
        c4.metric("Spearman p", f"{spearman_p:.4f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Scatter with trendline
        fig = px.scatter(data_clean, x=var1, y=var2, trendline="ols", 
                        title=f"Correlation: {var1} vs {var2}")
        safe_plot(fig, use_container_width=True)

    # --- CHI-SQUARE TEST ---
    elif test_type == "Chi-Square Test of Independence":
        with col1:
            var1 = st.selectbox("Select First Categorical Variable", cat_cols)
        with col2:
            var2 = st.selectbox("Select Second Categorical Variable", cat_cols)
        
        ct = pd.crosstab(df[var1], df[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write(f"**Chi-Square Test of Independence**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chi-Square", f"{chi2:.4f}")
        c2.metric("P-Value", f"{p_value:.4f}")
        c3.metric("DOF", dof)
        c4.metric("Result", "✓ Significant" if p_value < 0.05 else "✗ Not Sig.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Contingency Table")
        safe_dataframe(ct)

# --- REGRESSION (STATISTICAL) PHASE ---
def render_regression(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🔧 Regression Analysis</div>
            <div class="phase-subtitle">Linear & Machine Learning Modeling</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox("Select Target Variable (Y)", num_cols)
    with col2:
        features = st.multiselect("Select Features (X)", [c for c in num_cols if c != target], default=[c for c in num_cols[:2] if c != target])
    
    if not features:
        st.error("Select at least one feature.")
        return
    
    model_type = st.radio("Model Type", ["Linear Regression (OLS)", "Multiple Regression", "Random Forest Regression"])
    
    # Prepare data
    data_clean = df[features + [target]].dropna()
    X = data_clean[features]
    y = data_clean[target]
    
    if len(X) < 10:
        st.error("Need at least 10 observations.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- LINEAR REGRESSION ---
    if model_type == "Linear Regression (OLS)":
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write("**Regression Summary**")
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared", f"{model.rsquared:.4f}")
        c2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
        c3.metric("F-Statistic", f"{model.fvalue:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Coefficients")
        coef_df = pd.DataFrame({
            'Coefficient': model.params,
            'Std Err': model.bse,
            't-Statistic': model.tvalues,
            'P-Value': model.pvalues
        })
        safe_dataframe(coef_df)
        
        # VIF for multicollinearity
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        try:
            # Ensure safe VIF calculation
            if X.shape[1] > 1:
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            else:
                vif_data["VIF"] = [0] # Not applicable for single feature
        except Exception:
            vif_data["VIF"] = [0] * len(features)
        
        st.subheader("Multicollinearity Check (VIF)")
        safe_dataframe(vif_data)
        st.caption("VIF > 10 indicates high multicollinearity")

    # --- MULTIPLE REGRESSION WITH FORMULA ---
    elif model_type == "Multiple Regression":
        formula = f"{target} ~ {' + '.join(features)}"
        model = ols(formula, data=data_clean).fit()
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write("**Multiple Regression Summary**")
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared", f"{model.rsquared:.4f}")
        c2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
        c3.metric("F-Statistic", f"{model.fvalue:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Full Model Summary")
        st.text(str(model.summary()))

    # --- RANDOM FOREST ---
    elif model_type == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        st.markdown("<div class='stat-summary'>", unsafe_allow_html=True)
        st.write("**Random Forest Performance**")
        c1, c2 = st.columns(2)
        c1.metric("Train R²", f"{train_score:.4f}")
        c2.metric("Test R²", f"{test_score:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (Random Forest)")
        safe_plot(fig, use_container_width=True)
        
        # Predictions vs Actual
        y_pred = model.predict(X_test)
        pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df['Actual'], y=pred_df['Predicted'], mode='markers', name='Predictions'))
        fig.add_trace(go.Scatter(x=[pred_df['Actual'].min(), pred_df['Actual'].max()], 
                                 y=[pred_df['Actual'].min(), pred_df['Actual'].max()], 
                                 mode='lines', name='Perfect Fit'))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        safe_plot(fig, use_container_width=True)

# ============================================================================
# 12. PREDICTIVE MODELING - CLASSIFICATION & COMPARISON (NEW)
# ============================================================================

def render_predictive_modeling(df):
    """Production-grade predictive modeling with model comparison"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🤖 Predictive Modeling</div>
            <div class="phase-subtitle">Classification, Regression & Model Comparison</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    # --- MODEL SELECTION ---
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    
    if problem_type == "Classification":
        target = st.selectbox("Select Target Variable", cat_cols if cat_cols else num_cols)
        
        if target in df.columns and df[target].nunique() <= 10:
            features = st.multiselect("Select Features", 
                                    [c for c in num_cols if c != target],
                                    default=num_cols[:min(3, len(num_cols))])
            
            if features:
                # Performance Guard
                if len(df) > 50000:
                    st.warning("⚠️ Large dataset detected. Sampling to 50k rows for performance.")
                    df = df.sample(50000, random_state=42)

                # Clean Data
                X, y = clean_xy(df, target, features)
                
                if len(X) < 20: 
                    st.error("Not enough data for split (min 20 samples).")
                    return

                # Encode Target if Categorical
                if problem_type == "Classification": # Assuming problem_type is the task_type
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    target_labels = le.classes_
                
                test_size = 0.2 # Define test_size as it's used in train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                   random_state=42, 
                                                                   stratify=y if len(np.unique(y)) > 1 else None)
                
                # --- MODEL COMPARISON ---
                st.subheader("Model Comparison")
                
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
                }
                
                results = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y)) == 2 else None
                    
                    results[name] = {
                        'Model': model,
                        'Accuracy': accuracy,
                        'AUC': auc if auc else np.nan,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                
                # Display results
                results_df = pd.DataFrame({
                    'Model': results.keys(),
                    'Accuracy': [results[m]['Accuracy'] for m in results.keys()],
                    'AUC': [results[m]['AUC'] for m in results.keys()]
                })
                
                fig = px.bar(results_df, x='Model', y='Accuracy', title="Model Accuracy Comparison",
                           color='Accuracy', color_continuous_scale='Viridis')
                safe_plot(fig, use_container_width=True)
                
                # Best model analysis
                best_model_name = max(results.items(), 
                                     key=lambda x: x[1]['Accuracy'])[0]
                best_result = results[best_model_name]
                
                st.subheader(f"Best Model: {best_model_name}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{best_result['Accuracy']:.4f}")
                if not np.isnan(best_result['AUC']):
                    col2.metric("AUC-ROC", f"{best_result['AUC']:.4f}")
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix as conf_matrix
                cm = conf_matrix(y_test, best_result['y_pred'])
                
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                              x=['Predicted Negative', 'Predicted Positive'],
                              y=['Actual Negative', 'Actual Positive'],
                              color_continuous_scale='Blues')
                safe_plot(fig, use_container_width=True)
                
                # Feature importance
                if hasattr(best_result['Model'], 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': best_result['Model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title="Feature Importance")
                    safe_plot(fig, use_container_width=True)
                
                # --- METRIC VISUALIZATION ---
                if len(np.unique(y_test)) == 2:
                    from sklearn.metrics import precision_recall_curve, f1_score
                    y_pred = best_result['y_pred']
                    try:
                        f1 = f1_score(y_test, y_pred)
                        st.metric("F1 Score", f"{f1:.4f}")
                    except: pass
    
    else:  # Regression
        target = st.selectbox("Select Target Variable", num_cols)
        
        features = st.multiselect("Select Features",
                                [c for c in num_cols if c != target],
                                default=num_cols[:min(3, len(num_cols))])
        
        if features:
            # Performance Guard
            if len(df) > 50000:
                st.warning("⚠️ Large dataset detected. Sampling to 50k rows for performance.")
                df = df.sample(50000, random_state=42)

            X, y = clean_xy(df, target, features)
            
            if len(X) < 20:
                st.error("Need at least 20 observations.")
                return
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                'Linear Regression': LinearRegression(),
                'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae, 'y_pred': y_pred}
            
            results_df = pd.DataFrame(results).T
            safe_dataframe(results_df, use_container_width=True)

# ============================================================================
# 13. BUSINESS INTELLIGENCE - RFM & COHORT ANALYSIS (NEW)
# ============================================================================

def render_business_analytics(df):
    """RFM Analysis, Cohort Analysis, Funnel Analysis"""
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">💼 Business Analytics</div>
            <div class="phase-subtitle">Customer Segmentation, RFM Analysis & Funnel Metrics</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None:
        st.info("Please upload a file to begin.")
        return
    
    num_cols, cat_cols, date_cols = get_column_types(df)
    
    analysis_type = st.selectbox("Select Analysis Type",
                               ["RFM Segmentation", "Cohort Analysis", 
                                "Funnel Analysis", "Conversion Funnel"])
    
    if analysis_type == "RFM Segmentation":
        st.subheader("RFM Analysis (Recency, Frequency, Monetary)")
        
        if 'customer_id' in df.columns or 'id' in df.columns:
            id_col = 'customer_id' if 'customer_id' in df.columns else 'id'
            
            date_col = st.selectbox("Select Date Column", date_cols if date_cols else cat_cols)
            amount_col = st.selectbox("Select Amount Column", num_cols)
            
            if id_col and date_col and amount_col:
                # Calculate RFM
                reference_date = pd.to_datetime(df[date_col]).max()
                
                rfm = df.groupby(id_col).agg({
                    date_col: lambda x: (reference_date - pd.to_datetime(x).max()).days,
                    id_col: 'count',
                    amount_col: 'sum'
                }).rename(columns={
                    date_col: 'Recency',
                    id_col: 'Frequency',
                    amount_col: 'Monetary'
                })
                
                # Scoring
                for col in ['Recency', 'Frequency', 'Monetary']:
                    rfm[f'{col}_Score'] = pd.qcut(rfm[col], 5, labels=[5,4,3,2,1], duplicates='drop')
                
                rfm['RFM_Score'] = rfm['Recency_Score'].astype(int) + rfm['Frequency_Score'].astype(int) + rfm['Monetary_Score'].astype(int)
                
                # Segment
                def rfm_segment(score):
                    if score >= 13: return 'Champions'
                    elif score >= 10: return 'Loyal'
                    elif score >= 7: return 'At Risk'
                    else: return 'Lost'
                
                rfm['Segment'] = rfm['RFM_Score'].apply(rfm_segment)
                
                safe_dataframe(rfm.head(10), use_container_width=True)
                
                # Segment distribution
                seg_counts = rfm['Segment'].value_counts()
                fig = px.pie(values=seg_counts.values, names=seg_counts.index,
                           title="Customer Segmentation")
                safe_plot(fig, use_container_width=True)
    
    elif analysis_type == "Cohort Analysis":
        st.subheader("Cohort Analysis")
        st.info("Create monthly/weekly cohorts and track retention/growth")
        
        if date_cols:
            date_col = st.selectbox("Select Date Column", date_cols)
            
            df[date_col] = pd.to_datetime(df[date_col])
            df['Cohort'] = df[date_col].dt.to_period('M')
            
            cohort_data = df.groupby('Cohort').size()
            
            fig = px.line(x=cohort_data.index.astype(str), y=cohort_data.values,
                        title="Cohort Sizes Over Time",
                        markers=True)
            safe_plot(fig, use_container_width=True)
    
    elif analysis_type == "Funnel Analysis":
        st.subheader("Conversion Funnel")
        
        funnel_stages = st.multiselect("Define Funnel Stages (in order)", 
                                      cat_cols if cat_cols else num_cols)
        
        if len(funnel_stages) >= 2 and st.button("Calculate Funnel"):
            stage_counts = []
            for stage in funnel_stages:
                count = df[stage].notna().sum()
                stage_counts.append(count)
            
            conversion_rates = [100] + [round(stage_counts[i+1]/stage_counts[i]*100, 1) 
                                       for i in range(len(stage_counts)-1)]
            
            fig = px.funnel(x=stage_counts, y=funnel_stages,
                          title="Conversion Funnel")
            safe_plot(fig, use_container_width=True)
            
            # Conversion metrics
            st.subheader("Conversion Rates")
            for i, stage in enumerate(funnel_stages):
                st.write(f"**{stage}**: {stage_counts[i]:,} | Conversion: {conversion_rates[i]:.1f}%")

# --- CLUSTER PHASE ---
def render_cluster(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🧠 Cluster Phase</div>
            <div class="phase-subtitle">K-Means Segmentation</div>
        </div>
    """, unsafe_allow_html=True)
    if df is None: return
    num_cols, _, _ = get_column_types(df)
    
    if len(num_cols) < 2: 
        st.warning("Need 2+ numeric columns.")
        return
    
    cols = st.multiselect("Features", num_cols, default=num_cols[:3])
    k = st.slider("Clusters", 2, 8, 3)
    
    if len(cols) >= 2:
        X = df[cols].dropna()
        if len(X) == 0:
            st.error("Selected columns contain no valid data.")
            return
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        X['Cluster'] = model.fit_predict(X_scaled).astype(str)
        
        pca = PCA(2)
        components = pca.fit_transform(X_scaled)
        X['PCA1'] = components[:,0]
        X['PCA2'] = components[:,1]
        
        fig = px.scatter(X, x='PCA1', y='PCA2', color='Cluster', title="Cluster Projection (PCA)")
        safe_plot(fig, use_container_width=True)
        
        st.subheader("Cluster Profiles (Mean Values)")
        safe_dataframe(X.groupby("Cluster")[cols].mean())

# --- TIME SERIES PHASE ---
def render_timeseries(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📈 Time Series Phase</div>
            <div class="phase-subtitle">Trend Analysis</div>
        </div>
    """, unsafe_allow_html=True)
    if df is None: return
    
    _, _, dates = get_column_types(df)
    nums, _, _ = get_column_types(df)
    
    if not dates: 
        st.warning("No date columns found.")
        return
    
    dc = st.selectbox("Date Column", dates)
    nc = st.selectbox("Value Column", nums)
    
    agg = df.groupby(dc)[nc].sum().reset_index().sort_values(dc)
    
    fig = px.line(agg, x=dc, y=nc, title=f"{nc} Trend over Time")
    fig.update_xaxes(rangeslider_visible=True)
    safe_plot(fig, use_container_width=True)

    # --- ADVANCED TIME SERIES: DECOMPOSITION & ACF/PACF ---
    if len(agg) > 20:
        st.subheader("Time Series Decomposition")
        try:
            # Decomposition only works with DateTime index and frequency
            ts_data = agg.set_index(dc)[nc]
            
            # Infer freq (optional, or let statsmodels try)
            if pd.infer_freq(ts_data.index):
                decomp = seasonal_decompose(ts_data, model='additive', period=12 if len(ts_data) > 24 else 4)
                
                fig = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                fig.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, mode='lines', name='Observed'), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, mode='lines', name='Trend'), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, mode='lines', name='Residual'), row=4, col=1)
                fig.update_layout(height=800, title="Additive Decomposition")
                safe_plot(fig, use_container_width=True)
            else:
                st.info("Could not automatically determine frequency for decomposition. Ensure dates are regular.")
                
        except Exception as e:
            st.warning(f"Decomposition unavailable: {e}")

    # --- FORECAST SIMULATION (Monte Carlo Lite) ---
    st.subheader("Forecast Simulation (Monte Carlo)")
    steps = st.slider("Forecast Steps", 10, 100, 30)
    sims = st.slider("Number of Simulations", 10, 200, 50)
    
    @st.cache_data(show_spinner=False)
    def run_monte_carlo(last_val, daily_vol, steps, sims):
        sim_df = pd.DataFrame()
        for i in range(sims):
            price_series = [last_val]
            for _ in range(steps):
                price = price_series[-1] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
            sim_df[f'Sim_{i}'] = price_series
        return sim_df

    if st.button("Run Simulation"):
        last_val = agg[nc].iloc[-1]
        returns = agg[nc].pct_change().dropna()
        daily_vol = returns.std()
        
        sim_df = run_monte_carlo(last_val, daily_vol, steps, sims)
            
        fig = px.line(sim_df, title=f"Monte Carlo Simulation ({steps} days ahead)")
        fig.update_traces(line=dict(width=1), opacity=0.5)
        safe_plot(fig, use_container_width=True)

# --- IMPACT PHASE ---
def render_impact(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🎯 Impact Phase</div>
            <div class="phase-subtitle">A/B Test Simulator</div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        m_a = st.number_input("Control Mean", 100.0)
        m_b = st.number_input("Variant Mean", 110.0)
        std = st.number_input("Std Dev", 15.0)
    with c2:
        n = st.slider("Sample Size", 100, 5000, 1000)
    
    x = np.linspace(min(m_a,m_b)-3*std, max(m_a,m_b)+3*std, 500)
    y_a = stats.norm.pdf(x, m_a, std)
    y_b = stats.norm.pdf(x, m_b, std)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_a, fill='tozeroy', name='Control', line_color='grey'))
    fig.add_trace(go.Scatter(x=x, y=y_b, fill='tozeroy', name='Variant', line_color='#00ADB5'))
    fig.update_layout(title="Distribution Overlap")
    safe_plot(fig, use_container_width=True)

# --- REPORT PHASE ---
def render_report(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📝 Report Phase</div>
            <div class="phase-subtitle">Automated Analysis & Document Generation</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return

    num_cols = df.select_dtypes(include=[np.number]).columns
    
    with st.spinner("Generating Insights..."):
        n_rows, n_cols = df.shape
        missing_pct = df.isna().mean().mean() * 100
        
        corr_insight = "No strong correlations found."
        if len(num_cols) > 1:
            corr_mat = df[num_cols].corr().abs()
            np.fill_diagonal(corr_mat.values, 0)
            max_corr = corr_mat.max().max()
            if max_corr > 0.7:
                c1, c2 = corr_mat.stack().idxmax()
                corr_insight = f"Strong relationship detected between **{c1}** and **{c2}** ({max_corr:.2f})."
        
        html_report = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Arial', sans-serif; line-height: 1.6; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #16a085; border-bottom: 2px solid #16a085; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .stat-box {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #00ADB5; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Lumina Analytics Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>1. Executive Summary</h2>
            <div class="stat-box">
                <p>The dataset contains <strong>{n_rows} rows</strong> and <strong>{n_cols} columns</strong>.</p>
                <p>Data Completeness: <strong>{100-missing_pct:.1f}%</strong> (Average missing data: {missing_pct:.1f}%)</p>
                <p>{corr_insight}</p>
            </div>

            <h2>2. Statistical Overview</h2>
            {df.describe().to_html(classes='table')}

            <h2>3. Recommendations</h2>
            <ul>
                <li><strong>Data Quality:</strong> { 'Consider imputation strategies for missing values.' if missing_pct > 0 else 'Data quality is robust.' }</li>
                <li><strong>Analysis:</strong> {'Investigate drivers behind correlations.' if len(num_cols) > 1 else 'Explore categorical breakdowns for hidden patterns.'}</li>
                <li><strong>Next Steps:</strong> Use the Impact Phase to simulate potential interventions.</li>
            </ul>
        </body>
        </html>
        """

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Report Preview")
        st.markdown(f"""
        **Executive Summary** The dataset contains **{n_rows} rows** and **{n_cols} columns**.  
        Data Completeness: **{100-missing_pct:.1f}%** | **Insight**: {corr_insight}
        """, unsafe_allow_html=True)
        
        with st.expander("View Full Statistical Table"):
            safe_dataframe(df.describe())

    with col2:
        st.subheader("💾 Export")
        st.info("Download this analysis as a Word-compatible document.")
        
        st.download_button(
            label="Download Report (.doc)",
            data=html_report,
            file_name=f"Lumina_Report_{datetime.now().strftime('%Y%m%d')}.doc",
            mime="application/msword"
        )
        st.caption("*Format: HTML-based Word Document*")


# ============================================================================
# 15. POST-GRADUATE STATISTICS (NEW)
# ============================================================================

def render_glm(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📈 Generalized Linear Models (GLM)</div>
            <div class="phase-subtitle">Poisson, Gamma & Negative Binomial Regression</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    num_cols, _, _ = get_column_types(df)
    
    target = st.selectbox("Response Variable (Y)", num_cols)
    features = st.multiselect("Predictors (X)", [c for c in num_cols if c != target])
    
    if not features: return
    
    family_name = st.selectbox("Distribution Family", ["Gaussian", "Poisson", "Gamma", "Binomial", "Negative Binomial"])
    link_name = st.selectbox("Link Function", ["Identity", "Log", "Logit", "Inverse_Power"])
    
    # Map Families
    fam_map = {
        "Gaussian": families.Gaussian(),
        "Poisson": families.Poisson(),
        "Gamma": families.Gamma(),
        "Binomial": families.Binomial(),
        "Negative Binomial": families.NegativeBinomial()
    }

    # Map Links - If user selects non-default, we apply it
    # Note: Statsmodels families have defaults, this overrides if needed.
    link_map = {
        "Identity": families.links.identity(),
        "Log": families.links.log(),
        "Logit": families.links.logit(),
        "Inverse_Power": families.links.inverse_power()
    }
    
    if st.button("Fit GLM"):
        try:
            # Use clean_xy to prevent NaNs
            X, y = clean_xy(df, target, features)
            
            if len(X) < 10:
                st.warning("Not enough data after cleaning.")
                return

            X = sm.add_constant(X)
            
            
            chosen_fam = fam_map[family_name]
            # Apply link if compatible (simple implementation)
            try:
               # We can pass link argument in init if we reconstruct, or just set link.
               # Easier: families.Gaussian(link=link_map[link_name])
               if family_name == "Gaussian": chosen_fam = families.Gaussian(link=link_map[link_name])
               elif family_name == "Poisson": chosen_fam = families.Poisson(link=link_map[link_name])
               elif family_name == "Gamma": chosen_fam = families.Gamma(link=link_map[link_name])
               elif family_name == "Binomial": chosen_fam = families.Binomial(link=link_map[link_name])
               elif family_name == "Negative Binomial": chosen_fam = families.NegativeBinomial(link=link_map[link_name])
            except:
               st.warning("Link function incompatible with Family. Using default.")

            model = sm.GLM(y, X, family=chosen_fam).fit()
            
            st.success(f"GLM Fitted! Deviance: {model.deviance:.2f}")
            st.text(model.summary().as_text())
            
        except Exception as e:
            st.error(f"GLM Error: {e}")

def render_multivariate(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🕸️ Multivariate Analysis</div>
            <div class="phase-subtitle">MANOVA & Factor Analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    st.subheader("MANOVA (Multivariate ANOVA)")
    
    dependents = st.multiselect("Dependent Variables (DVs)", num_cols)
    group = st.selectbox("Independent Grouping Variable (IV)", cat_cols)
    
    if len(dependents) > 1 and group:
        if st.button("Run MANOVA"):
            try:
                # Formula format: "dv1 + dv2 ~ group"
                dv_str = " + ".join(dependents)
                formula = f"{dv_str} ~ {group}"
                
                manova = MANOVA.from_formula(formula, data=df)
                st.write(manova.mv_test())
                
            except Exception as e:
                st.error(f"MANOVA Error: {e}")
                
def render_survival(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">⏳ Survival Analysis</div>
            <div class="phase-subtitle">Kaplan-Meier Estimator & Log-Rank Test</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    dur_col = st.selectbox("Duration Column (Time)", num_cols)
    event_col = st.selectbox("Event Column (1=Event, 0=Censor)", num_cols)
    group_col = st.selectbox("Group By (Optional)", [None] + cat_cols)
    
    if st.button("Run Survival Analysis"):
        # Validation
        if len(df[event_col].unique()) > 2 or not set(df[event_col].unique()).issubset({0, 1}):
            st.error("Event Column must be binary (0/1).")
            return

        if LIFELINES_AVAILABLE:
            kmf = KaplanMeierFitter()
            
            # Setup Plot
            
            if group_col:
                # Binary Check
                unique_events = df[event_col].dropna().unique()
                if not set(unique_events).issubset({0, 1}):
                    st.error("Event column must be binary (0/1).")
                    return

                groups = df[group_col].unique()
                for g in groups:
                    mask = df[group_col] == g
                    kmf.fit(df.loc[mask, dur_col], df.loc[mask, event_col], label=str(g))
                    
                    # Get KM data
                    survival_df = kmf.survival_function_
                    unique_times = dists[dur_col].unique()
                    
                    for t in unique_times:
                        at_risk = dists[dists[dur_col] >= t]
                        events = at_risk[at_risk[event_col] == 1]
                        d_i = len(events[events[dur_col] == t])
                        n_i = len(at_risk)
                        
                        if n_i > 0:
                            survival *= (1 - d_i/n_i)
                        km_data.append({'Time': t, 'Survival Probability': survival})
                    return pd.DataFrame(km_data)
                
                fig = go.Figure()
                if group_col:
                    groups = df[group_col].unique()
                    for g in groups:
                        sub_df = df[df[group_col] == g]
                        km = calculate_km(sub_df)
                        fig.add_trace(go.Scatter(x=km['Time'], y=km['Survival Probability'], mode='lines', step='hv', name=str(g)))
                else:
                    km = calculate_km(df)
                    fig.add_trace(go.Scatter(x=km['Time'], y=km['Survival Probability'], mode='lines', step='hv', name='All Data'))
                    
                fig.update_layout(title="Kaplan-Meier Survival Curve (Manual)", xaxis_title="Time", yaxis_title="Survival Probability")
                safe_plot(fig, use_container_width=True)


def render_power_analysis(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🔋 Power Analysis</div>
            <div class="phase-subtitle">Sample Size & Effect Size Calculation</div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    test_type = c1.selectbox("Test Type", ["T-Test Ind", "ANOVA"])
    alpha = c2.number_input("Significance Level (alpha)", 0.01, 0.10, 0.05)
    power = c2.slider("Desired Power", 0.7, 0.99, 0.8)
    effect_size = c1.number_input("Effect Size", 0.1, 2.0, 0.5)
    
    # Conditional Input
    k = 3
    if test_type == "ANOVA":
        k = c1.number_input("Number of Groups", 2, 10, 3)
    
    if st.button("Calculate Sample Size"):
        if test_type == "T-Test Ind":
            analysis = TTestIndPower()
            n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
            st.metric("Required N (per group)", f"{math.ceil(n)}")
        elif test_type == "ANOVA":
            analysis = FTestAnovaPower()
            # k is already defined above
            n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k)
            st.metric("Total N", f"{math.ceil(n)}")


# ============================================================================
# 14. ADVANCED ANALYTICS: PARETO & MARKET BASKET (NEW)
# ============================================================================

def render_pareto_analysis(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">📉 Pareto Analysis</div>
            <div class="phase-subtitle">The 80/20 Rule & ABC Classification</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    cat = st.selectbox("Category (Items)", cat_cols)
    metric = st.selectbox("Metric (Value)", num_cols)
    
    if cat and metric:
        # Aggregation
        pareto_df = df.groupby(cat)[metric].sum().reset_index().sort_values(metric, ascending=False)
        pareto_df['Cumulative Percentage'] = pareto_df[metric].cumsum() / pareto_df[metric].sum() * 100
        
        # ABC Class
        def abc_classify(p):
            if p <= 80: return 'A'
            elif p <= 95: return 'B'
            else: return 'C'
            
        pareto_df['Class'] = pareto_df['Cumulative Percentage'].apply(abc_classify)
        
        # Dual Axis Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=pareto_df[cat], y=pareto_df[metric], name=metric, marker_color='#2c3e50'), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df[cat], y=pareto_df['Cumulative Percentage'], name='Cumulative %', mode='lines', line=dict(color='#e74c3c')), secondary_y=True)
        
        fig.update_layout(title=f"Pareto Analysis of {metric} by {cat}")
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110])
        safe_plot(fig, use_container_width=True)
        
        st.subheader("ABC Classification Summary")
        st.table(pareto_df['Class'].value_counts())


def render_market_basket(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🛒 Market Basket Analysis</div>
            <div class="phase-subtitle">Association Rules & Cross-Selling</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    tid = st.selectbox("Transaction ID", cat_cols + df.select_dtypes(include=[np.number]).columns.tolist())
    item = st.selectbox("Item ID", cat_cols)
    
    min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
    
    # Large Dataset Handling
    if len(df) > 50000:
        st.warning("⚠️ Large dataset detected (>50k rows). Sampling 50k for performance.")
        df = df.sample(50000, random_state=42)
    
    if st.button("Run Apriori Analysis"):
        with st.spinner("Mining rules..."):
            # Prepare transactions
            transactions = df.groupby(tid)[item].apply(list).tolist()
            
            # 1. Frequent Itemsets (Simplified Apriori)
            item_counts = {}
            for t in transactions:
                for i in t:
                    item_counts[i] = item_counts.get(i, 0) + 1
            
            n = len(transactions)
            frequent_items = {k: v/n for k, v in item_counts.items() if v/n >= min_support}
            
            # 2. Pairs
            pair_counts = {}
            for t in transactions:
                unique_t = list(set(t))
                for i in range(len(unique_t)):
                    for j in range(i+1, len(unique_t)):
                        item_a, item_b = sorted([unique_t[i], unique_t[j]])
                        if item_a in frequent_items and item_b in frequent_items:
                            pair = (item_a, item_b)
                            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            # 3. Rules
            rules = []
            for pair, count in pair_counts.items():
                support_pair = count / n
                if support_pair >= min_support:
                    item_a, item_b = pair
                    
                    # Rule A -> B
                    conf_a_b = support_pair / frequent_items[item_a]
                    lift_a_b = conf_a_b / frequent_items[item_b]
                    rules.append({'Rule': f"{item_a} -> {item_b}", 'Support': support_pair, 'Confidence': conf_a_b, 'Lift': lift_a_b})
                    
                    # Rule B -> A
                    conf_b_a = support_pair / frequent_items[item_b]
                    lift_b_a = conf_b_a / frequent_items[item_a]
                    rules.append({'Rule': f"{item_b} -> {item_a}", 'Support': support_pair, 'Confidence': conf_b_a, 'Lift': lift_b_a})
            
            if rules:
                rules_df = pd.DataFrame(rules).sort_values("Lift", ascending=False)
                st.success(f"Found {len(rules_df)} rules!")
                safe_dataframe(rules_df, use_container_width=True)
                
                # Viz
                if len(rules_df) > 0:
                    fig = px.scatter(rules_df, x="Support", y="Confidence", size="Lift", color="Lift",
                                   hover_name="Rule", title="Association Rules Matrix")
                    safe_plot(fig, use_container_width=True)
            else:
                st.warning("No rules found with current parameters. Try lowering support.")

def render_smart_narrative(df):
    st.markdown("""
        <div class="phase-container">
            <div class="phase-title">🧠 Smart Narrative Engine</div>
            <div class="phase-subtitle">Automated Data Storytelling</div>
        </div>
    """, unsafe_allow_html=True)
    
    if df is None: return
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # 1. Outlier Insight
    for col in num_cols[:3]: # Limit to first 3 for speed
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
        if outliers > 0:
            insights.append(f"• **{col}**: Contains **{outliers}** outliers ({outliers/len(df)*100:.1f}% of data). Possible anomalies.")
    
    # 2. Correlation Insight
    if len(num_cols) > 1:
        corr_mat = df[num_cols].corr()
        # Find max correlation < 1
        np.fill_diagonal(corr_mat.values, 0)
        max_c = corr_mat.max().max()
        if max_c > 0.7:
            r, c = corr_mat.stack().idxmax()
            insights.append(f"• **Strong Association**: **{r}** and **{c}** move together (Correlation: {max_c:.2f}).")
    
    # 3. skewness
    for col in num_cols[:3]:
        skew = df[col].skew()
        if abs(skew) > 1:
            desc = "Heavily right-skewed (high values tail)" if skew > 1 else "Heavily left-skewed (low values tail)"
            insights.append(f"• **{col}**: {desc}. Consider Log transformation.")
            
    for i in insights:
        st.write(i)
        
    st.divider()
    st.caption("AI-Generated Insights based on Statistical Properties")

# ============================================================================
# 4. MAIN ROUTING
# ============================================================================
# ============================================================================
# ADD TO MAIN SIDEBAR NAVIGATION
# ============================================================================

# Update the sidebar phase selection to include new phases:

PHASE_ICONS = {
                'Monitor':'🔍 Monitor',
                'Explore':'📊 Explore',
                'Cluster':'🧠 Cluster',
                'Time Series':'📈 Time Series',
                'Statistical Test':'📉 Statistical',
                'Regression':'🔧 Regression',
                'Impact':'🎯 Impact',
                'Report':'📝 Report',
                'Data Quality':'🏥 Data Quality',
                'Anomaly Detection':'🔎 Anomalies',
                'Feature Eng':'⚙️ Features',
                'Correlation':'📊 Correlations',
                'Predictive Model':'🤖 Models',
                'Business Analytics':'💼 Business',
                'Pareto Analysis':'📉 Pareto (80/20)',
                'Market Basket':'🛒 Market Basket',
                'Smart Narrative':'🧠 Smart Insights',
                'GLM':'📈 GLM',
                'Multivariate':'🕸️ Multivariate',
                'Survival':'⏳ Survival',
                'Power Analysis':'🔋 Power'
            }

def sidebar_processor():
    """Updated sidebar with all new phases"""
    with st.sidebar:
        st.markdown("## 🔮 Lumina Analytics Suite")
        
        phase = st.radio(
            "Workflow",
            list(PHASE_ICONS.keys()),
            format_func=lambda x: PHASE_ICONS[x]
        )
        
        st.divider()
        st.markdown("### 📂 Data Ingestion")
        
        uploaded_files = st.file_uploader(
            "Upload File(s)", 
            type=['csv', 'xlsx', 'xls', 'txt', 'tsv', 'json', 'parquet'],
            accept_multiple_files=True,
            help="Upload multiple files to merge them."
        )
        
        df = None
        
        if uploaded_files:
            # Multi-file Logic
            raw_dfs = {}
            for f in uploaded_files:
                d, s = parse_uploaded_file(f)
                if s == 'single': raw_dfs[f.name] = d
                elif s == 'multi': 
                    for k, v in d.items(): raw_dfs[f"{f.name} - {k}"] = v
            
            if len(raw_dfs) > 0:
                # Merge or Select
                if len(raw_dfs) > 1:
                    df = merge_datasets(raw_dfs)
                else:
                    df = list(raw_dfs.values())[0]
            
            if df is not None:
                # smart_date_converter
                df = smart_date_converter(df)
                st.session_state.data = df
                st.success(f"✅ Active Data: {len(df):,} rows")

                with st.expander("🛠️ Data Preparation", expanded=False):
                    st.caption(f"Processing Data")
                    
                    drop_cols = st.multiselect("Drop Columns", df.columns)
                    if drop_cols: 
                        df = df.drop(columns=drop_cols)
                    
                    if st.checkbox("Handle Missing Values"):
                        method = st.radio("Method", ["Drop Rows", "Fill 0", "Fill Mean/Mode"])
                        if method == "Drop Rows":
                            df = df.dropna()
                        elif method == "Fill 0":
                            df = df.fillna(0)
                        elif method == "Fill Mean/Mode":
                            num = df.select_dtypes(include=np.number).columns
                            cat = df.select_dtypes(exclude=np.number).columns
                            df[num] = df[num].fillna(df[num].mean())
                            for c in cat:
                                df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
                                
                    if st.checkbox("Remove Duplicates"):
                        df = df.drop_duplicates()

                st.divider()
                st.markdown("### 📥 Export Output")
                if st.button("Prepare Excel Download"):
                    try:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                            df.describe().to_excel(writer, sheet_name='Summary_Statistics')
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            if len(num_cols) > 1:
                                df[num_cols].corr().to_excel(writer, sheet_name='Correlations')
                        
                        st.download_button(
                            label="Download Excel Workbook",
                            data=buffer.getvalue(),
                            file_name=f"Lumina_Output_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    except Exception as e:
                        st.error(f"Export Error: {e}")
                
        st.divider()
        st.caption("v1.0")
        return phase, df

# ============================================================================
# UPDATE MAIN() FUNCTION TO ROUTE NEW PHASES
# ============================================================================

def main():
    phase, df = sidebar_processor()
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    def safe_render(render_func, df):
        try:
            render_func(df)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.checkbox("Show technical details"):
                st.exception(e)

    if phase == "Monitor": safe_render(render_monitor, df)
    elif phase == "Explore": safe_render(render_explore, df)
    elif phase == "Cluster": safe_render(render_cluster, df)
    elif phase == "Time Series": safe_render(render_timeseries, df)
    elif phase == "Statistical Test": safe_render(render_statistical_test, df)
    elif phase == "Regression": safe_render(render_regression, df)
    elif phase == "Impact": safe_render(render_impact, df)
    elif phase == "Report": safe_render(render_report, df)
    elif phase == "Data Quality": safe_render(render_data_quality, df)
    elif phase == "Anomaly Detection": safe_render(render_anomaly_detection, df)
    elif phase == "Feature Eng": safe_render(render_feature_engineering, df)
    elif phase == "Correlation": safe_render(render_correlation_analysis, df)
    elif phase == "Predictive Model": safe_render(render_predictive_modeling, df)
    elif phase == "Business Analytics": safe_render(render_business_analytics, df)
    elif phase == "Pareto Analysis": safe_render(render_pareto_analysis, df)
    elif phase == "Market Basket": safe_render(render_market_basket, df)
    elif phase == "Smart Narrative": safe_render(render_smart_narrative, df)
    elif phase == "GLM": safe_render(render_glm, df)
    elif phase == "Multivariate": safe_render(render_multivariate, df)
    elif phase == "Survival": safe_render(render_survival, df)
    elif phase == "Power Analysis": safe_render(render_power_analysis, df)

if __name__ == "__main__":
    main()