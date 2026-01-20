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
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
import io
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics Imports (Professional Phases)
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
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


# Advanced NLP Imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    TOPIC_MODELING_AVAILABLE = True
except ImportError:
    TOPIC_MODELING_AVAILABLE = False

# Neural Network Imports
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

# PDF Generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Network Analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# AutoML
try:
    from sklearn.model_selection import RandomizedSearchCV
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False

# SHAP & XGBoost Availability Check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import AnovaRM
    STATSMODELS_ADV_AVAILABLE = True
except ImportError:
    STATSMODELS_ADV_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False


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
def find_valid_header_row(sample_df):
    """
    Heuristic to find the most likely header row in a dataframe sample.
    Returns 0-indexed row number to use as header.
    """
    best_row = 0
    max_score = -1
    
    # Check first 20 rows or length of sample
    limit = min(20, len(sample_df))
    
    for i in range(limit):
        row = sample_df.iloc[i]
        
        # Criteria for a good header:
        # 1. Row is mostly complete (few NaNs) - Weight 1.0
        # 2. Values are Strings (Column names are text) - Weight 2.0
        # 3. Values are Unique (Column names shouldn't repeat) - Weight 0.5
        
        non_null_count = row.count()
        str_count = sum(isinstance(x, str) for x in row.dropna())
        unique_count = row.nunique()
        total_cols = len(row)
        
        # Normalized scores
        completeness = non_null_count / total_cols if total_cols > 0 else 0
        string_density = str_count / total_cols if total_cols > 0 else 0
        
        score = (completeness * 1) + (string_density * 2) + (unique_count * 0.1)
        
        # Penalty: If row looks purely numeric, it's likely data, not header
        try:
            pd.to_numeric(row.dropna())
            is_numeric = True
            score -= 2 # Heavy penalty for numeric rows
        except:
            is_numeric = False
            
        if score > max_score:
            max_score = score
            best_row = i
            
    return best_row

@st.cache_data(show_spinner=False)
def parse_uploaded_file(uploaded_file):
    """Universal parser with robust header detection."""
    try:
        filename = uploaded_file.name.lower()
        
        # --- EXCEL (.XLSX, .XLS) ---
        if filename.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            parsed_sheets = {}
            
            for sheet in sheet_names:
                # 1. Read sample to detect header
                sample = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20)
                if not sample.empty:
                    header_idx = find_valid_header_row(sample)
                    # 2. Read full sheet with correct header
                    df = pd.read_excel(xls, sheet_name=sheet, header=header_idx)
                    parsed_sheets[sheet] = df
            
            if len(parsed_sheets) == 1:
                return list(parsed_sheets.values())[0], "single"
            elif len(parsed_sheets) > 1:
                return parsed_sheets, "multi"
            else:
                 return None, "error"

        # --- CSV / TEXT ---
        elif filename.endswith(('.csv', '.txt', '.tsv')):
            sep = '\t' if filename.endswith(('.tsv', '.txt')) else ','
            
            # 1. Read sample
            uploaded_file.seek(0)
            sample = pd.read_csv(uploaded_file, sep=sep, header=None, nrows=20)
            
            # 2. Detect Header
            header_idx = find_valid_header_row(sample)
            
            # 3. Read Full
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=sep, header=header_idx), "single"

        # --- OTHER FORMATS ---
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

import uuid

def sanitize_col_name(name):
    """Sanitize column names for Statsmodels formulas."""
    if ' ' in name or not name.isidentifier():
        return f"Q('{name}')"
    return name

def safe_dataframe(data, **kwargs):
    if data is None: return
    
    df_disp = data.copy()
    for col in df_disp.select_dtypes(include=['object']):
        try:
            df_disp[col] = df_disp[col].astype(str)
        except: pass
    
    # Remove deprecated/invalid params
    kwargs.pop('use_container_width', None)
    kwargs.pop('width', None)
    
    # NEW API: use width='stretch' instead of use_container_width=True
    # NEW API: use width='stretch' instead of use_container_width=True
    # Streamlit > 1.40 prefers width='stretch'
    try:
        # Optimistically try new API
        st.dataframe(df_disp, width="stretch", **kwargs)
    except TypeError:
        # Fallback for older Streamlit
        st.dataframe(df_disp, use_container_width=True, **kwargs)

def safe_plot(fig, height=None, **kwargs):
    if fig is None: return
    # Remove potentially conflicting args if present
    kwargs.pop('use_container_width', None)
    
    if height:
        fig.update_layout(height=height)
    
    # NEW API: use width='stretch' instead of use_container_width=True
    # NEW API: use width='stretch' instead of use_container_width=True
    try:
        # Optimistically try new API
        # Remove legacy arg from kwargs if present
        kwargs.pop('use_container_width', None)
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        # Fallback for older Streamlit (which won't understand width='stretch' string or arg)
        st.plotly_chart(fig, use_container_width=True, **kwargs)

def get_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Enhanced datetime detection
    datetime_cols = df.select_dtypes(include=[
        'datetime64[ns]', 
        'datetime64[ns, UTC]', 
        'datetime64',
        'datetimetz'
    ]).columns.tolist()
    return numeric, categorical, datetime_cols

def clean_xy(df, target, features):
    """Helper to ensure X and y are perfectly aligned and clean."""
    data = df[[target] + features].dropna()
    X = data[features]
    y = data[target]
    return X, y

def check_dataset_size(df, limit=50000, phase_name="this analysis"):
    """Global performance guard for large datasets."""
    if len(df) > limit:
        st.warning(f"⚠️ Large dataset detected ({len(df):,} rows) for {phase_name}.")
        # Use unique key to prevent collisions
        unique_key = f"sample_{phase_name}_{uuid.uuid4().hex[:8]}"
        if st.checkbox("Sample data for performance?", value=True, key=unique_key):
            return df.sample(limit, random_state=42)
    return df

import traceback
def safe_render(render_func, df):
    """Wrap phase calls to prevent crashes."""
    try:
        render_func(df)
    except Exception as e:
        st.error(f"An error occurred in this phase: {str(e)}")
        with st.expander("🔧 Technical Details"):
            st.code(traceback.format_exc())

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

    if not all_cols:
        st.warning("Dataset has no columns to visualize.")
        return

    with st.container():
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        
        with c1:
            x_col = st.selectbox("X Axis (Required)", all_cols, index=0)
        
        with c2:
            def_y_idx = 0
            # Default to first numeric column if possible
            if len(num_cols) > 0 and x_col not in num_cols:
                try:
                    def_y_idx = all_cols.index(num_cols[0])
                except ValueError:
                    def_y_idx = 0
            
            # Safe index calculation
            options_y = ["None"] + all_cols
            target_idx = def_y_idx + 1
            if target_idx >= len(options_y):
                target_idx = 0
                
            y_col = st.selectbox("Y Axis (Optional)", options_y, index=target_idx)
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
        distance_matrix = 1 - np.abs(corr_matrix)
        distance_matrix = distance_matrix.fillna(0)
        
        # Use create_dendrogram directly - it handles everything
        fig = ff.create_dendrogram(
            distance_matrix.values,
            labels=distance_matrix.columns.tolist(),
            orientation='bottom'
        )
        fig.update_layout(
            title="Correlation Dendrogram",
            xaxis_title="Features",
            yaxis_title="Distance",
            height=500
        )
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
        if not cat_cols:
            st.error("This test requires at least one categorical column for grouping.")
            return

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
        if not cat_cols:
            st.error("This test requires at least one categorical column for grouping.")
            return

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
        if not cat_cols:
            st.error("This test requires at least one categorical column for grouping.")
            return

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
        if not cat_cols:
            st.error("This test requires at least one categorical column for grouping.")
            return

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
        if not cat_cols:
            st.error("This test requires at least one categorical column for grouping.")
            return

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
        formula = f"{sanitize_col_name(target)} ~ {' + '.join([sanitize_col_name(f) for f in features])}"
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
    
    # Global performance guard
    df = check_dataset_size(df, limit=50000, phase_name="Predictive Modeling")
    
    # --- MODEL SELECTION ---
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
    
    if problem_type == "Classification":
        target = st.selectbox("Select Target Variable", cat_cols if cat_cols else num_cols)
        
        if target in df.columns and df[target].nunique() <= 10:
            features = st.multiselect("Select Features", 
                                    [c for c in num_cols if c != target],
                                    default=num_cols[:min(3, len(num_cols))])
            
            if features:
                # Clean Data
                X, y = clean_xy(df, target, features)
                
                if len(X) < 20: 
                    st.error("Not enough data for split (min 20 samples).")
                    return

                # Encode Target if Categorical
                if problem_type == "Classification": # Assuming problem_type is the task_type
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
                
                rfm = df.groupby(id_col).agg(
                    Recency=(date_col, lambda x: (reference_date - pd.to_datetime(x).max()).days),
                    Frequency=(date_col, 'count'),
                    Monetary=(amount_col, 'sum')
                ).reset_index()
                
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
        
        # Parquet export
        st.divider()
        st.caption("**Preserve Data Types**")
        
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        st.download_button(
            label="Download Cleaned Data (.parquet)",
            data=buffer.getvalue(),
            file_name=f"Lumina_Cleaned_{datetime.now().strftime('%Y%m%d')}.parquet",
            mime="application/octet-stream"
        )
        st.caption("*Parquet preserves date/categorical types that Excel loses.*")


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
    
    # Dynamic link options based on family
    valid_links = {
        "Gaussian": ["Identity", "Log"],
        "Poisson": ["Log", "Identity"],
        "Gamma": ["Log", "Inverse_Power", "Identity"],
        "Binomial": ["Logit", "Log"],
        "Negative Binomial": ["Log"]
    }
    
    link_name = st.selectbox("Link Function", valid_links.get(family_name, ["Identity"]))
    
    # Map Families
    fam_map = {
        "Gaussian": families.Gaussian(),
        "Poisson": families.Poisson(),
        "Gamma": families.Gamma(),
        "Binomial": families.Binomial(),
        "Negative Binomial": families.NegativeBinomial()
    }

    # Map Links
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
            
            # Apply Link Function
            chosen_fam = fam_map[family_name]
            try:
                # Re-instantiate family with link
                if family_name == "Gaussian": chosen_fam = families.Gaussian(link=link_map[link_name])
                elif family_name == "Poisson": chosen_fam = families.Poisson(link=link_map[link_name])
                elif family_name == "Gamma": chosen_fam = families.Gamma(link=link_map[link_name])
                elif family_name == "Binomial": chosen_fam = families.Binomial(link=link_map[link_name])
                elif family_name == "Negative Binomial": chosen_fam = families.NegativeBinomial(link=link_map[link_name])
            except:
                st.warning("Link function incompatible. Using default.")

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
                formula = f"{' + '.join([sanitize_col_name(c) for c in dependents])} ~ {sanitize_col_name(group)}"
                
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
        unique_events = df[event_col].dropna().unique()
        if not set(unique_events).issubset({0, 1, 0.0, 1.0}):
            st.error("Event Column must be binary (0/1).")
            return
        
        if LIFELINES_AVAILABLE:
            fig = go.Figure()
            
            if group_col:
                groups = df[group_col].dropna().unique()
                kmf_results = {}
                
                for g in groups:
                    mask = df[group_col] == g
                    sub_df = df.loc[mask].dropna(subset=[dur_col, event_col])
                    
                    kmf = KaplanMeierFitter()
                    kmf.fit(sub_df[dur_col], sub_df[event_col], label=str(g))
                    kmf_results[g] = (sub_df[dur_col], sub_df[event_col])
                    
                    # Plot survival function
                    survival_df = kmf.survival_function_
                    fig.add_trace(go.Scatter(
                        x=survival_df.index,
                        y=survival_df[str(g)],
                        mode='lines',
                        name=str(g),
                        line_shape='hv'
                    ))
                
                # Log-Rank Test
                if len(groups) == 2:
                    g1, g2 = list(groups)[:2]
                    try:
                        result = logrank_test(
                            kmf_results[g1][0], kmf_results[g2][0],
                            kmf_results[g1][1], kmf_results[g2][1]
                        )
                        
                        st.subheader("Log-Rank Test")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Test Statistic", f"{result.test_statistic:.4f}")
                        c2.metric("P-Value", f"{result.p_value:.4f}")
                        c3.metric("Result", "✓ Significant" if result.p_value < 0.05 else "✗ Not Sig.")
                    except Exception as e:
                        st.warning(f"Could not run Log-Rank test: {e}")

            else:
                sub_df = df.dropna(subset=[dur_col, event_col])
                kmf = KaplanMeierFitter()
                kmf.fit(sub_df[dur_col], sub_df[event_col], label='All Data')
                
                survival_df = kmf.survival_function_
                fig.add_trace(go.Scatter(
                    x=survival_df.index,
                    y=survival_df['All Data'],
                    mode='lines',
                    name='All Data',
                    line_shape='hv'
                ))
            
            fig.update_layout(
                title="Kaplan-Meier Survival Curve",
                xaxis_title="Time",
                yaxis_title="Survival Probability",
                yaxis_range=[0, 1.05]
            )
            safe_plot(fig, use_container_width=True)
        else:
            st.warning("Install `lifelines` for full survival analysis: `pip install lifelines`")


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
# 5. ADVANCED ANALYTICS (NEW MODULES)
# ============================================================================

# --- RETAINED SPECIALIZED MODULES ---

def render_3d_scatter(df):
    """Interactive 3D Scatter Plot visualization."""
    with stat_box("3D Visualization Config"):
        st.markdown("### 🌐 3D Scatter Plot")
        
        if df is None: return
        
        num_cols, cat_cols, _ = get_column_types(df)
        
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns for 3D visualization.")
            return
        
        col1, col2, col3 = st.columns(3)
        x_col = col1.selectbox("X Axis", num_cols, index=0)
        y_col = col2.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))
        z_col = col3.selectbox("Z Axis", num_cols, index=min(2, len(num_cols)-1))
        
        col4, col5 = st.columns(2)
        color_col = col4.selectbox("Color By", ["None"] + cat_cols + num_cols)
        size_col = col5.selectbox("Size By", ["None"] + num_cols)
        
        if color_col == "None": color_col = None
        if size_col == "None": size_col = None

    # Create 3D scatter plot
    plot_df = df.copy()
    if len(plot_df) > 5000:
        st.info(f"Sampling 5000 points from {len(plot_df)} for performance.")
        plot_df = plot_df.sample(5000, random_state=42)
    
    fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col,
                        color=color_col, size=size_col,
                        opacity=0.7, title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}")
    
    safe_plot(fig, height=700)

def render_sankey_diagram(df):
    """Interactive Sankey Diagram for flow visualization."""
    st.markdown("### 🔀 Sankey Diagram")
    if df is None: return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    if len(cat_cols) < 2:
        st.warning("Need at least 2 categorical columns.")
        return
        
    col1, col2, col3 = st.columns(3)
    source = col1.selectbox("Source", cat_cols, index=0)
    target = col2.selectbox("Target", cat_cols, index=min(1, len(cat_cols)-1))
    value = col3.selectbox("Value (Optional)", ["Count"] + num_cols)
    
    if st.button("Generate Sankey"):
        if value == "Count":
            flow = df.groupby([source, target]).size().reset_index(name='value')
        else:
            flow = df.groupby([source, target])[value].sum().reset_index(name='value')
            
        # Create labels
        all_nodes = list(pd.concat([flow[source], flow[target]]).unique())
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        link_source = [node_map[x] for x in flow[source]]
        link_target = [node_map[x] for x in flow[target]]
        link_value = flow['value']
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=all_nodes, pad=15, thickness=20, line=dict(color="black", width=0.5)),
            link=dict(source=link_source, target=link_target, value=link_value)
        )])
        safe_plot(fig)

def render_network_graph(df):
    """Network Graph using NetworkX."""
    st.markdown("### 🕸️ Network Graph")
    if df is None: return
    if not NETWORKX_AVAILABLE:
        st.error("NetworkX not installed.")
        return
        
    num_cols, _, _ = get_column_types(df)
    if len(num_cols) < 2: return
    
    threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5)
    
    if st.button("Generate Correlation Network"):
        corr = df[num_cols].corr()
        G = nx.Graph()
        
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j and abs(corr.loc[c1, c2]) > threshold:
                    G.add_edge(c1, c2, weight=abs(corr.loc[c1, c2]))
        
        if len(G.nodes()) == 0:
            st.warning("No connections found at this threshold.")
            return

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=list(G.nodes()), textposition="top center",
            marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[len(list(G.neighbors(n))) for n in G.nodes()])
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))
        safe_plot(fig)

def render_pdf_report(df):
    """Generate PDF Report."""
    st.markdown("### 📄 PDF Report Generator")
    if df is None: return
    if not REPORTLAB_AVAILABLE: st.error("ReportLab not installed."); return
    
    title = st.text_input("Report Title", "Lumina Analytics Report")
    
    if st.button("Generate PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph(title, styles['Title']), Spacer(1, 12)]
        
        # Summary
        summary = f"Dataset contains {len(df)} rows and {len(df.columns)} columns."
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Stats
        desc = df.describe().round(2).reset_index().values.tolist()
        cols = ['Stat'] + df.describe().columns.tolist()
        data = [cols] + desc
        # Limit columns to fit page
        if len(data[0]) > 6: data = [r[:6] for r in data]
        
        t = Table(data)
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(t)
        
        doc.build(story)
        st.download_button("Download PDF", buffer.getvalue(), "report.pdf", "application/pdf")

# --- NEW ANALYTICS SUITES (2.0) ---

def render_explainability(df):
    st.header("🔍 Explainability & Interpretability")
    
    if df is None: return
    num_cols, cat_cols, _ = get_column_types(df)
    
    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Features", [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:5])
    
    if not features:
        st.warning("Select features to proceed.")
        return

    # Prepare Data
    X, y = clean_xy(df, target, features)
    is_classification = False
    if df[target].dtype == 'object' or df[target].nunique() < 10:
        is_classification = True
        le = LabelEncoder()
        y = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    model.fit(X, y)
    
    tab1, tab2, tab3, tab4 = st.tabs(["SHAP", "LIME", "Partial Dependence", "Interactions"])
    
    # --- SHAP ---
    with tab1:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            # Limit to 100 samples for performance
            X_sample = X.sample(min(100, len(X)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            if is_classification and isinstance(shap_values, list):
                shap_values = shap_values[1] # Positive class
            
            st.write("**Summary Plot**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)
            
            st.write("**Waterfall Plot (First Sample)**")
            fig2, ax2 = plt.subplots()
            
            # Safe Waterfall Plot
            base_val = explainer.expected_value
            if is_classification and isinstance(base_val, list):
                base_val = base_val[1]
                
            shap_exp = shap.Explanation(values=shap_values[0], 
                                        base_values=base_val, 
                                        data=X_sample.iloc[0].values, 
                                        feature_names=features)
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig2)
            plt.close(fig2)
            plt.close(fig)
        else:
            st.error("SHAP library not installed.")

    # --- LIME ---
    with tab2:
        st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
        if LIME_AVAILABLE:
            idx = st.slider("Select Instance Index", 0, len(X)-1, 0)
            
            mode = 'classification' if is_classification else 'regression'
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X),
                feature_names=features,
                class_names=[str(c) for c in np.unique(y)] if is_classification else None,
                mode=mode
            )
            
            if is_classification:
                exp = explainer.explain_instance(X.iloc[idx], model.predict_proba, num_features=10)
            else:
                exp = explainer.explain_instance(X.iloc[idx], model.predict, num_features=10)
            
            st.components.v1.html(exp.as_html(), height=500, scrolling=True)
        else:
            st.error("LIME library not installed.")

    # --- PDP ---
    with tab3:
        st.subheader("Partial Dependence Plots")
        feature_to_plot = st.selectbox("Select Feature for PDP", features)
        
        common_params = {
            "grid_resolution": 20,
        }
        
        display = partial_dependence(model, X, [feature_to_plot], kind="average", **common_params)
        
        fig = px.line(x=display['grid_values'][0], y=display['average'][0], 
                      title=f"Partial Dependence: {feature_to_plot}",
                      labels={'x': feature_to_plot, 'y': 'Partial Dependence'})
        safe_plot(fig)

    # --- Interactions ---
    with tab4:
        st.subheader("Feature Interaction Heatmap (Correlation of Features)")
        # Simple proxy for interaction: Correlation matrix of features
        corr = X.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
        safe_plot(fig)
        st.caption("Note: High correlation between features often implies interaction effects in linear models, though tree models handle them natively.")

def render_advanced_timeseries(df):
    st.header("📈 Advanced Time Series")
    
    if df is None: return
    _, _, date_cols = get_column_types(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not date_cols:
        st.error("No date columns found.")
        return

    date_col = st.selectbox("Date Column", date_cols)
    val_col = st.selectbox("Value Column", num_cols)
    
    # Prepare TS Data
    ts_df = df[[date_col, val_col]].dropna().sort_values(by=date_col)
    ts_df = ts_df.set_index(date_col)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prophet", "ARIMA", "Anomaly Detection", "VAR", "Changepoint"])
    
    # --- Prophet ---
    with tab1:
        st.subheader("Facebook Prophet")
        if PROPHET_AVAILABLE:
            periods = st.number_input("Forecast Periods", 30, 365, 30)
            
            prophet_df = ts_df.reset_index().rename(columns={date_col: 'ds', val_col: 'y'})
            
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False))
            safe_plot(fig)
            
            st.write("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.error("Prophet not installed.")

    # --- ARIMA ---
    with tab2:
        st.subheader("ARIMA / SARIMA")
        if STATSMODELS_ADV_AVAILABLE:
            p = st.number_input("p (AR)", 0, 5, 1)
            d = st.number_input("d (I)", 0, 2, 1)
            q = st.number_input("q (MA)", 0, 5, 1)
            
            if st.button("Fit ARIMA"):
                model = ARIMA(ts_df, order=(p,d,q))
                model_fit = model.fit()
                st.text(model_fit.summary())
                
                # Forecast
                forecast_steps = 30
                forecast = model_fit.forecast(steps=forecast_steps)
                
                fig = px.line(y=ts_df[val_col], title="Historical Data")
                fig.add_scatter(x=pd.date_range(start=ts_df.index[-1], periods=forecast_steps+1, freq='D')[1:], y=forecast, name='Forecast')
                safe_plot(fig)
        else:
            st.error("Statsmodels not installed.")

    # --- Anomaly Detection ---
    with tab3:
        st.subheader("Anomaly Detection (STL Residuals)")
        try:
            stl = seasonal_decompose(ts_df[val_col], model='additive', period=12)
            resid = stl.resid.dropna()
            threshold = resid.std() * 3
            anomalies = resid[abs(resid) > threshold]
            
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[val_col], name="Original"), row=1, col=1)
            fig.add_trace(go.Scatter(x=anomalies.index, y=ts_df.loc[anomalies.index][val_col], mode='markers', marker=dict(color='red', size=10), name="Anomaly"), row=1, col=1)
            fig.add_trace(go.Scatter(x=resid.index, y=resid, name="Residuals"), row=2, col=1)
            safe_plot(fig)
            
            st.write(f"Detected {len(anomalies)} anomalies.")
        except Exception as e:
            st.error(f"STL Decomposition failed: {e}. Ensure data has frequency.")

    # --- VAR ---
    with tab4:
        st.subheader("Multi-Variate Time Series (VAR)")
        vars_select = st.multiselect("Select Variables for VAR", num_cols, default=num_cols[:2])
        if len(vars_select) > 1:
            var_data = df[vars_select].dropna()
            model = VAR(var_data)
            results = model.fit(2)
            st.text(results.summary())
        else:
            st.warning("Select at least 2 variables.")

    # --- Changepoint ---
    with tab5:
        st.subheader("Changepoint Detection (Rolling Statistics)")
        window = st.slider("Rolling Window", 5, 50, 14)
        rolling_mean = ts_df[val_col].rolling(window=window).mean()
        rolling_std = ts_df[val_col].rolling(window=window).std()
        
        # Simple logic: if value deviates significantly from rolling mean
        deviations = abs(ts_df[val_col] - rolling_mean) > (3 * rolling_std)
        changepoints = ts_df[deviations]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[val_col], name="Data"))
        fig.add_trace(go.Scatter(x=changepoints.index, y=changepoints[val_col], mode='markers', name="Potential Structural Break", marker=dict(color='orange')))
        safe_plot(fig)

def render_advanced_stats(df):
    st.header("🧪 Advanced Statistical Testing")
    if df is None: return
    
    num_cols, cat_cols, _ = get_column_types(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Post-Hoc", "Repeated Measures", "Mixed Effects", "Bayesian A/B", "Bootstrap"])
    
    # --- Post-Hoc ---
    with tab1:
        st.subheader("Post-Hoc Tests (Tukey HSD)")
        if len(cat_cols) > 0:
            group_col = st.selectbox("Group", cat_cols, key='ph_g')
            val_col = st.selectbox("Value", num_cols, key='ph_v')
            
            if st.button("Run Tukey HSD"):
                tukey = pairwise_tukeyhsd(endog=df[val_col], groups=df[group_col], alpha=0.05)
                st.text(tukey.summary())
                fig = tukey.plot_simultaneous()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning("No categorical columns.")

    # --- Repeated Measures ---
    with tab2:
        st.subheader("Repeated Measures ANOVA")
        st.info("Requires: Subject ID, Within-Subject Factor, Dependent Variable")
        subj = st.selectbox("Subject ID", df.columns, key='rm_s')
        within = st.selectbox("Within-Subject Factor", df.columns, key='rm_w')
        dep = st.selectbox("Dependent Variable", num_cols, key='rm_d')
        
        if st.button("Run RM ANOVA"):
            try:
                aovrm = AnovaRM(df, dep, subj, within=[within])
                res = aovrm.fit()
                st.text(res.summary())
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Mixed Effects ---
    with tab3:
        st.subheader("Mixed Effects Models")
        formula = st.text_input("Formula (e.g., Weight ~ Time)", key='me_f')
        group = st.selectbox("Group (Random Effect)", df.columns, key='me_g')
        
        if st.button("Fit MixedLM"):
            try:
                model = smf.mixedlm(formula, df, groups=df[group])
                result = model.fit()
                st.text(result.summary())
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Bayesian A/B ---
    with tab4:
        st.subheader("Bayesian A/B Testing")
        col_a = st.number_input("Successes A", 10)
        n_a = st.number_input("Total A", 100)
        col_b = st.number_input("Successes B", 12)
        n_b = st.number_input("Total B", 100)
        
        x = np.linspace(0, 1, 1000)
        pdf_a = stats.beta.pdf(x, col_a + 1, n_a - col_a + 1)
        pdf_b = stats.beta.pdf(x, col_b + 1, n_b - col_b + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf_a, name='Posterior A', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=x, y=pdf_b, name='Posterior B', fill='tozeroy'))
        safe_plot(fig)
        
        prob_b_better = (np.random.beta(col_b+1, n_b-col_b+1, 10000) > np.random.beta(col_a+1, n_a-col_a+1, 10000)).mean()
        st.metric("Probability B is better than A", f"{prob_b_better:.2%}")

    # --- Bootstrap ---
    with tab5:
        st.subheader("Non-Parametric Bootstrap")
        col = st.selectbox("Column", num_cols, key='bs_c')
        n_boot = int(st.number_input("Iterations", 100, 10000, 1000))
        
        if st.button("Run Bootstrap"):
            data = df[col].dropna().values
            means = []
            for _ in range(n_boot):
                sample = np.random.choice(data, size=len(data), replace=True)
                means.append(np.mean(sample))
            
            lower = np.percentile(means, 2.5)
            upper = np.percentile(means, 97.5)
            
            st.metric("Bootstrap Mean", f"{np.mean(means):.2f}")
            st.write(f"95% CI: [{lower:.2f}, {upper:.2f}]")
            fig = px.histogram(x=means, title="Bootstrap Distribution")
            safe_plot(fig)

def render_deep_learning(df):
    st.header("🧠 Deep Learning & Neural Networks")
    if df is None: return
    num_cols, cat_cols, _ = get_column_types(df)
    
    target = st.selectbox("Target", df.columns, key='dl_t')
    features = st.multiselect("Features", [c for c in num_cols if c != target], key='dl_f')
    
    if not features: return
    
    X, y = clean_xy(df, target, features)
    is_class = df[target].dtype == 'object' or df[target].nunique() < 10
    
    if is_class:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    tab1, tab2, tab3, tab4 = st.tabs(["AutoML", "NN Builder", "Hyperparameter Tuning", "Ensembling"])
    
    # --- AutoML ---
    with tab1:
        st.subheader("AutoML Integration")
        if st.button("Run Simple AutoML"):
            models = []
            if is_class:
                models = [
                    ("LogReg", LogisticRegression()),
                    ("RF", RandomForestClassifier()),
                    ("GB", GradientBoostingClassifier())
                ]
            else:
                models = [
                    ("LinReg", LinearRegression()),
                    ("RF", RandomForestRegressor()),
                    ("GB", GradientBoostingRegressor())
                ]
            
            results = []
            for name, model in models:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                results.append({"Model": name, "Score": score})
            
            st.dataframe(pd.DataFrame(results).sort_values("Score", ascending=False))

    # --- NN Builder ---
    with tab2:
        st.subheader("Neural Network Builder (MLP)")
        layers_str = st.text_input("Hidden Layers (comma sep)", "100,50")
        layers = tuple(map(int, layers_str.split(',')))
        activation = st.selectbox("Activation", ['relu', 'tanh', 'logistic'])
        
        if st.button("Train MLP"):
            if is_class:
                mlp = MLPClassifier(hidden_layer_sizes=layers, activation=activation, max_iter=500)
            else:
                mlp = MLPRegressor(hidden_layer_sizes=layers, activation=activation, max_iter=500)
            
            mlp.fit(X_train, y_train)
            st.success(f"Training Complete. Score: {mlp.score(X_test, y_test):.4f}")
            
            fig = px.line(y=mlp.loss_curve_, title="Loss Curve")
            safe_plot(fig)

    # --- Hyperparameter Tuning ---
    with tab3:
        st.subheader("Hyperparameter Tuning (Optuna/GridSearch)")
        if OPTUNA_AVAILABLE:
            st.info("Using Optuna")
            if st.button("Optimize RF"):
                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 10, 100)
                    max_depth = trial.suggest_int('max_depth', 2, 32)
                    if is_class:
                        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    else:
                        clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=3).mean()

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=10)
                st.write("Best Params:", study.best_params)
        else:
            st.info("Optuna not found. Using RandomizedSearchCV")
            if st.button("Optimize RF (RandomSearch)"):
                param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
                if is_class:
                    clf = RandomForestClassifier()
                else:
                    clf = RandomForestRegressor()
                search = RandomizedSearchCV(clf, param_dist, n_iter=5, cv=3)
                search.fit(X_train, y_train)
                st.write("Best Params:", search.best_params_)

    # --- Ensembling ---
    with tab4:
        st.subheader("Model Ensembling")
        if st.button("Train Voting Ensemble"):
            if is_class:
                clf1 = LogisticRegression()
                clf2 = RandomForestClassifier()
                eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='soft')
                eclf.fit(X_train, y_train)
                st.write(f"Ensemble Accuracy: {eclf.score(X_test, y_test):.4f}")
            else:
                reg1 = LinearRegression()
                reg2 = RandomForestRegressor()
                ereg = VotingRegressor(estimators=[('lr', reg1), ('rf', reg2)])
                ereg.fit(X_train, y_train)
                st.write(f"Ensemble R2: {ereg.score(X_test, y_test):.4f}")

def render_nlp_suite(df):
    st.header("📝 NLP Suite")
    if df is None: return
    
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        st.error("No text columns found.")
        return
        
    text_col = st.selectbox("Select Text Column", text_cols)
    text_data = df[text_col].astype(str).dropna()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Topic Modeling", "NER", "Word Cloud"])
    
    # --- Sentiment ---
    with tab1:
        st.subheader("Text Column Analysis (Sentiment)")
        if TEXTBLOB_AVAILABLE:
            if st.button("Analyze Sentiment"):
                polarities = [TextBlob(text).sentiment.polarity for text in text_data]
                df['polarity'] = polarities
                fig = px.histogram(x=polarities, title="Sentiment Polarity Distribution")
                safe_plot(fig)
                st.write("Top Positive:", df.nlargest(5, 'polarity')[[text_col, 'polarity']])
                st.write("Top Negative:", df.nsmallest(5, 'polarity')[[text_col, 'polarity']])
        else:
            st.error("TextBlob not installed.")

    # --- Topic Modeling ---
    with tab2:
        st.subheader("Topic Modeling (LDA)")
        n_topics = st.slider("Number of Topics", 2, 10, 3)
        if st.button("Run LDA"):
            cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = cv.fit_transform(text_data)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(dtm)
            
            for index, topic in enumerate(lda.components_):
                st.write(f"**Topic {index}**")
                top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
                st.write(", ".join(top_words))

    # --- NER ---
    with tab3:
        st.subheader("Named Entity Recognition")
        if SPACY_AVAILABLE:
            if st.button("Extract Entities"):
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    st.warning("Downloading language model 'en_core_web_sm' (one-time setup)...")
                    from spacy.cli import download
                    download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                # Limit to first 50 docs for speed
                docs = list(nlp.pipe(text_data[:50]))
                entities = []
                for doc in docs:
                    for ent in doc.ents:
                        entities.append((ent.text, ent.label_))
                
                ent_df = pd.DataFrame(entities, columns=['Entity', 'Label'])
                st.dataframe(ent_df['Label'].value_counts())
                st.dataframe(ent_df.head(20))
        else:
            st.error("Spacy not installed.")

    # --- Word Cloud ---
    with tab4:
        st.subheader("Word Cloud Generation")
        if WORDCLOUD_AVAILABLE:
            if st.button("Generate Cloud"):
                text = " ".join(text_data)
                wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                st.image(wc.to_array())
        else:
            st.error("WordCloud not installed.")

def render_bi_analytics(df):
    st.header("💼 Business Intelligence")
    if df is None: return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CLV", "Churn", "Retention", "Price Elasticity", "What-If"])
    
    # --- CLV ---
    with tab1:
        st.subheader("Customer Lifetime Value")
        if LIFETIMES_AVAILABLE:
            id_col = st.selectbox("Customer ID", df.columns, key='clv_id')
            date_col = st.selectbox("Date", df.columns, key='clv_date')
            val_col = st.selectbox("Value", df.columns, key='clv_val')
            
            if st.button("Calculate CLV"):
                rfm = summary_data_from_transaction_data(df, id_col, date_col, monetary_value_col=val_col)
                bgf = BetaGeoFitter(penalizer_coef=0.0)
                bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])
                
                rfm['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(30, rfm['frequency'], rfm['recency'], rfm['T'])
                
                ggf = GammaGammaFitter(penalizer_coef=0)
                ggf.fit(rfm['frequency'], rfm['monetary_value'])
                rfm['clv'] = ggf.customer_lifetime_value(bgf, rfm['frequency'], rfm['recency'], rfm['T'], rfm['monetary_value'], time=12, discount_rate=0.01)
                
                st.dataframe(rfm.sort_values('clv', ascending=False).head())
        else:
            st.error("Lifetimes library not installed.")

    # --- Churn ---
    with tab2:
        st.subheader("Churn Prediction Pipeline")
        target = st.selectbox("Churn Target (Binary)", df.columns, key='churn_t')
        features = st.multiselect("Features", df.columns, key='churn_f')
        
        if st.button("Predict Churn"):
            X, y = clean_xy(df, target, features)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            
            st.write("Accuracy:", clf.score(X_test, y_test))
            feat_imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(feat_imp)

    # --- Retention ---
    with tab3:
        st.subheader("Retention Cohort Heatmaps")
        st.info("Requires Date and User ID")
        date_c = st.selectbox("Date", df.columns, key='ret_d')
        id_c = st.selectbox("ID", df.columns, key='ret_i')
        
        if st.button("Generate Cohort"):
            df_cohort = df.copy()
            df_cohort['OrderPeriod'] = df_cohort[date_c].apply(lambda x: x.strftime('%Y-%m'))
            df_cohort.set_index(id_c, inplace=True)
            df_cohort['CohortGroup'] = df_cohort.groupby(level=0)[date_c].min().apply(lambda x: x.strftime('%Y-%m'))
            df_cohort.reset_index(inplace=True)
            
            grouped = df_cohort.groupby(['CohortGroup', 'OrderPeriod'])
            cohorts = grouped.agg({id_c: pd.Series.nunique})
            cohorts.rename(columns={id_c: 'TotalUsers'}, inplace=True)
            
            def cohort_period(df):
                df['CohortPeriod'] = np.arange(len(df)) + 1
                return df
            
            cohorts = cohorts.groupby(level=0).apply(cohort_period)
            cohort_matrix = cohorts.pivot_table(index='CohortGroup', columns='CohortPeriod', values='TotalUsers')
            cohort_size = cohort_matrix.iloc[:,0]
            retention = cohort_matrix.divide(cohort_size, axis=0)
            
            fig = px.imshow(retention, text_auto='.0%', color_continuous_scale='Blues')
            safe_plot(fig)

    # --- Price Elasticity ---
    with tab4:
        st.subheader("Price Elasticity Analysis")
        price_col = st.selectbox("Price Column", df.select_dtypes(include=np.number).columns, key='pe_p')
        qty_col = st.selectbox("Quantity Column", df.select_dtypes(include=np.number).columns, key='pe_q')
        
        if st.button("Calculate Elasticity"):
            # Log-Log model
            df_log = df[[price_col, qty_col]].dropna()
            df_log = df_log[(df_log[price_col] > 0) & (df_log[qty_col] > 0)]
            df_log['log_price'] = np.log(df_log[price_col])
            df_log['log_qty'] = np.log(df_log[qty_col])
            
            # Sanitize columns for statsmodels formula
            # Although 'log_price' and 'log_qty' are safe, we use them directly.
            model = smf.ols("log_qty ~ log_price", data=df_log).fit()
            elasticity = model.params['log_price']
            
            st.metric("Price Elasticity", f"{elasticity:.2f}")
            st.write("Interpretation: A 1% increase in price leads to a {:.2f}% change in quantity.".format(elasticity))
            
            fig = px.scatter(df, x=price_col, y=qty_col, trendline="ols", log_x=True, log_y=True)
            safe_plot(fig)

    # --- What-If ---
    with tab5:
        st.subheader("What-If Scenario Simulator")
        target_wi = st.selectbox("Target Outcome", df.columns, key='wi_t')
        features_wi = st.multiselect("Drivers", [c for c in df.select_dtypes(include=np.number).columns if c != target_wi], key='wi_f')
        
        if features_wi:
            X, y = clean_xy(df, target_wi, features_wi)
            model = LinearRegression()
            model.fit(X, y)
            
            st.write("Adjust sliders to simulate outcome:")
            input_data = {}
            for feat in features_wi:
                val = st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
                input_data[feat] = val
            
            pred = model.predict(pd.DataFrame([input_data])[features_wi])[0]
            st.metric("Predicted Outcome", f"{pred:.2f}")

# ============================================================================
# ADD TO MAIN SIDEBAR NAVIGATION
# ============================================================================

# Update the sidebar phase selection to include new phases:



def sidebar_processor():
    """Updated sidebar with all new phases"""
    with st.sidebar:
        st.markdown("## 🔮 Lumina Analytics Suite")
        
        # New Categorized Navigation
        NAV_STRUCTURE = {
            "🔍 Data & Quality": [
                ('Monitor', 'Monitor'), ('Data Quality', 'Data Quality'), 
                ('Anomaly Detection', 'Anomalies'), ('Feature Eng', 'Features')
            ],
            "📊 Exploratory & Visuals": [
                ('Explore', 'Explore'), ('Correlation', 'Correlations'), ('Cluster', 'Cluster'),
                ('3D Scatter', '3D Scatter'), ('Sankey Diagram', 'Sankey'), ('Network Graph', 'Network')
            ],
            "🤖 Predictive Modeling": [
                ('Predictive Model', 'Models'), ('Regression', 'Regression'), ('Deep Learning', 'Deep Learning'),
                ('Explainability', 'Explainability'), ('NLP Suite', 'NLP Suite')
            ],
            "📉 Statistical Analysis": [
                ('Statistical Test', 'Statistical'), ('Statistics Advanced', 'Statistics+'),
                ('GLM', 'GLM'), ('Multivariate', 'Multivariate'), ('Survival', 'Survival'), 
                ('Power Analysis', 'Power')
            ],
            "📈 Time Series": [
                ('Time Series', 'Time Series'), ('Timeseries Advanced', 'Time Series+')
            ],
            "💼 Business Intelligence": [
                ('Business Analytics', 'Business'), ('BI Analytics', 'BI Analytics'),
                ('Market Basket', 'Market Basket'), ('Pareto Analysis', 'Pareto (80/20)'), 
                ('Impact', 'Impact')
            ],
            "📝 Reporting": [
                ('Report', 'Report'), ('PDF Report', 'PDF Report'), ('Smart Narrative', 'Smart Insights')
            ]
        }
        
        # 1. Select Category
        category = st.radio("Navigation", list(NAV_STRUCTURE.keys()))
        st.divider()
        
        # 2. Select Phase within Category
        # Create mapping for display
        phase_options = NAV_STRUCTURE[category]
        phase_key = st.radio(
            "Module",
            phase_options,
            format_func=lambda x: f"{x[1]}"
        )[0] # Get the Key (0 index)
        
        phase = phase_key # Pass to return

        
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
                
                # Enforce string column names to prevent Streamlit/Plotly type errors (e.g. 1960 int vs str)
                df.columns = df.columns.astype(str)
                
                st.session_state.data = df
                st.success(f"✅ Active Data: {len(df):,} rows")

                with st.expander("🛠️ Robust ETL Pipeline", expanded=False):
                    st.caption(f"Data Cleaning & Transformation")
                    
                    # 1. Column Management
                    st.markdown("**1. Column Management**")
                    all_cols = df.columns.tolist()
                    drop_cols = st.multiselect("Drop Columns", all_cols)
                    if drop_cols: 
                        df = df.drop(columns=drop_cols)
                    
                    # 2. Text Standardization
                    st.markdown("**2. Text Standardization**")
                    text_ops = st.multiselect("Text Operations", 
                                            ["Trim Whitespace", "To Lowercase", "To Uppercase", "Remove Special Chars"])
                    
                    if text_ops:
                        obj_cols = df.select_dtypes(include=['object']).columns
                        for col in obj_cols:
                            if "Trim Whitespace" in text_ops:
                                df[col] = df[col].astype(str).str.strip()
                            if "To Lowercase" in text_ops:
                                df[col] = df[col].astype(str).str.lower()
                            if "To Uppercase" in text_ops:
                                df[col] = df[col].astype(str).str.upper()
                            if "Remove Special Chars" in text_ops:
                                df[col] = df[col].astype(str).str.replace(r'[^A-Za-z0-9\s]', '', regex=True)

                    # 3. Missing Value Handling
                    st.markdown("**3. Missing Values**")
                    if st.checkbox("Handle Missing Data"):
                        miss_method = st.selectbox("Imputation Method", 
                                                 ["Drop Rows", "Fill 0", "Fill Mean (Numeric) / Mode (Cat)", 
                                                  "Forward Fill (Time Series)", "Backward Fill"])
                        
                        if miss_method == "Drop Rows":
                            df = df.dropna()
                        elif miss_method == "Fill 0":
                            df = df.fillna(0)
                        elif miss_method == "Fill Mean (Numeric) / Mode (Cat)":
                            num = df.select_dtypes(include=np.number).columns
                            cat = df.select_dtypes(exclude=np.number).columns
                            if not num.empty: df[num] = df[num].fillna(df[num].mean())
                            for c in cat:
                                if not df[c].mode().empty:
                                    df[c] = df[c].fillna(df[c].mode()[0])
                                else:
                                    df[c] = df[c].fillna("Unknown")
                        elif miss_method == "Forward Fill (Time Series)":
                            df = df.ffill()
                        elif miss_method == "Backward Fill":
                            df = df.bfill()
                            
                    # 4. Data Type Enforcement
                    st.markdown("**4. Type Standardization**")
                    if st.checkbox("Enforce Numeric Types"):
                        num_force_cols = st.multiselect("Select Columns to Force Numeric", df.columns)
                        for c in num_force_cols:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                            
                    # 5. Deduplication
                    st.markdown("**5. Deduplication**")
                    if st.checkbox("Remove Duplicate Rows"):
                        init_len = len(df)
                        df = df.drop_duplicates()
                        final_len = len(df)
                        if init_len != final_len:
                            st.info(f"Removed {init_len - final_len} duplicates")

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
    # Advanced Phases
    elif phase == '3D Scatter': safe_render(render_3d_scatter, df)
    elif phase == 'Sankey Diagram': safe_render(render_sankey_diagram, df)
    elif phase == 'Network Graph': safe_render(render_network_graph, df)
    elif phase == 'PDF Report': safe_render(render_pdf_report, df)
    
    # Advanced Suites
    elif phase == 'Explainability': safe_render(render_explainability, df)
    elif phase == 'Timeseries Advanced': safe_render(render_advanced_timeseries, df)
    elif phase == 'Statistics Advanced': safe_render(render_advanced_stats, df)
    elif phase == 'Deep Learning': safe_render(render_deep_learning, df)
    elif phase == 'NLP Suite': safe_render(render_nlp_suite, df)
    elif phase == 'BI Analytics': safe_render(render_bi_analytics, df)

if __name__ == "__main__":
    main()