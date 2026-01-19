import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Mock Streamlit BEFORE importing app
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Mock cache_data to handle both @st.cache_data and @st.cache_data(...)
def mock_cache_data(*args, **kwargs):
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

st.cache_data = mock_cache_data
st.sidebar = MagicMock()
st.columns = lambda n: [MagicMock() for _ in range(n)]
st.tabs = lambda names: [MagicMock() for _ in range(len(names))]

# Add app path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    render_explainability,
    render_deep_learning,
    render_nlp_suite,
    render_advanced_timeseries,
    render_advanced_stats,
    render_bi_analytics,
    get_column_types
)

class TestAdvancedSuites(unittest.TestCase):

    def setUp(self):
        # Create Synthetic Data
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            'numeric_1': np.random.randn(n),
            'numeric_2': np.random.rand(n) * 100,
            'category_1': np.random.choice(['A', 'B', 'C'], n),
            'target_reg': np.random.randn(n) + 10,
            'target_class': np.random.choice([0, 1], n),
            'date': pd.date_range('2023-01-01', periods=n),
            'text': np.random.choice(['Good product', 'Bad service', 'Okay', 'Excellent stuff'], n),
            'id': range(n)
        })

    def test_explainability_smoke(self):
        """Smoke test for Explainability Suite"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.multiselect') as mock_multi, \
                 patch('streamlit.slider') as mock_slider:
                
                mock_sel.side_effect = ['target_reg', 'numeric_1']
                mock_multi.return_value = ['numeric_1', 'numeric_2']
                mock_slider.return_value = 0 # Index
                
                render_explainability(self.df)
        except Exception as e:
            self.fail(f"Explainability Suite crashed: {e}")

    def test_deep_learning_smoke(self):
        """Smoke test for Deep Learning"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.multiselect') as mock_multi, \
                 patch('streamlit.text_input') as mock_txt:
                
                mock_sel.side_effect = ['target_reg', 'relu']
                mock_multi.return_value = ['numeric_1', 'numeric_2']
                mock_txt.side_effect = ["64,32"] 
                
                render_deep_learning(self.df)
        except Exception as e:
            self.fail(f"Deep Learning Suite crashed: {e}")

    def test_nlp_suite_smoke(self):
        """Smoke test for NLP"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.slider') as mock_slider:
                
                mock_sel.return_value = 'text'
                mock_slider.return_value = 2 # n_topics
                render_nlp_suite(self.df)
        except Exception as e:
            self.fail(f"NLP Suite crashed: {e}")

    def test_time_series_smoke(self):
        """Smoke test for Time Series"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.number_input') as mock_num, \
                 patch('streamlit.slider') as mock_slider, \
                 patch('streamlit.multiselect') as mock_multi:
                
                mock_sel.side_effect = ['date', 'numeric_1']
                mock_multi.return_value = ['numeric_1', 'numeric_2']
                mock_num.return_value = 1 # p, d, q, etc.
                mock_slider.return_value = 10 # window
                
                render_advanced_timeseries(self.df)
        except Exception as e:
            self.fail(f"Time Series Suite crashed: {e}")

    def test_advanced_stats_smoke(self):
        """Smoke test for Stats"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.multiselect') as mock_multi, \
                 patch('streamlit.number_input') as mock_num:
                
                # PostHoc: Group(Cat), Value(Num). RM: Subject(ID), Within(Cat), Dep(Num). Mixed: Group(Cat). Boot: Col(Num)
                mock_sel.side_effect = ['category_1', 'numeric_1', 'id', 'category_1', 'numeric_1', 'category_1', 'numeric_1']
                mock_num.return_value = 100
                
                render_advanced_stats(self.df)
        except Exception as e:
            self.fail(f"Advanced Stats Suite crashed: {e}")
            
    def test_bi_analytics_smoke(self):
        """Smoke test for BI"""
        try:
            with patch('streamlit.selectbox') as mock_sel, \
                 patch('streamlit.multiselect') as mock_multi, \
                 patch('streamlit.slider') as mock_slider:
                
                # Churn Target (Binary), Features, CLV ID, etc...
                # The suite has multiple tabs. Churn is Tab 2. 
                # Ideally we mock side_effect to handle calls in order: 
                # Tab1 (CLV): ID, Date, Value. Tab2 (Churn): Target, Features...
                # Let's provide a safe sequence.
                mock_sel.side_effect = ['id', 'date', 'numeric_1', 'target_class'] + ['numeric_1']*10
                mock_multi.return_value = ['numeric_2']
                mock_slider.return_value = 10
                
                render_bi_analytics(self.df)
        except Exception as e:
            self.fail(f"BI Suite crashed: {e}")

if __name__ == '__main__':
    unittest.main()
