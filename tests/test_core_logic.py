
import unittest
import pandas as pd
import numpy as np
import io
import sys
import os
from unittest.mock import MagicMock, patch

# --- MOCK SETUP START ---
# We must mock 'streamlit' BEFORE importing the app to handle decorators like @st.cache_data
sys.modules['streamlit'] = MagicMock()
import streamlit as st
# Helper to make cache_data a simple pass-through decorator
def cache_data_mock(show_spinner=False):
    def decorator(func):
        return func
    return decorator
st.cache_data = cache_data_mock
# --- MOCK SETUP END ---

# Now import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class TestCoreLogic(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [10.5, 20.1, 30.2, 40.5, 50.1]
        })
        
        # Setup Mock Session State
        # Configured to behave like an object (st.session_state.foo) AND a dict (st.session_state['foo'])
        self.mock_session_state = MagicMock()
        self.mock_session_state.__setitem__ = self.mock_session_state.__setattr__
        self.mock_session_state.__getitem__ = self.mock_session_state.__getattr__
        self.mock_session_state.__contains__.side_effect = lambda key: hasattr(self.mock_session_state, key)
        
        # Patch st.session_state in the app module
        self.session_patcher = patch('app.st.session_state', self.mock_session_state)
        self.session_patcher.start()

    def tearDown(self):
        self.session_patcher.stop()

    def test_check_dataset_size_small(self):
        """Test check_dataset_size with a small dataset"""
        with patch('app.st.checkbox', return_value=True):
           result = app.check_dataset_size(self.df, limit=100)
           self.assertTrue(len(result) <= 100)

    def test_smart_date_converter(self):
        """Test date conversion"""
        df_date = pd.DataFrame({'Date': ['2021-01-01', '2021-01-02']})
        df_conv = app.smart_date_converter(df_date)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_conv['Date']))

    def test_csv_ingestion(self):
        """Test CSV parsing logic"""
        csv_content = b"col1,col2\n1,2\n3,4"
        # Mock file object needs 'name' and 'seek' and output suitable for pd.read_csv
        csv_file = io.BytesIO(csv_content)
        csv_file.name = "test.csv"
        
        # We also need to patch pd.read_csv if we want to isolate from pandas, 
        # but here we want to test the logic using real pandas.
        # The issue before was likely st.cache_data hashing the file object.
        # With st.cache_data mocked as pass-through, this should work.
        
        df, file_type = app.parse_uploaded_file(csv_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(file_type, 'single')

    def test_pipeline_manager_add_step(self):
        """Test PipelineManager adding steps"""
        # Ensure 'etl_pipeline' exists (simulated __contains__)
        # self.mock_session_state.etl_pipeline = []  <-- Init handled by __init__
        
        # We need to simulate the 'if not in session_state' check in __init__
        # By default, mock attributes don't exist until set. 
        # However, our __contains__ logic uses hasattr.
        
        # Force 'etl_pipeline' to NOT exist initially so __init__ creates it
        del self.mock_session_state.etl_pipeline 
        
        pm = app.PipelineManager()
        pm.add_step("drop_cols", {'cols': ['A']}, "Drop A")
        
        pipeline = pm.get_pipeline()
        self.assertEqual(len(pipeline), 1)
        self.assertEqual(pipeline[0]['type'], 'drop_cols')

    def test_sanitize_col_name(self):
        """Test column name sanitization for statsmodels"""
        self.assertEqual(app.sanitize_col_name("ValidName"), "ValidName")
        self.assertEqual(app.sanitize_col_name("Invalid Name"), "Q('Invalid Name')")
        self.assertEqual(app.sanitize_col_name("123Start"), "Q('123Start')")

    def test_recommend_chart_type(self):
        """Test chart recommendation logic"""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6],
            'cat1': ['a', 'b', 'c'],
            'date1': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        
        # Numeric vs Numeric -> Scatter
        self.assertEqual(app.recommend_chart_type('num1', 'num2', df), "Scatter Plot")
        # Date vs Numeric -> Line Chart
        self.assertEqual(app.recommend_chart_type('date1', 'num1', df), "Line Chart")
        # Cat vs Numeric -> Box Plot
        self.assertEqual(app.recommend_chart_type('cat1', 'num1', df), "Box Plot")
        # Numeric vs Cat -> Box Plot (Horiz)
        self.assertEqual(app.recommend_chart_type('num1', 'cat1', df), "Box Plot (Horiz)")
        # Cat vs Cat -> Heatmap
        self.assertEqual(app.recommend_chart_type('cat1', 'cat1', df), "Heatmap")
        # Single Numeric -> Histogram
        self.assertEqual(app.recommend_chart_type('num1', None, df), "Histogram")

    def test_find_valid_header_row(self):
        """Test header detection heuristic"""
        # Case 1: Header at row 0
        df1 = pd.DataFrame([['A', 'B'], [1, 2], [3, 4]])
        self.assertEqual(app.find_valid_header_row(df1), 0)
        
        # Case 2: Garbage at row 0, Header at row 1
        df2 = pd.DataFrame([[np.nan, np.nan], ['A', 'B'], [1, 2]])
        self.assertEqual(app.find_valid_header_row(df2), 1)

    def test_get_column_types(self):
        """Test column type classification"""
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat': ['a', 'b', 'c'],
            'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        })
        
        numeric, categorical, datetime_cols = app.get_column_types(df)
        self.assertIn('num', numeric)
        self.assertIn('cat', categorical)
        self.assertIn('date', datetime_cols)

if __name__ == '__main__':
    unittest.main()
