
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import app

class TestUserAcceptance(unittest.TestCase):
    """
    Simulates User Acceptance Testing (UAT) by verifying end-to-end workflows.
    We mock the UI (Streamlit) but execute the real logic engines.
    """

    def setUp(self):
        # Create a realistic "Happy Path" dataset
        np.random.seed(42)
        self.df = pd.DataFrame({
            'transaction_id': range(100),
            'sales': np.random.normal(100, 20, 100),
            'profit': np.random.normal(50, 10, 100),
            'cost': np.random.normal(30, 5, 100), # Added for regression features
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range(start='2023-01-01', periods=100)
        })
        
        # Ensure correct types
        self.df['sales'] = self.df['sales'].astype(float)
        self.df['profit'] = self.df['profit'].astype(float)
        self.df['cost'] = self.df['cost'].astype(float)

    @patch('app.st')
    def test_uat_workflow_exploration(self, mock_st):
        """UAT Scenario 1: User Uploads Data and runs Exploration."""
        print("\nTesting UAT Scenario 1: Exploration Phase...")
        
        # Mock columns to return 4 mocks
        mock_st.columns.return_value = [MagicMock() for _ in range(4)]
        
        # Mock Selectbox to return valid column names
        def side_effect_explore(label, options, **kwargs):
            if "X Axis" in label: return 'sales'
            if "Y Axis" in label: return 'profit'
            if "Chart Type" in label: return 'Scatter Plot'
            if "Color" in label: return 'None'
            return options[0] if options else None
            
        mock_st.selectbox.side_effect = side_effect_explore
        
        # 1. Simulate "Explore" Phase
        app.render_explore(self.df)
        
        # Verify Key Interactions
        # render_explore generates chart -> safe_plot -> st.plotly_chart
        self.assertTrue(mock_st.plotly_chart.called)
        
    @patch('app.st')
    def test_uat_workflow_correlation(self, mock_st):
        """UAT Scenario 2: User runs Correlation Analysis."""
        print("\nTesting UAT Scenario 2: Correlation Phase...")
        
        # Mock columns flexible return
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)] if isinstance(n, int) else [MagicMock() for _ in n]
        
        # Execute Logic
        app.render_correlation_analysis(self.df)
        
        # Check validation
        self.assertTrue(mock_st.plotly_chart.called)
        
    @patch('app.st')
    def test_uat_workflow_regression(self, mock_st):
        """UAT Scenario 3: User runs Regression Modeling."""
        print("\nTesting UAT Scenario 3: Regression Phase...")
        
        # Mock inputs
        def side_effect_selectbox(label, options, **kwargs):
            if "Select Target" in label: return 'sales'
            return options[0] if options else None

        mock_st.selectbox.side_effect = side_effect_selectbox
        mock_st.multiselect.return_value = ['profit', 'cost']
        mock_st.radio.return_value = "Linear Regression (OLS)"
        
        # Mock Button Click to TRAIN
        mock_st.button.return_value = True
        
        # Mock columns: Sequence of calls
        # 1. st.columns(2) for inputs
        # 2. st.columns(3) for metrics
        cols_input = [MagicMock(), MagicMock()]
        cols_metric = [MagicMock(), MagicMock(), MagicMock()]
        
        # side_effect iterates through the list for each call
        mock_st.columns.side_effect = [cols_input, cols_metric]
        
        # Execute
        app.render_regression(self.df)
        
        # Verify Success
        # Check if metric was called on the metric columns
        self.assertTrue(cols_metric[0].metric.called)

if __name__ == '__main__':
    unittest.main()
