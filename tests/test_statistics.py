
import unittest
import pandas as pd
import numpy as np
import io
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure app can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class TestStatisticalFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load real data once for all tests."""
        cls.xlsx_path = os.path.join(os.path.dirname(__file__), "API_NY.GDP.MKTP.CD_DS2_en_excel_v2_174254.xlsx")
        if not os.path.exists(cls.xlsx_path):
            raise unittest.SkipTest("World Bank Data file not found")
            
        with open(cls.xlsx_path, 'rb') as f:
            file_content = f.read()
        
        mock_file = io.BytesIO(file_content)
        mock_file.name = cls.xlsx_path
        
        df_result, file_type = app.parse_uploaded_file(mock_file)
        if file_type == "multi":
            cls.df = list(df_result.values())[0]
        else:
            cls.df = df_result

        # --- PRE-PROCESSING FOR TESTS ---
        # 1. Convert columns to string first to find years
        cls.df.columns = cls.df.columns.astype(str)
        
        # 2. Identify likely year columns (numeric headers)
        year_cols = [c for c in cls.df.columns if c.isdigit()]
        
        # 3. Create a clean working Copy
        cls.data = cls.df.copy()
        
        # 4. Enforce numeric on year columns
        for c in year_cols:
            cls.data[c] = pd.to_numeric(cls.data[c], errors='coerce')
            
        # 5. Create a Dummy Grouping Variable for T-Test/ANOVA
        # Split data into 3 groups based on row index
        conditions = [
            (cls.data.index % 3 == 0),
            (cls.data.index % 3 == 1),
            (cls.data.index % 3 == 2)
        ]
        choices = ['Group A', 'Group B', 'Group C']
        cls.data['Test_Group'] = np.select(conditions, choices, default='Group A')
        
        # 6. Create a Dummy categorical for Chi-Square
        cls.data['Region_Dummy'] = np.where(cls.data.index % 2 == 0, 'North', 'South')
        
        # 7. Select a target Analysis Column (e.g., '2020')
        # Use the last available year column that has data
        valid_years = [y for y in year_cols if cls.data[y].count() > 10]
        if valid_years:
            cls.target_col = valid_years[-1]
        else:
            cls.target_col = None # Should fail if no data
            
        print(f"\n[Setup] Loaded Data. Rows: {len(cls.data)}. Target Metric: {cls.target_col}")

    def setUp(self):
        # Mock Streamlit to prevent rendering
        self.mock_st = patch('app.st').start()
        self.mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        
    def tearDown(self):
        patch.stopall()

    def test_01_shapiro_wilk(self):
        """Test Normality (Shapiro-Wilk)"""
        print("Testing Shapiro-Wilk...")
        self.mock_st.selectbox.side_effect = ["Normality Test (Shapiro-Wilk)", self.target_col]
        app.render_statistical_test(self.data)
        # Verify metric called (means test ran)
        # We can't easily check metric calls on mock columns without more setup, 
        # but lack of exception is good.

    def test_02_levene(self):
        """Test Levene (Equal Variance)"""
        print("Testing Levene...")
        # Select Test -> Variable -> Group
        self.mock_st.selectbox.side_effect = [
            "Equal Variance Test (Levene)", 
            self.target_col, 
            'Test_Group'
        ]
        app.render_statistical_test(self.data)

    def test_03_ttest(self):
        """Test Independent T-Test"""
        print("Testing T-Test...")
        # Select Test -> Variable -> Group -> Group 1 -> Group 2
        self.mock_st.selectbox.side_effect = [
            "T-Test (Independent Samples)", 
            self.target_col, 
            'Test_Group',
            'Group A',
            'Group B'
        ]
        app.render_statistical_test(self.data)

    def test_04_mann_whitney(self):
        """Test Mann-Whitney U"""
        print("Testing Mann-Whitney...")
        self.mock_st.selectbox.side_effect = [
            "Mann-Whitney U Test", 
            self.target_col, 
            'Test_Group',
            'Group A',
            'Group B'
        ]
        app.render_statistical_test(self.data)

    def test_05_anova(self):
        """Test One-Way ANOVA"""
        print("Testing ANOVA...")
        self.mock_st.selectbox.side_effect = [
            "One-Way ANOVA", 
            self.target_col, 
            'Test_Group'
        ]
        app.render_statistical_test(self.data)

    def test_06_kruskal(self):
        """Test Kruskal-Wallis"""
        print("Testing Kruskal-Wallis...")
        self.mock_st.selectbox.side_effect = [
            "Kruskal-Wallis Test", 
            self.target_col, 
            'Test_Group'
        ]
        app.render_statistical_test(self.data)

    def test_07_correlation(self):
        """Test Correlation Analysis"""
        print("Testing Correlation...")
        self.mock_st.selectbox.side_effect = [
            "Correlation Analysis",
            self.target_col,    # Var 1
            '2019'              # Var 2 (Assumed available from set up)
        ]
        app.render_statistical_test(self.data)

    def test_08_chisquare(self):
        """Test Chi-Square"""
        print("Testing Chi-Square...")
        self.mock_st.selectbox.side_effect = [
            "Chi-Square Test of Independence",
            'Test_Group',
            'Region_Dummy'
        ]
        app.render_statistical_test(self.data)

if __name__ == '__main__':
    unittest.main()
