
import unittest
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class TestRealData(unittest.TestCase):
    def setUp(self):
        self.xls_path = os.path.join(os.path.dirname(__file__), "API_NY.GDP.MKTP.CD_DS2_en_excel_v2_174254.xls")
        self.xlsx_path = os.path.join(os.path.dirname(__file__), "API_NY.GDP.MKTP.CD_DS2_en_excel_v2_174254.xlsx")
        
    def test_parse_world_bank_xls(self):
        """Test parsing of the World Bank XLS file with metadata rows."""
        import io
        if not os.path.exists(self.xls_path):
            self.skipTest(f"File {self.xls_path} not found")
            
        print(f"\nTesting parsing of {self.xls_path}...")
        
        with open(self.xls_path, 'rb') as f:
            file_content = f.read()
            
        # Create BytesIO object mimicking UploadedFile
        mock_file = io.BytesIO(file_content)
        mock_file.name = self.xls_path
        
        df_result, file_type = app.parse_uploaded_file(mock_file)
        
        # WB files often have multiple sheets ("Data", "Metadata...")
        if file_type == "multi":
            print(f"Detected Multi-sheet file. Sheets: {list(df_result.keys())}")
            # Usually the first sheet is the data
            df = list(df_result.values())[0]
        else:
            df = df_result
            
        self.assertIsNotNone(df)
        
        # World Bank data header check
        print("Columns Found:", df.columns.tolist()[:5])
        self.assertIn("Country Name", df.columns, "Robust parser failed to find 'Country Name' header")
        self.assertFalse(df.empty)

    def test_parse_world_bank_xlsx(self):
        """Test parsing of the World Bank XLSX file."""
        import io
        if not os.path.exists(self.xlsx_path):
            self.skipTest(f"File {self.xlsx_path} not found")

        print(f"\nTesting parsing of {self.xlsx_path}...")
        
        with open(self.xlsx_path, 'rb') as f:
            file_content = f.read()
            
        mock_file = io.BytesIO(file_content)
        mock_file.name = self.xlsx_path
            
        df_result, file_type = app.parse_uploaded_file(mock_file)
        
        if file_type == "multi":
             df = list(df_result.values())[0]
        else:
             df = df_result
             
        self.assertIsNotNone(df)
        self.assertIn("Country Name", df.columns)

    @patch('app.st')
    def test_simulation_run(self, mock_st):
        """Run a simulation of the app with the loaded data."""
        import io
        if not os.path.exists(self.xlsx_path):
            self.skipTest("XLSX file missing for simulation")
            
        with open(self.xlsx_path, 'rb') as f:
            file_content = f.read()
            
        mock_file = io.BytesIO(file_content)
        mock_file.name = self.xlsx_path
        
        df_result, file_type = app.parse_uploaded_file(mock_file)
        
        if file_type == "multi":
             df = list(df_result.values())[0]
        else:
             df = df_result
             
        # Mock st functions to prevent crashes
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)] if isinstance(n, int) else [MagicMock() for _ in n]
        
        # Smart Selectbox Mock: Returns first option available
        def smart_selectbox(label, options, **kwargs):
            if options and len(options) > 0:
                return options[0]
            return None
            
        mock_st.selectbox.side_effect = smart_selectbox
        mock_st.multiselect.return_value = df.columns[:3].tolist()
        
        # 1. Test Explore
        print("\nSimulating Explore Phase...")
        app.render_explore(df)
        
        # 2. Test Data Quality
        print("Simulating Data Quality Phase...")
        app.render_data_quality(df)
        
        # 3. Test Statistical
        print("Simulating Statistical Test Phase...")
        # Need numeric cols. World Bank data is wide (years are cols)
        # We need to coerce year cols to numeric if they aren't
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) == 0:
             print("Warning: No numeric columns found. WB data might be strings?")
        else:
             app.render_statistical_test(df)

if __name__ == '__main__':
    unittest.main()
