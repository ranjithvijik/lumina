import unittest
import pandas as pd
import numpy as np
import sys
import os
import io

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import get_column_types, clean_xy, smart_date_converter, parse_uploaded_file

class TestLuminaUtils(unittest.TestCase):
    
    def setUp(self):
        # Create sample dataframes for testing
        self.df_mixed = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'D': [1.1, 2.2, 3.3]
        })
        
        self.df_nan = pd.DataFrame({
            'target': [1, 2, np.nan, 4],
            'feat1': [10, 20, 30, 40],
            'feat2': [1, np.nan, 3, 4]
        })

    def test_get_column_types(self):
        """Test column type detection logic."""
        num, cat, dates = get_column_types(self.df_mixed)
        
        self.assertIn('A', num)
        self.assertIn('D', num)
        self.assertIn('B', cat)
        self.assertIn('C', dates)
        
        # Verify lists are disjoint where expected
        self.assertTrue(set(num).isdisjoint(set(cat)))
        self.assertTrue(set(num).isdisjoint(set(dates)))

    def test_clean_xy_output_shape(self):
        """Test that only complete rows are kept."""
        X, y = clean_xy(self.df_nan, 'target', ['feat1', 'feat2'])
        
        # Expected: 2 rows kept (rows 0 and 3). 
        # Row 1 has NaN in feature, Row 2 has NaN in target.
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)
        
        # Check integrity
        self.assertEqual(X.iloc[0]['feat1'], 10)
        self.assertEqual(y.iloc[0], 1)

    def test_clean_xy_alignment(self):
        """Test that X and y remain aligned by index."""
        X, y = clean_xy(self.df_nan, 'target', ['feat1'])
        # Only row 2 (index 2) dropped due to NaN target
        # Row 1 kept (target=2, feat1=20) - checking my logic...
        # clean_xy does dropna on subset.
        # df_nan:
        # 0: 1, 10, 1 -> OK
        # 1: 2, 20, NaN -> feat2 is NaN. But we only requested ['feat1']. So this row should be KEPT?
        # Let's verify clean_xy implementation: data = df[[target] + features].dropna()
        # If I select target='target', features=['feat1'], then subset is:
        # 0: 1, 10
        # 1: 2, 20
        # 2: NaN, 30
        # 3: 4, 40
        # So row 2 is dropped. Rows 0, 1, 3 remain. Total 3.
        
        X_sub, y_sub = clean_xy(self.df_nan, 'target', ['feat1'])
        self.assertEqual(len(X_sub), 3)
        self.assertEqual(y_sub.iloc[1], 2) # Index 1 should be present

    def test_smart_date_converter(self):
        """Test automatic string-to-datetime conversion."""
        df_dates = pd.DataFrame({
            'date_str': ['2023-01-01', '2023-01-02', 'invalid'],
            'random': ['a', 'b', 'c']
        })
        
        df_conv = smart_date_converter(df_dates.copy())
        
        # 'date_str' should detect 'date' in name and try conversion.
        # 'invalid' will likely become NaT or prevent conversion depending on implementation.
        # Implementation uses pd.to_datetime without errors='coerce' inside a try/except block.
        # If one value fails, the whole block fails catch(ValueError). So it should remain object.
        self.assertEqual(df_conv['date_str'].dtype, 'object')
        
        # Try a valid one
        df_valid = pd.DataFrame({'date_col': ['2023-01-01', '2023-01-02']})
        df_valid_conv = smart_date_converter(df_valid)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_valid_conv['date_col']))

    def test_safe_dataframe_logic(self):
        """Test safe_dataframe logic (not the actual streamlit call)."""
        # Test mixed type conversion
        df_mixed = pd.DataFrame({'mixed': [1, 'a', 2.5]})
        # In app.py, safe_dataframe copies and converts object cols to str
        # We can simulate the logic here or import the function if possible (but it calls st.dataframe)
        # Since safe_dataframe is in app.py and calls st, we can mock st.
        
        from unittest.mock import patch, MagicMock
        with patch('app.st.dataframe') as mock_st_df:
            import app
            app.safe_dataframe(df_mixed, use_container_width=True)
            
            # Verify call args
            args, kwargs = mock_st_df.call_args
            df_arg = args[0]
            
            # Check type conversion
            self.assertTrue(pd.api.types.is_string_dtype(df_arg['mixed']))
            
            # Check kwarg replacement
            self.assertNotIn('use_container_width', kwargs)
            self.assertEqual(kwargs.get('width'), 'stretch')

    def test_safe_plot_logic(self):
        """Test safe_plot logic."""
        from unittest.mock import patch, MagicMock
        mock_fig = MagicMock()
        
        with patch('app.st.plotly_chart') as mock_st_plot:
            import app
            app.safe_plot(mock_fig, use_container_width=True)
            
            args, kwargs = mock_st_plot.call_args
            
            self.assertEqual(args[0], mock_fig)
            self.assertNotIn('use_container_width', kwargs)
            self.assertEqual(kwargs.get('width'), 'stretch')

if __name__ == '__main__':
    unittest.main()
