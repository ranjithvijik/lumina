import unittest
import pandas as pd
import sys
import os
import io

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import parse_uploaded_file

class MockUploadedFile:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        
    def read(self):
        with open(self.path, 'rb') as f:
            return f.read()
            
    # Allow passing self to pandas read functions which expect a file-like object or path
    # But parse_uploaded_file calls pd.read_csv(uploaded_file). 
    # For pd.read_csv to work on an object, it usually needs a .read() method OR be a path.
    # But Streamlit UploadedFile works like a BytesIO.
    # Let's make this class behave like an open file.
    
    def __getattr__(self, name):
        # Delegate to a real file object
        with open(self.path, 'rb') as f:
             # This is tricky because we need the file to stay open.
             # Better approach: Load content into BytesIO and add .name
             pass
             
class MockFile(io.BytesIO):
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        super().__init__(data)
        # Streamlit hasher tries to stat the file if .name is present.
        # We must provide the actual path so it can find it.
        # Logic in app.py parses extension from .name, which works with full paths too.
        self.name = path

class TestFileParsing(unittest.TestCase):
    
    def test_csv_parsing(self):
        f = MockFile('tests/data/test.csv')
        df, status = parse_uploaded_file(f)
        self.assertEqual(status, 'single')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        
    def test_excel_parsing(self):
        f = MockFile('tests/data/test.xlsx')
        df, status = parse_uploaded_file(f)
        self.assertEqual(status, 'single')
        self.assertEqual(len(df), 10)
        
    def test_json_parsing(self):
        f = MockFile('tests/data/test.json')
        df, status = parse_uploaded_file(f)
        self.assertEqual(status, 'single')
        self.assertEqual(len(df), 10)
        
    def test_parquet_parsing(self):
        f = MockFile('tests/data/test.parquet')
        df, status = parse_uploaded_file(f)
        self.assertEqual(status, 'single')
        self.assertEqual(len(df), 10)

if __name__ == '__main__':
    unittest.main()
