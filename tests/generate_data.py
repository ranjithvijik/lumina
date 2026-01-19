import pandas as pd
import numpy as np
import os

# Create dummy data
df = pd.DataFrame({
    'id': range(1, 11),
    'category': ['A', 'B'] * 5,
    'value': np.random.randn(10),
    'date': pd.date_range('2023-01-01', periods=10)
})

# Save as CSV
df.to_csv('tests/data/test.csv', index=False)

# Save as Excel
df.to_excel('tests/data/test.xlsx', index=False)

# Save as JSON
df.to_json('tests/data/test.json', orient='records')

# Save as Parquet
df.to_parquet('tests/data/test.parquet')

print("Test data generated in tests/data/")
