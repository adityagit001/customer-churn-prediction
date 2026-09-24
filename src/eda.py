import pandas as pd
import numpy as np

def inspect_dataset(df):
    print("--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Distribution (Churn):\n", df['Churn'].value_counts(normalize=True) * 100)
    
    # Check data types and problematic columns
    print("\nData Types:\n", df.dtypes)
    