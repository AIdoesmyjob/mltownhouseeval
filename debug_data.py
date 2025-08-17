import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Jan 1 2015_Aug 13 2025.csv')
print(f"Initial shape: {df.shape}")

# Check Price column
print(f"\nPrice column type: {df['Price'].dtype}")
print(f"Price sample:\n{df['Price'].head()}")

# Convert price
def to_numeric(series):
    if pd.api.types.is_object_dtype(series):
        series = (series.astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .str.replace(' ', '', regex=False))
    return pd.to_numeric(series, errors='coerce')

df['Price'] = to_numeric(df['Price'])
print(f"\nAfter conversion, Price type: {df['Price'].dtype}")
print(f"Price sample:\n{df['Price'].head()}")

# Check for valid prices
valid_mask = df['Price'].notna() & (df['Price'] > 0)
print(f"\nValid prices: {valid_mask.sum()}")

# Check List Date
print(f"\nList Date sample:\n{df['List Date'].head()}")
df['List Date'] = pd.to_datetime(df['List Date'], errors='coerce')
print(f"After conversion:\n{df['List Date'].head()}")

# Filter
df_filtered = df[valid_mask].copy()
print(f"\nFiltered shape: {df_filtered.shape}")

# Check if we have the data we need
print(f"\nHas TypeDwel: {'TypeDwel' in df.columns}")
print(f"Has Postal Code: {'Postal Code' in df.columns}")
print(f"Has TotFlArea: {'TotFlArea' in df.columns}")

if 'TotFlArea' in df.columns:
    df_filtered['TotFlArea'] = to_numeric(df_filtered['TotFlArea'])
    print(f"TotFlArea sample:\n{df_filtered['TotFlArea'].head()}")
    print(f"TotFlArea > 0: {(df_filtered['TotFlArea'] > 0).sum()}")