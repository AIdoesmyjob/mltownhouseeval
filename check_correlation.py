"""
Check if Square Footage and Number of Floors are actually the same signal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/monstrcow/projects/TownhouseAlgo/Same Townhouse Data from May 1st 2025.csv')

# Clean the data
df['SqFt'] = pd.to_numeric(df['TotFlArea'].str.replace(',', ''), errors='coerce')
df['Floors'] = df['No. Floor Levels'].fillna(1)
df['Price_Clean'] = pd.to_numeric(df['Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# Remove outliers
df_clean = df[(df['SqFt'] > 0) & (df['SqFt'] < 5000)].copy()

print("="*60)
print("INVESTIGATING: Are SqFt and Floors the same signal?")
print("="*60)

# 1. Direct correlation
correlation = df_clean[['SqFt', 'Floors', 'Price_Clean']].corr()
print("\n1. CORRELATION MATRIX:")
print(correlation.round(3))

# 2. Average SqFt by number of floors
print("\n2. AVERAGE SQFT BY NUMBER OF FLOORS:")
sqft_by_floors = df_clean.groupby('Floors')['SqFt'].agg(['mean', 'median', 'count', 'std']).round(0)
print(sqft_by_floors)

# 3. Are they redundant? Check multicollinearity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Can we predict floors from sqft?
X = df_clean[['SqFt']].dropna()
y = df_clean.loc[X.index, 'Floors']
model = LinearRegression()
model.fit(X, y)
r2_floors_from_sqft = r2_score(y, model.predict(X))

print(f"\n3. PREDICTABILITY TEST:")
print(f"   Can we predict Floors from SqFt? R² = {r2_floors_from_sqft:.3f}")
if r2_floors_from_sqft > 0.7:
    print("   ✓ YES - They're highly related (multicollinear)")
else:
    print("   ✗ NO - They capture different information")

# 4. Check if they provide different information about price
print("\n4. INDEPENDENT VALUE TEST:")

# Model 1: Just SqFt
X1 = df_clean[['SqFt']].dropna()
y1 = df_clean.loc[X1.index, 'Price_Clean']
model1 = LinearRegression()
model1.fit(X1, y1)
r2_sqft_only = r2_score(y1, model1.predict(X1))

# Model 2: Just Floors
X2 = df_clean[['Floors']].dropna()
y2 = df_clean.loc[X2.index, 'Price_Clean']
model2 = LinearRegression()
model2.fit(X2, y2)
r2_floors_only = r2_score(y2, model2.predict(X2))

# Model 3: Both
X3 = df_clean[['SqFt', 'Floors']].dropna()
y3 = df_clean.loc[X3.index, 'Price_Clean']
model3 = LinearRegression()
model3.fit(X3, y3)
r2_both = r2_score(y3, model3.predict(X3))

print(f"   SqFt alone → Price:     R² = {r2_sqft_only:.3f}")
print(f"   Floors alone → Price:   R² = {r2_floors_only:.3f}")
print(f"   Both together → Price:  R² = {r2_both:.3f}")

improvement = r2_both - r2_sqft_only
print(f"\n   Adding Floors to SqFt improves R² by: {improvement:.3f}")
if improvement > 0.01:
    print("   ✓ Floors adds independent value beyond SqFt")
else:
    print("   ✗ Floors doesn't add much beyond SqFt")

# 5. Look at specific examples
print("\n5. SPECIFIC EXAMPLES:")
print("-" * 60)

# Find properties with similar sqft but different floors
for sqft_target in [1200, 1500, 1800]:
    similar = df_clean[(df_clean['SqFt'] > sqft_target - 100) & 
                       (df_clean['SqFt'] < sqft_target + 100)][['Address', 'SqFt', 'Floors', 'Price_Clean']]
    if len(similar) > 1:
        print(f"\nProperties around {sqft_target} sqft:")
        by_floors = similar.groupby('Floors')['Price_Clean'].mean()
        for floors, price in by_floors.items():
            count = len(similar[similar['Floors'] == floors])
            print(f"   {floors:.0f} floors: ${price:,.0f} avg (n={count})")

# 6. The real insight
print("\n" + "="*60)
print("THE REAL INSIGHT")
print("="*60)

# Calculate price per sqft by floors
price_per_sqft = df_clean.groupby('Floors').apply(
    lambda x: (x['Price_Clean'] / x['SqFt']).mean()
).round(0)

print("\nPrice per Square Foot by Number of Floors:")
for floors, ppsf in price_per_sqft.items():
    print(f"   {floors:.0f} floors: ${ppsf:.0f}/sqft")

# Vertical vs horizontal space
print("\n🏠 VERTICAL vs HORIZONTAL SPACE:")
single_story = df_clean[df_clean['Floors'] == 1]
multi_story = df_clean[df_clean['Floors'] > 1]

if len(single_story) > 0 and len(multi_story) > 0:
    print(f"Single-story: {len(single_story)} properties, ${single_story['Price_Clean'].mean():,.0f} avg")
    print(f"Multi-story:  {len(multi_story)} properties, ${multi_story['Price_Clean'].mean():,.0f} avg")
    
    # Control for size
    size_ranges = [(1000, 1300), (1300, 1600), (1600, 2000)]
    print("\nControlling for size:")
    for min_size, max_size in size_ranges:
        single = single_story[(single_story['SqFt'] >= min_size) & (single_story['SqFt'] < max_size)]
        multi = multi_story[(multi_story['SqFt'] >= min_size) & (multi_story['SqFt'] < max_size)]
        if len(single) > 0 and len(multi) > 0:
            diff = multi['Price_Clean'].mean() - single['Price_Clean'].mean()
            pct = diff / single['Price_Clean'].mean() * 100
            print(f"   {min_size}-{max_size} sqft: Multi-story premium = ${diff:,.0f} ({pct:+.1f}%)")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"Correlation between SqFt and Floors: {correlation.loc['SqFt', 'Floors']:.3f}")
if abs(correlation.loc['SqFt', 'Floors']) > 0.7:
    print("✓ They ARE measuring the same thing (highly correlated)")
    print("  → More floors = more square footage")
    print("  → It's really just ONE insight: SIZE MATTERS")
else:
    print("✗ They measure DIFFERENT things")
    print("  → Floors adds value beyond just square footage")
    print("  → Buyers prefer vertical layouts even at same size")