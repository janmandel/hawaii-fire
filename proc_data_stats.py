import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function to print debug information
def debug_variable(name, data):
    print("=" * 50)
    print(f"Variable: {name}")
    if not isinstance(data, pd.Series):
        print(f"  - Error: Expected pandas Series, got {type(data)}")
        return
    print(f"  - Type: {type(data)}")
    print(f"  - Length: {len(data)}")
    missing_count = data.isna().sum()
    print(f"  - Missing Values: {missing_count} ({missing_count / len(data) * 100:.2f}%)")

    if pd.api.types.is_numeric_dtype(data):
        print(f"  - Mean: {data.mean():.2f}, Median: {data.median():.2f}")
        print(f"  - Min: {data.min():.2f}, Max: {data.max():.2f}")
        print(f"  - Std Dev: {data.std():.2f}")
    elif pd.api.types.is_datetime64_any_dtype(data):
        print(f"  - Date Range: {data.min()} to {data.max()}")
    elif pd.api.types.is_string_dtype(data) or pd.api.types.is_categorical_dtype(data):
        print(f"  - Unique Values: {data.nunique()}")
        print(f"  - Most Frequent Value: {data.value_counts().idxmax()} ({data.value_counts().max()})")
    else:
        print("  - Data type not recognized for additional analysis.")
    print("=" * 50)

# Helper function to print category-label distribution
def print_category_label_distribution(df, category_col, label_col):
    if category_col in df.columns and label_col in df.columns:
        print(f"\nDistribution of {label_col} by {category_col}:")
        category_label_distribution = (
            df.groupby([category_col, label_col], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        print(category_label_distribution.to_string())  # Bash-friendly table output

# Load the data
f = 'processed_data.pkl'
df = pd.read_pickle(f)

# Handle empty strings and invalid entries in the date column
if "date" in df.columns:
    print("\nCleaning the 'date' column...")
    # Replace empty strings with NaN
    df["date"] = df["date"].replace("", np.nan)
    # Convert the column to datetime, coercing errors to NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Report missing values after cleaning
    print(f"  - Missing values in 'date' after cleaning: {df['date'].isna().sum()}")

# Clean numeric columns: Convert to numeric and handle invalid/masked values
numeric_columns = ["temp", "rain", "rhum", "wind", "sw"]
for col in numeric_columns:
    if col in df.columns:
        print(f"\nCleaning column: {col}")
        df[col] = df[col].replace("--", np.nan)
        # Replace invalid or masked elements with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Handle masked arrays by replacing them with NaN
        if isinstance(df[col].values, np.ma.MaskedArray):
            df[col] = df[col].values.filled(np.nan)
        # Report missing values
        print(f"  - Missing values after cleaning: {df[col].isna().sum()}")

# Ensure 'fuelmod' is treated as categorical
if "fuelmod" in df.columns:
    print("\nCleaning the 'fuelmod' column...")
    df["fuelmod"] = df["fuelmod"].astype("category")
    print(f"  - Unique categories in 'fuelmod': {df['fuelmod'].cat.categories.tolist()}")
    print(f"  - Missing values in 'fuelmod': {df['fuelmod'].isna().sum()}")

# Debug individual variables
print("\n=== Debugging Individual Variables ===")
for col in df.columns:
    debug_variable(col, df[col])

# Perform grouped analysis by label for cleaned numeric columns
print("\n=== Grouped Analysis of Meteorological Variables by Label ===")
for col in numeric_columns:
    if col in df.columns:
        print(f"\n{col}:")
        grouped = df.groupby("label")[col]
        try:
            print(grouped.describe())  # Show statistics for each group
        except Exception as e:
            print(f"  Error in grouped analysis for {col}: {e}")

# Cross-variable analysis: Correlation matrix for numerical variables
print("\n=== Cross-Variable Analysis ===")
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    print("Numerical Variables Correlation Matrix:")
    print(df[numerical_cols].corr())
else:
    print("No numerical columns found for correlation analysis.")

print("=" * 50)

# Distribution of fuelmod labels
print_category_label_distribution(df, "fuelmod", "label")


