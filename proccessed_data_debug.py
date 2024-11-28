import pandas as pd
import numpy as np

# Load the data
data_path = "test_processed_data.pkl"
df = pd.read_pickle(data_path)

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
    elif pd.api.types.is_string_dtype(data) or isinstance(data.dtype, pd.CategoricalDtype):
        print(f"  - Unique Values: {data.nunique()}")
        print(f"  - Most Frequent Value: {data.value_counts().idxmax()} ({data.value_counts().max()})")
    else:
        print("  - Data type not recognized for additional analysis.")
    print("=" * 50)


# Debug each variable in the DataFrame
print("=== Debugging Individual Variables ===")
for col in df.columns:
    debug_variable(col, df[col])

# Cross-variable analysis
print("\n=== Cross-Variable Analysis ===")

# Correlation matrix for numerical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns
print("Numerical Variables Correlation Matrix:")
print(df[numerical_cols].corr())
print("=" * 50)

# Distribution of meteorological variables by label
for col in ["temp", "rain", "rhum", "wind", "sw"]:
    if col in df.columns:
        print(f"Distribution of {col} by Label:")
        print(df.groupby("label")[col].describe())
        print("=" * 50)

# Distribution of categorical variables (fuelmod) by label
if "fuelmod" in df.columns:
    print("Distribution of Fuel Model by Label:")
    print(df.groupby("label")["fuelmod"].value_counts())
    print("=" * 50)

# Check for extreme values or anomalies
print("\n=== Extreme Values and Anomalies ===")
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: Found {len(outliers)} outliers outside the range ({lower:.2f}, {upper:.2f})")
    elif col == "fuelmod":
        print(f"Uncommon Fuel Models: {df['fuelmod'].value_counts().tail(5).to_dict()}")
    print("=" * 50)

