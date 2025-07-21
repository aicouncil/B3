# Comprehensive Data Analysis Techniques

## 1. Statistics Fundamentals

### 1.1 Measures of Central Tendency

**Mean**: Arithmetic average of values  
Formula: `sum(values)/count(values)`  
Example: For `marks = [3, 47, 33, 30, 44]`  
Calculation: `(3+47+33+30+44)/5 = 31.4`

**Median**: Middle value in ordered data  
Example: Sorted marks `[3, 30, 33, 44, 47]`  
Median: `33` (for odd count)  
For even counts: Average of two middle values

**Mode**: Most frequent value(s)  
Example: In `[30, 30, 33, 44]`  
Mode: `30`

### 1.2 Measures of Dispersion

**Variance**: Average squared deviation from mean  
Formula: `sum((x - mean)²)/n`  
Example: For marks with mean=31.4  
Variance = `((3-31.4)² + (47-31.4)² + ...)/5 ≈ 209.74`

**Standard Deviation**: Square root of variance  
`√209.74 ≈ 14.48`

### 1.3 Outlier Detection (IQR Method)

1. Sort data: `[3, 3, 5, ..., 47, 99]`
2. Calculate quartiles:
   - Q1 (25th %ile): `18.25`
   - Q3 (75th %ile): `42.5`
3. Compute IQR: `42.5 - 18.25 = 24.25`
4. Determine fences:
   - Lower: `18.25 - 1.5*24.25 = -18.125`
   - Upper: `42.5 + 1.5*24.25 = 78.875`
5. Identify outliers: `99 > 78.875` → Outlier

## 2. Pandas Data Manipulation

### 2.1 Data Handling Basics

```python
import pandas as pd
df = pd.read_csv('titanic.csv')  # Load dataset
df.head(3)  # Preview first 3 rows
```

### 2.2 Missing Value Management
```
df.isna().sum()  # Count missing values per column
# Output: Age(177), Cabin(687), Embarked(2)
```

#### Detection:
```
df.isna().sum()  # Count missing values per column
# Output: Age(177), Cabin(687), Embarked(2)
```
#### Imputation:
```
# Numerical columns
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Mean imputation
df['Age'].fillna(df['Age'].median(), inplace=True)  # Robust alternative
```
```
# Categorical columns
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

#### Dropping Data:
```
df.dropna(subset=['Age'])  # Remove rows with missing Age
df.drop(columns=['Cabin'])  # Remove entire Cabin column
```
### 2.3 Feature Engineering

## Creating Age Groups:
```
conditions = [
    (df['Age'] <= 16),
    (df['Age'] <= 32),
    (df['Age'] <= 48),
    (df['Age'] <= 64),
    (df['Age'] > 64)
]
choices = [0, 1, 2, 3, 4]
df['Age_band'] = np.select(conditions, choices)
```
### 2.4 Categorical Data Encoding

#### One-Hot Encoding:
```
pd.get_dummies(
    df, 
    columns=['Sex', 'Embarked'], 
    drop_first=True  # Avoid dummy variable trap
)
```
### 2.5 Advanced Outlier Handling
```
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Filtering outliers
clean_df = df[
    (df['Fare'] >= Q1 - 1.5*IQR) & 
    (df['Fare'] <= Q3 + 1.5*IQR)
]
```
## 3. Practical Applications
# Titanic Dataset Workflow:

    Load data → 2. Impute Age → 3. Drop Cabin → 4. Encode Sex/Embarked → 5. Remove Fare outliers

# Big Mart Sales Example:
```
# Convert categorical columns
cat_cols = ['Item_Fat_Content', 'Outlet_Type']
df[cat_cols] = df[cat_cols].astype('category')

# One-hot encoding
pd.get_dummies(df, columns=cat_cols)
```

# Key Takeaways

    ## Statistical Foundations are crucial for data interpretation

    ## Pandas provides:

        Efficient missing value handling

        Flexible data transformation

        Built-in statistical operations

    Best Practices:

        Always explore data before processing

        Choose imputation methods based on data distribution

        Handle categorical data before modeling

Note: All code examples are executable in a Python environment with Pandas/Numpy installed.
