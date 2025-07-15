# Pandas Data Analysis: Step-by-Step Guide

This guide walks you through the fundamental steps of data analysis using pandas, a powerful Python library. We'll use the Titanic dataset as an example, covering each concept in detail with examples and explanations.

---

## 1. Importing Libraries

Before starting any data analysis project, it's crucial to import the necessary libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib.pyplot**: For data visualization.

---

## 2. Loading the Dataset

Use pandas to load a CSV dataset.

```python
df = pd.read_csv('titanic.csv')
```

- `pd.read_csv()` reads a CSV file into a DataFrame.
- `df.head()` displays the first 5 rows:

```python
df.head()
```

**Example Output:**
| PassengerId | Survived | Pclass | Name            | Sex | Age | ... |
|-------------|----------|--------|-----------------|-----|-----|-----|
| 1           | 0        | 3      | Braund, Mr. Owen| male| 22  | ... |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley| female| 38 | ... |

---

## 3. Checking for Missing Values

Missing data can skew your analysis, so it's important to identify null values.

```python
df.isna()
```

- Returns a DataFrame of Boolean values (`True` if missing).

To count missing values in each column:

```python
df.isna().sum()
```

**Example Output:**
```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age             177
Cabin           687
Embarked         2
dtype: int64
```
**Explanation:**  
- The column `Age` has 177 missing values.
- The column `Cabin` has 687 missing values.

---

## 4. Handling Missing Data

There are several strategies for dealing with missing data:
- **Remove rows/columns with missing values**
- **Impute missing values with mean, median, or mode**

**Removing rows with missing values:**

```python
df_clean = df.dropna()
```

**Imputing missing values with the mean (for numeric columns):**

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
```

**Imputing missing values with the mode (for categorical columns):**

```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

---

## 5. Data Summary and Exploration

Get a quick summary of your dataset.

```python
df.describe()
```

**Example Output:**
|       | PassengerId | Survived | Pclass | Age | ... |
|-------|-------------|----------|--------|-----|-----|
| count | 891         | 891      | 891    | 714 | ... |
| mean  | 446.0       | 0.38     | 2.31   | 29.7| ... |
| std   | 257.4       | 0.49     | 0.83   | 14.5| ... |

---

## 6. Data Visualization

Visualize features to uncover patterns.

**Histogram of Age:**

```python
df['Age'].hist(bins=30)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.title('Distribution of Passenger Ages')
plt.show()
```

**Bar plot of Survival Rate by Class:**

```python
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.show()
```

---

## 7. Conclusion

By following these steps, you can efficiently load, inspect, clean, and visualize your data using pandas. Understanding how to handle missing data and explore your dataset is foundational for any data analysis workflow.

---

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic)
