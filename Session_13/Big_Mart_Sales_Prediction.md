# Big Mart Sales Prediction: A Comprehensive Machine Learning Pipeline

This document provides a detailed explanation of a complete machine learning project for predicting Big Mart sales, encompassing data cleaning, feature engineering, model training, evaluation, interpretation, and preparation for deployment. The content is derived from the `g3_d12_ml6(big_mart_sales)(1).py` script.

## 1. Project Goal and Initial Data Loading

* **Goal:** The primary objective is to build a predictive model to estimate `Item_Outlet_Sales` based on various product and outlet characteristics.
* **Data Loading:** The `Big_Mart_Sales_Figure.csv` dataset is loaded into a Pandas DataFrame.
* **Initial Inspection:** The script starts by inspecting the data's shape, data types, unique values, and the count of missing values (`isna().sum()`) per column to identify data quality issues.

## 2. Data Cleaning and Preprocessing

A crucial step is to handle inconsistencies and missing values to prepare the data for the model.

* **`Item_Fat_Content` Standardization:** Inconsistent categorical entries like 'Low Fat', 'low fat', and 'LF' are all standardized to 'low_fat'. Similarly, 'Regular' and 'reg' are standardized to 'regular'.
    ```python
    def fix_fat_content(fat_content):
      if fat_content in ['Low Fat' , 'low fat' , 'LF']:
        return 'low_fat'
      else:
        return 'regular'
    df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(fix_fat_content)
    df['Item_Fat_Content'].unique()
    ```
* **`Item_Identifier` Feature Engineering:** The first two characters of `Item_Identifier` (e.g., 'FD', 'DR', 'NC') are extracted to create a new, potentially more relevant categorical feature that groups items by type.
    ```python
    df['Item_Identifier'] = df['Item_Identifier'].apply(lambda x : x[0:2])
    ```
* **Imputing Missing `Outlet_Size`:**
    * The `Outlet_Size` column contains missing values.
    * A crosstabulation between `Outlet_Size` and `Outlet_Location_Type` is used to reveal a relationship: 'Tier 1' and 'Tier 2' locations correlate with 'Small' and 'Medium' outlets, respectively, while 'Tier 3' locations have 'Medium' and 'High' outlets.
    * The missing `Outlet_Size` values are imputed based on the corresponding `Outlet_Location_Type`, which is a more informed approach than simply using the mode.
    ```python
    os_olt = df['Outlet_Location_Type'].apply(lambda olt : 'Small' if olt in ['Tier 1', 'Tier 2'] else 'Medium')
    df['Outlet_Size'] = df['Outlet_Size'].fillna(os_olt)
    ```
* **Imputing Missing `Item_Weight`:**
    * Missing values in the `Item_Weight` column are filled with the mean of the column.
    ```python
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    ```
* **Final Check for Missing Values:** `df.isna().sum()` confirms that all missing values have been handled.

## 3. Feature Engineering and Encoding for Machine Learning

Features are transformed and encoded into a format suitable for a linear regression model.

* **Feature Engineering for `Outlet_Identifier` and `Outlet_Establishment_Year`:**
    * The last two characters of `Outlet_Identifier` (e.g., 'OUT049' -> '49') are extracted and converted to an integer.
    * The `Outlet_Establishment_Year` is converted into a feature representing the outlet's age by subtracting it from a reference year (2025).
* **Label Encoding for Ordinal Data:** Categorical columns with an implied order are mapped to numerical integers.
    * `Item_Fat_Content`: 'low_fat' -> `0`, 'regular' -> `1`.
    * `Outlet_Size`: 'Small' -> `0`, 'Medium' -> `1`, 'High' -> `2`.
    * `Outlet_Location_Type`: 'Tier 1' -> `1`, 'Tier 2' -> `2`, 'Tier 3' -> `3`.
* **One-Hot Encoding for Nominal Data:** `pd.get_dummies()` is used to convert the remaining nominal categorical features (e.g., `Item_Identifier`, `Item_Type`, `Outlet_Type`) into numerical, binary columns.
* **Target Transformation:** The `Item_Outlet_Sales` (target variable) is log-transformed using `np.log()`. This is done because its original distribution is likely skewed, and log transformation helps linear models by normalizing the distribution and stabilizing variance.

## 4. Model Training, Evaluation, and Interpretation

A `LinearRegression` model is built to predict the sales values.

* **Data Splitting and Scaling:**
    * The prepared features (`X`) are scaled to a [0, 1] range using `MinMaxScaler`.
    * The scaled features (`Xt`) and log-transformed target (`yt`) are split into training and test sets.
* **Model Training:** A `LinearRegression` model (`modelA`) is trained on the training data.
* **Evaluation Metrics:** The model's performance is evaluated using:
    * **Mean Absolute Error (MAE):** The average magnitude of prediction errors on both training and test data.
    * **R-squared Score:** Measures the proportion of variance in the target that is predictable from the features. Both a manual calculation and `sklearn.metrics.r2_score` are shown.
    * **Adjusted R-squared:** A metric that adjusts R-squared for the number of predictors, penalizing the model for irrelevant features. A calculation example is provided, showing the formula: $1 - (1 - R^2) \times \frac{n-1}{n-p-1}$.
* **Model Interpretation:**
    * **Feature Coefficients:** The coefficients of the `LinearRegression` model are extracted and visualized in a bar plot to show the impact of each feature on sales prediction.
    * **Correlation Heatmap:** A heatmap of the correlation matrix is generated using `seaborn` to visualize multicollinearity among features.
* **Multicollinearity Check using VIF:**
    * **Definition:** The Variance Inflation Factor (VIF) is a measure to detect multicollinearity, where high correlation among predictors can lead to unstable regression coefficients.
    * **Implementation:** The script uses `statsmodels` to calculate VIF for a subset of features after adding a constant term (intercept).
    ```python
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    Xs = sm.add_constant(X_vif) # X_vif is a subset of features
    vif_data['VIF'] = [variance_inflation_factor(Xs.values , i) for i in range(len(Xs.columns))]
    ```

## 5. Model Deployment and Prediction

The final steps involve saving the trained model and scaler and demonstrating how to reuse them to make a prediction on a new data point.

* **Saving Artifacts:** The trained model (`modelB`) is saved as `big_mart.pkl` and the scaler (`scaler`) as `data_scaler.pkl` using `joblib.dump()`. This ensures the model and preprocessing steps can be reused consistently.
* **Loading Artifacts:** The saved model and scaler are loaded back into memory.
* **Prediction on New Input:**
    * A new data point with raw values is defined.
    * An input array is manually constructed with the correct size (38 zeros).
    * The raw input is encoded using the same logic and lambda functions from the training phase.
    * One-hot encoded features are handled by finding their index and setting the value to 1.
    * The prepared input array is scaled using the loaded `_scaler` and then used for prediction with the loaded `_model`.
    * The log-transformed prediction is converted back to the original sales scale using `np.exp()` for the final result.
