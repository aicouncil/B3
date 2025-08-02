````markdown
# Regression Modeling Techniques and Big Mart Sales Prediction Project

This document provides a detailed explanation of non-linear regression using polynomial features and feature interactions, alongside a comprehensive machine learning pipeline for predicting Big Mart sales. The content is derived from `g3_d12_ml5(nonlinearregression).py` and `g3_d12_ml6(big_mart_sales).py`.

## Part 1: Non-Linear Regression using Polynomial Features and Feature Interactions (from `g3_d12_ml5(nonlinearregression).py`)

This section explores how to model non-linear relationships using linear regression by transforming features, focusing on polynomial and interaction terms.

### 1.1. Introduction and Data Preparation

* **Goal:** Implement and train a non-linear regression model using two features (TV & radio advertisement costs) to predict sales, and evaluate its performance.
* **Data Loading:** The `Advertising.csv` dataset is loaded, which contains advertising costs and sales figures.
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv)', index_col=0)
    df.head()
    ```
* **Feature and Target Selection:** `X` consists of 'TV' and 'radio' features, and `y` is the 'sales' target.
    ```python
    X = df[['TV' , 'radio']]
    y = df['sales']
    ```
* **Data Splitting:** The data is split into training and test sets using `train_test_split`.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y)
    ```

### 1.2. Non-Linear Regression using Full Polynomial Features

This approach creates all polynomial combinations of the input features up to a specified degree, allowing a linear model to capture non-linear relationships.

* **Concept (`PolynomialFeatures`):** `PolynomialFeatures(degree=2, include_bias=False)` is used. For two features ($x_1, x_2$), `degree=2` generates new features including $x_1, x_2, x_1^2, x_2^2, x_1x_2$. `include_bias=False` prevents adding a constant bias term, as the `LinearRegression` model will add its own.
* **Transformation:** The `fit()` method learns the polynomial features, and `transform()` applies the transformation to both training and test sets.
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    pol = PolynomialFeatures(degree = 2 , include_bias=False)
    pol.fit(xtrain) # Learns polynomial features from training data
    xtrainpol = pol.transform(xtrain) # Transforms training features
    xtestpol = pol.transform(xtest)   # Transforms test features
    ```
* **Model Training:** A `LinearRegression` model (`modelA`) is trained on the transformed polynomial features (`xtrainpol`).
    ```python
    from sklearn.linear_model import LinearRegression
    modelA = LinearRegression()
    modelA.fit(xtrainpol , ytrain)
    ```
* **Retrieve Coefficients:** The coefficients of the trained model for each polynomial feature are displayed.
    ```python
    print(modelA.coef_) # Output: Coefficients for TV, radio, TV^2, radio^2, TV*radio
    ```
* **Model Evaluation:** Mean Absolute Error (MAE) and R-squared score are used to evaluate the model's performance on both training and test data.
    ```python
    ytrainP = modelA.predict(xtrainpol)
    ytestP = modelA.predict(xtestpol)
    maeTrain = abs(ytrain - ytrainP).mean()
    maeTest = abs(ytest - ytestP).mean()
    print("Mean absolute error (Train)-" , maeTrain)
    print("Mean absolute error (Test)-" , maeTest)

    from sklearn.metrics import r2_score
    print("Using full non linear transformed data -")
    print("R2_Score_train" , r2_score(ytrain , ytrainP))
    print("R2_Score_test" , r2_score(ytest , ytestP))
    ```
    *Benefits*: This approach allows linear models to capture complex, non-linear relationships.
    *Limitations*: Can lead to a high number of features (high dimensionality) if `degree` is large, potentially causing overfitting.

### 1.3. Non-Linear Regression using Feature Interaction Only

This approach focuses specifically on creating interaction terms between features without generating individual polynomial terms (e.g., $x_1^2, x_2^2$).

* **Concept (`PolynomialFeatures` with `interaction_only=True`):** `PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)` is used. For two features ($x_1, x_2$), `degree=2` with `interaction_only=True` will only generate the $x_1x_2$ term (and original $x_1, x_2$), but not $x_1^2$ or $x_2^2$.
* **Transformation and Model Training:** Similar to the full polynomial approach, the interaction features are learned and transformed, and a `LinearRegression` model (`modelB`) is trained on these features.
    ```python
    pol = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    pol.fit(xtrain)
    xtrain_ft = pol.transform(xtrain)
    xtest_ft = pol.transform(xtest)

    from sklearn.linear_model import LinearRegression
    modelB = LinearRegression()
    modelB.fit(xtrain_ft , ytrain)
    ```
* **Retrieve Coefficients:** The coefficients for the trained model are displayed.
    ```python
    print(modelB.coef_) # Output: Coefficients for TV, radio, and TV*radio interaction
    ```
* **Model Evaluation:** MAE and R-squared score are used to evaluate performance on training and test sets.
    ```python
    ytrainP = modelB.predict(xtrain_ft)
    ytestP = modelB.predict(xtest_ft)
    maeTrain_ft = abs(ytrain - ytrainP).mean()
    maeTest_ft = abs(ytest - ytestP).mean()
    print("Mean absolute error (Train)-" , maeTrain_ft)
    print("Mean absolute error (Test)-" , maeTest_ft)

    from sklearn.metrics import r2_score
    print("Using only feature Interaction -")
    print("R2_Score_train" , r2_score(ytrain , ytrainP))
    print("R2_Score_test" , r2_score(ytest , ytestP))
    ```
    *Benefits*: Can capture interactions between features without adding excessive individual polynomial terms, potentially reducing overfitting compared to full polynomial features when only interactions are relevant.

## Part 2: Big Mart Sales Prediction Project (from `g3_d12_ml6(big_mart_sales).py`)

This section details a comprehensive machine learning pipeline for predicting `Item_Outlet_Sales`, including extensive data cleaning, preprocessing, feature engineering, and model evaluation.

### 2.1. Project Goal and Initial Data Loading

* **Goal:** Build and evaluate a model to predict `Item_Outlet_Sales`.
* **Data Loading:** The `Big_Mart_Sales_Figure.csv` dataset is loaded into a Pandas DataFrame.
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Big_Mart_Sales_Figure.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Big_Mart_Sales_Figure.csv)')
    df.head()
    ```
* **Initial Data Inspection:** Common DataFrame methods are used to understand the data's basic characteristics, including shape, data types, unique values, and the count of missing values per column.
    ```python
    df.shape
    df.dtypes
    df.nunique()
    df.isna().sum()
    ```

### 2.2. Data Cleaning and Preprocessing

A crucial step to ensure data quality and prepare features for modeling. A copy (`df1 = df.copy()`) is made for modifications.

* **`Item_Fat_Content` Standardization:** Inconsistent entries like 'Low Fat', 'low fat', 'LF' are standardized to 'low_fat', and 'Regular', 'reg' to 'regular'.
    ```python
    df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(lambda fat_content: 'low_fat' if fat_content in ['Low Fat', 'low fat', 'LF'] else 'regular')
    df['Item_Fat_Content'].unique()
    ```
* **`Item_Identifier` Feature Engineering:** The first two characters of `Item_Identifier` (e.g., 'FD', 'DR', 'NC') are extracted to create a new, potentially more informative categorical feature.
    ```python
    df['Item_Identifier'] = df['Item_Identifier'].apply(lambda x: x[0:2])
    df.head() # Shows updated Item_Identifier
    ```
* **Missing Value Imputation for `Outlet_Size`:**
    * The `Outlet_Size` column has missing values.
    * A crosstabulation between `Outlet_Size` and `Outlet_Location_Type` is used to identify a pattern: 'Tier 1' and 'Tier 2' locations primarily have 'Small' or 'Medium' outlets, while 'Tier 3' locations have 'Medium' or 'High'.
    * Missing `Outlet_Size` values are imputed based on `Outlet_Location_Type`: 'Small' if location is 'Tier 1' or 'Tier 2', otherwise 'Medium' (for 'Tier 3'). This is a more informed imputation than just using the global mode.
    ```python
    pd.crosstab(df['Outlet_Size'] , df['Outlet_Location_Type']) # To understand relationships
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Location_Type'].apply(lambda olt : 'Small' if olt in ['Tier 1', 'Tier 2'] else 'Medium'))
    ```
* **Missing Value Imputation for `Item_Weight`:**
    * `Item_Weight` also has missing values.
    * Missing `Item_Weight` values are filled with the mean of the `Item_Weight` column.
    ```python
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    ```
* **Final Missing Value Check:** `df.isna().sum()` is run again to confirm all missing values have been handled.

### 2.3. Feature Engineering and Encoding for Machine Learning

Features are transformed and encoded to be suitable for the machine learning model.

* **Identify Non-Numerical Columns:** Lists all columns that are not of numerical data type.
    ```python
    non_num_coulumns = df1.select_dtypes(exclude='number').columns.to_list()
    # Output: ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    ```
* **Label Encoding for Ordinal Data:**
    * `Item_Fat_Content`: Mapped to `0` for 'low_fat' and `1` for 'regular'.
    * `Outlet_Size`: Mapped to `0` for 'Small', `1` for 'Medium', and `2` for 'High' (assuming an inherent order).
    * `Outlet_Location_Type`: Mapped to `1` for 'Tier 1', `2` for 'Tier 2', and `3` for 'Tier 3' (assuming an inherent order).
    ```python
    df1['Item_Fat_Content'] = df1['Item_Fat_Content'].map({'low_fat':0, 'regular':1})
    df1['Outlet_Size'] = df1['Outlet_Size'].map({'Small':0,'Medium':1, 'High':2 })
    df1['Outlet_Location_Type'] = df1['Outlet_Location_Type'].map({'Tier 1':1, 'Tier 2':2 , 'Tier 3':3})
    df1.head(2)
    ```
* **`Outlet_Identifier` Feature Engineering:** The last two characters of `Outlet_Identifier` are extracted (e.g., 'OUT049' becomes '49'). This converts the identifier into a numerical string, which will later be one-hot encoded along with other nominal features.
    ```python
    df1['Outlet_Identifier'] = df1['Outlet_Identifier'].apply(lambda x :x[-2:])
    df1.head(3)
    ```
* **One-Hot Encoding for Nominal Data:** `pd.get_dummies()` is applied to `df1` (which now has several numerical features and remaining categorical features). This converts all remaining categorical columns (like `Item_Identifier`, `Item_Type`, `Outlet_Type`, and the transformed `Outlet_Identifier`) into binary (0 or 1) columns. `dtype=int` ensures integer output for dummy variables.
    ```python
    df2 = pd.get_dummies(df1, dtype=int)
    df2.head()
    ```
* **`Outlet_Establishment_Year` Feature Engineering:** The year is transformed into "age" of the outlet by subtracting it from a reference year (e.g., 2025). This might provide a more relevant feature for the model than the raw year.
    ```python
    df2['Outlet_Establishment_Year'] = 2025 - df2['Outlet_Establishment_Year']
    df2.head()
    ```
* **Target Transformation (`Item_Outlet_Sales`):**
    * A histogram of `df['Item_Outlet_Sales']` is plotted, likely showing a skewed distribution.
    * The `Item_Outlet_Sales` (target variable) is log-transformed using `np.log()`. This often helps linear models perform better when the target variable is skewed, by normalizing its distribution and stabilizing variance.
    ```python
    df['Item_Outlet_Sales'].plot.hist() # Original distribution
    np.log(df['Item_Outlet_Sales']).plot.hist() # Log-transformed distribution
    yt = np.log(df['Item_Outlet_Sales']) # The transformed target used for model training
    ```

### 2.4. Model Training and Evaluation

A `LinearRegression` model is built and evaluated to predict the log-transformed sales.

* **Feature and Target Selection:** `X` contains all processed features from `df2` (excluding `Item_Outlet_Sales`), and `y` is the original `Item_Outlet_Sales` (before log transformation for `yt`).
    ```python
    X = df2.drop(columns = ['Item_Outlet_Sales']) # Features
    y = df2['Item_Outlet_Sales'] # Original target
    ```
* **Feature Scaling:** All features in `X` are scaled to a [0, 1] range using `MinMaxScaler`.
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    Xt = scaler.fit_transform(X) # Scaled features
    ```
* **Target for Training:** `yt` (the log-transformed sales) is used as the target for model training.
    ```python
    yt = np.log(df['Item_Outlet_Sales']) # Log-transformed target
    ```
* **Data Splitting:** The scaled features (`Xt`) and log-transformed target (`yt`) are split into training and test sets.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(Xt,yt)
    ```
* **Model Training:** A `LinearRegression` model (`modelA`) is initialized and trained on the `xtrain` and `ytrain` data.
    ```python
    from sklearn.linear_model import LinearRegression
    modelA = LinearRegression()
    modelA.fit(xtrain,ytrain)
    ```
* **Model Evaluation:**
    * **Predictions:** `ytrainP` and `ytestP` are predictions on the training and test sets, respectively.
    * **Mean Absolute Error (MAE):** Calculated for both training and test sets. It quantifies the average magnitude of errors in predictions.
        ```python
        maeTrain = abs(ytrain - ytrainP).mean()
        maeTest = abs(ytest - ytestP).mean()
        print(f"Mean absolute error (Train) - {maeTrain}")
        print(f"Mean absolute error (Test) - {maeTest}")
        ```
    * **R-squared Score:** Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. Higher values indicate a better fit. Both a manual calculation and `sklearn.metrics.r2_score` are shown.
        ```python
        # Manual R2 score calculation
        n1 = ((ytrain - ytrainP)**2).sum()
        d1 = ((ytrain - ytrain.mean())**2).sum()
        r2_score_manual = (1- (n1/d1))
        print(r2_score_manual)

        # Scikit-learn R2 score
        from sklearn.metrics import r2_score
        print("R2_Score_train" , r2_score(ytrain , ytrainP))
        print("R2_Score_test" , r2_score(ytest , ytestP))
        ```

### 2.5. Model Interpretation and Future Concepts

* **Model Coefficients Analysis:** The coefficients of the trained `LinearRegression` model indicate the strength and direction of the relationship between each feature and the target variable. These are displayed as a Pandas DataFrame and visualized using a bar plot.
    ```python
    model_coef = pd.DataFrame(modelA.coef_)
    model_coef.index = X.columns
    model_coef.plot.bar(figsize = (12,5))
    ```
* **Correlation Heatmap:** A heatmap of the correlation matrix for all features is generated using Seaborn. This helps visualize multicollinearity and relationships between features.
    ```python
    import seaborn as sns
    plt.figure(figsize = (25,25))
    sns.heatmap(df2.corr() , annot =True , cmap = 'RdYlGn' , fmt = '.2f')
    plt.show()
    ```
* **Future Concepts Mentioned:** The script briefly mentions advanced concepts like "Adjusted r-squared," "Variance inflation factor (VIF)" (for multicollinearity assessment), and "Flask api" (for model deployment).
````
