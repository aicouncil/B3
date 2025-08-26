### 1\. Data Preprocessing and Missing Value Handling

This section of the notebook focuses on loading the raw data, inspecting it for quality issues, and performing various preprocessing steps, including imputation and feature engineering.

  * **Data Loading and Initial Inspection**: The script loads the `Health Insurance Lead Prediction Raw Data.csv` dataset into a pandas DataFrame. It then inspects the data's dimensions (`df.shape`) and the distribution of the target variable (`Response`).

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv('https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Health%20Insurance%20Lead%20Prediction%20Raw%20Data.csv')
    df.head()
    ```

  * **Dropping Irrelevant Features**: The `ID` column is dropped from the DataFrame as it is a unique identifier and has no predictive power.

    ```python
    df1 = df.drop(columns = ['ID'])
    ```

  * **Handling Missing Values**: The script identifies that several columns, namely `Health Indicator`, `Holding_Policy_Duration`, and `Holding_Policy_Type`, have a significant number of missing values.

      * **`Health Indicator`**: Missing values in this categorical column (e.g., 'X1', 'X2', etc.) are imputed with a new category, 'X0', to represent the missing information explicitly.
        ```python
        df1['Health Indicator'] = df1['Health Indicator'].fillna('X0')
        ```
      * **Informed Imputation for `Holding_Policy_Duration` and `Holding_Policy_Type`**: The file demonstrates a more nuanced imputation strategy for these columns. By using `pd.crosstab`, it analyzes the relationship between these columns and other categorical features like `Accomodation_Type` and `Is_Spouse`. The logic is to impute missing values based on the most frequent value for a given related feature.
        ```python
        # Imputation logic based on Accommodation_Type
        df1['Holding_Policy_Duration'] = df1['Holding_Policy_Duration'].fillna(df1['Accomodation_Type'].apply(lambda x : 1 if x == 'Rented' else 14))
        df1.head()
        ```
      * **KNNImputer**: The script also shows an advanced imputation method using `KNNImputer`. This technique finds the K-nearest neighbors for a data point with a missing value and imputes it based on the values of those neighbors.
        ```python
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df2 = imputer.fit_transform(df1)
        ```

  * **Feature Engineering and Encoding**: The script converts several categorical and identifier columns into numerical formats suitable for machine learning models.

      * `City_Code` and `Health Indicator`: The prefixes 'C' and 'X' are removed, and the columns are cast to an integer data type.
      * `Accomodation_Type`, `Reco_Insurance_Type`, `Is_Spouse`: These binary categorical columns are mapped to integers `0` and `1`.
        ```python
        df1['City_Code'] = df1['City_Code'].str.replace('C','').astype(int)
        df1['Accomodation_Type'] = df1['Accomodation_Type'].map({'Rented':0, 'Owned':1})
        df1['Reco_Insurance_Type'] = df1['Reco_Insurance_Type'].map({'Individual':0, 'Joint':1})
        df1['Is_Spouse'] = df1['Is_Spouse'].map({'No':0, 'Yes':1})
        df1['Health Indicator'] = df1['Health Indicator'].str.replace('X','').astype(int)
        ```

After these steps, the entire DataFrame `df3` is converted to a numerical format with no missing values.

-----

### 2\. Model Building and Evaluation

This section demonstrates training and evaluating two classification models: **Support Vector Machine (SVM)** and **Naive Bayes**.

  * **Data Splitting and Scaling**: The oversampled data from the `SMOTE` algorithm (`Xr`, `yr`) is used to train the models. The features are scaled using `MinMaxScaler` to ensure uniform feature ranges. The data is split into training and test sets.
    ```python
    from imblearn.over_sampling import SMOTE
    ros = SMOTE()
    Xr,yr = ros.fit_resample(X,y)
    xtrain, xtest, ytrain, ytest = train_test_split(Xr,yr)
    scaler = MinMaxScaler()
    xtrainScaled = scaler.fit_transform(xtrain)
    xtestScaled = scaler.transform(xtest)
    ```
  * **Support Vector Machine (SVM) Classifier**:
      * **Definition**: SVM is a powerful classification algorithm that finds the optimal hyperplane to separate data points of different classes. The `kernel` parameter determines the function used to map the input data into a higher-dimensional space where separation might be easier.
      * **Training and Parameters**: The `SVC` model is initialized with a `poly` (polynomial) kernel and a `class_weight` parameter to handle class imbalance. The model is then trained on the scaled training data.
        ```python
        from sklearn.svm import SVC
        model_svm = SVC(kernel = 'poly' , class_weight={0:1 , 1:1.1})
        model_svm.fit(xtrainScaled,ytrain)
        ```
      * **Evaluation**: The `classification_report` is used to evaluate the SVM model's performance on both training and test data, providing metrics like precision, recall, and F1-score for each class.
  * **Naive Bayes Classifier**:
      * **Definition**: Naive Bayes is a probabilistic classifier based on Bayes' theorem. It makes a "naive" assumption that all features are independent of each other, given the class. `GaussianNB` is a specific variant that assumes features follow a normal distribution.
      * **Training**: A `GaussianNB` model is instantiated and trained on the scaled training data (`xtrainScaled`, `ytrain`).
        ```python
        from sklearn.naive_bayes import GaussianNB
        model_nb = GaussianNB()
        model_nb.fit(xtrainScaled, ytrain)
        ```
      * **Evaluation**: The `classification_report` is used to evaluate the Naive Bayes model on both training and test data.
