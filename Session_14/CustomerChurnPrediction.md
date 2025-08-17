# Customer Churn Prediction: A Classification Project Pipeline

This document provides a detailed explanation of the machine learning project outlined in the `g3_d13_ml(classificationassignment)_(1).py` script. It covers the end-to-end process of building a predictive model for customer churn, including exploratory data analysis (EDA), data preprocessing, handling of imbalanced data, and model evaluation using various classification metrics.

## 1\. Exploratory Data Analysis (EDA)

The first step in any data science project is to understand the dataset's characteristics and identify potential issues.

  * **Data Loading and Initial Inspection:** The script loads the `Churn_Modelling.csv` dataset into a DataFrame and inspects its size (`.shape`), columns, and data types (`.dtypes`).
  * **Dropping Irrelevant Columns:** Columns like `RowNumber`, `CustomerId`, and `Surname` are dropped because they are identifiers and not predictive features.
  * **Univariate Analysis:** The distribution of individual features is visualized using histograms and bar plots to understand data spread and value frequencies.
      * Histograms/Box plots for numerical features: `CreditScore`, `Age`, `Balance`, and `EstimatedSalary`.
      * Bar plots for categorical features: `Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, and `IsActiveMember`.
  * **Outlier Detection:** The Interquartile Range (IQR) method is used to check for outliers in the `Age` column. The script calculates the percentage of data that lies outside the upper and lower fences.
  * **Bivariate Analysis:** This analysis explores the relationship between pairs of variables, particularly how each feature affects the target variable (`Exited`).
      * **Crosstabulation:** `pd.crosstab` is used with a bar plot visualization to show how categorical features like `Geography`, `Gender`, `NumOfProducts`, etc., are related to a customer exiting.
      * **Group-by Aggregation:** The mean of numerical features like `CreditScore`, `Age`, `Balance`, and `EstimatedSalary` is calculated for each `Exited` category (0 for not exited, 1 for exited) to understand which values are more likely to be associated with churn.

## 2\. Data Preprocessing and Feature Engineering

This stage prepares the data for the machine learning model by converting non-numerical features into a numerical format.

  * **Label Encoding for `Gender`:** The `Gender` column, with values "Female" and "Male", is label encoded into a binary format: "Female" is mapped to `0`, and "Male" is mapped to `1`.
    ```python
    df1['Gender']=df1['Gender'].map({"Female":0 , "Male":1})
    ```
  * **One-Hot Encoding for `Geography`:** The `Geography` column, with multiple unique values like "France", "Germany", and "Spain", is one-hot encoded using `pd.get_dummies()`. This creates new binary columns (`Geography_France`, `Geography_Germany`, `Geography_Spain`), where a `1` indicates the presence of that country for a given row.
    ```python
    df2 = pd.get_dummies(df1, columns=['Geography'] , dtype = int)
    ```

## 3\. Baseline Model Training and Evaluation

A baseline model is trained and evaluated to provide a benchmark for performance.

  * **Feature and Target Selection:** The preprocessed DataFrame `df2` is split into `X` (features, all columns except `Exited`) and `y` (target, `Exited` column).
  * **Data Scaling:** The `MinMaxScaler` is used to scale all numerical features in `X` to a range of [0, 1]. This is a necessary step for distance-based algorithms like KNN.
  * **Model Training:** A `KNeighborsClassifier` model (`modelA`) is trained on the scaled training data (`xtrain`, `ytrain`).
  * **Initial Evaluation:** The model's accuracy is evaluated on both the training and test data using the `.score()` method.

## 4\. Handling Imbalanced Data

The dataset is highly imbalanced, as shown by `y.value_counts()` (e.g., 7963 customers did not exit, and 2037 exited). This bias can lead to models that perform well on the majority class but poorly on the minority class.

  * **Evaluation Metrics for Imbalanced Data:** Simple accuracy can be misleading. `confusion_matrix` and `classification_report` are used to provide a more detailed view of the model's performance on each class, showing **precision**, **recall**, and **F1-score**.
  * **Undersampling:**
      * **Definition:** Undersampling is a technique that reduces the number of samples in the majority class to achieve a more balanced distribution.
      * **Implementation:** `RandomUnderSampler` from `imblearn` is used to balance the dataset. After undersampling, the new target `yu` has an equal number of samples for both classes.
      * **Evaluation:** A new KNN model (`modelB`) is trained and evaluated on the undersampled data. The `classification_report` shows improved performance on the minority class (exited customers) compared to the initial model.
  * **Oversampling:**
      * **Definition:** Oversampling is a technique that increases the number of samples in the minority class to balance the dataset. **SMOTE** (Synthetic Minority Over-sampling Technique) is a popular oversampling algorithm that creates new, synthetic data points for the minority class.
      * **Implementation:** The `SMOTE` class from `imblearn.over_sampling` is used to resample the data. The new target `yo` has an equal number of samples for each class.
      * **Evaluation:** A new KNN model (`modelC`) is trained on the oversampled data. The `classification_report` shows a more balanced performance across both classes.
  * **Class Weight Management:**
      * **Definition:** This technique adjusts the model's loss function to penalize misclassifications of the minority class more heavily, encouraging the model to pay more attention to it.
      * **Implementation:** The `class_weight` parameter in `LogisticRegression` is set to `{0:1, 1:6}`. This assigns a weight of 6 to the minority class (exited customers) and a weight of 1 to the majority class.
      * **Evaluation:** A Logistic Regression model (`model_log_D`) is trained with this `class_weight` parameter, and its `classification_report` shows a more balanced recall for the minority class.

## 5\. Other Model Implementations

The script also demonstrates training a **Logistic Regression** model and evaluating its performance on the original data, and then re-evaluating it after oversampling and with class weights. This shows how different models can be used and how preprocessing and bias-handling techniques affect their outcomes.

  * **Logistic Regression:** A `LogisticRegression` model is imported from `sklearn.linear_model`, trained, and its accuracy and `classification_report` are printed for evaluation.
    ```python
    from sklearn.linear_model import LogisticRegression
    model_log_A = LogisticRegression()
    model_log_A.fit(xtrain,ytrain)
    # ... evaluation ...
    ```
