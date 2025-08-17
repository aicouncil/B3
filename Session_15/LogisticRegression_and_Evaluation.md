### Machine Learning Model Building and Evaluation

This document provides a detailed explanation of the machine learning model building and evaluation process as demonstrated in the provided Python file. The focus is on a classification problem to predict customer churn (`Exited`) and addresses the critical issue of imbalanced datasets.

-----

#### 1\. Initial Data Preparation for Modeling

Before building a model, the preprocessed data must be split and scaled appropriately.

  * **Feature and Target Selection:** The preprocessed DataFrame `df2` is split into `X` (features or independent variables) and `y` (the target or dependent variable, 'Exited').
    ```python
    X = df2.drop(columns = ['Exited'])
    y = df2['Exited']
    ```
  * **Data Scaling:** **Scaling** is a preprocessing step that standardizes the range of features. For distance-based algorithms like **K-Nearest Neighbors (KNN)**, features with a larger magnitude can disproportionately influence the distance calculation. To prevent this, `MinMaxScaler` is used to scale all features in `X` to a range of 0 to 1.
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)
    ```
  * **Data Splitting:** The scaled features (`Xscaled`) and the target (`y`) are split into training and test sets using `train_test_split`. The training set is used to train the model, while the test set is used to evaluate its performance on unseen data.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(Xscaled,y)
    ```

-----

#### 2\. Model Building and Evaluation (Baseline KNN)

A baseline model is built to provide an initial benchmark for the dataset.

  * **Model Instantiation and Training:** A `KNeighborsClassifier` model (`modelA`) is instantiated and trained using the training data (`xtrain`, `ytrain`).
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    modelA = KNeighborsClassifier()
    modelA.fit(xtrain,ytrain)
    ```
  * **Accuracy Evaluation:** The model's accuracy, which is the proportion of correctly predicted samples, is evaluated on both the training and test data using the `.score()` method.
    ```python
    print("Model accuracy on training data -",modelA.score(xtrain,ytrain))
    print("Model accuracy on test data -",modelA.score(xtest,ytest))
    ```

-----

#### 3\. Handling and Working with Biased Data

The notebook highlights a critical issue: the target variable `y` is highly imbalanced, with a significant majority of customers not exiting (class 0) compared to those who did (class 1).

  * **The Problem:** Simple accuracy is a misleading metric for imbalanced data, as a model can achieve a high score by simply predicting the majority class all the time.
  * **Advanced Evaluation Metrics:** To properly evaluate the model's performance on each class, **`confusion_matrix`** and **`classification_report`** from `sklearn.metrics` are used.
      * The `confusion_matrix` visualizes the number of **True Positives (TP)**, **True Negatives (TN)**, **False Positives (FP)**, and **False Negatives (FN)**.
      * The `classification_report` provides detailed metrics for each class, including **Precision**, **Recall**, and **F1-score**, which are more indicative of the model's bias and true performance.
    <!-- end list -->
    ```python
    from sklearn.metrics import confusion_matrix, classification_report
    ytrainP = modelA.predict(xtrain)
    print(confusion_matrix(ytrain , ytrainP))
    print(classification_report(ytrain , ytrainP))
    ```

#### Techniques to Handle Imbalance:

The file demonstrates three techniques to address this bias:

  * **1. Undersampling:**

      * **Definition:** Undersampling is a technique that reduces the number of samples in the majority class to create a balanced dataset.
      * **Method and Example:** `imblearn.under_sampling.RandomUnderSampler` is used. It randomly removes samples from the majority class (`y=0`) until the number of samples equals that of the minority class (`y=1`).
        ```python
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        Xu,yu = rus.fit_resample(X,y)
        # yu.value_counts() now shows a balanced distribution
        ```
      * **Evaluation:** A new KNN model (`modelB`) is trained on the undersampled data. While overall accuracy might decrease due to data loss, the `classification_report` shows more balanced precision and recall for both classes.

  * **2. Oversampling:**

      * **Definition:** Oversampling increases the number of samples in the minority class to balance the dataset. The **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm generates synthetic data points for the minority class.
      * **Method and Example:** `imblearn.over_sampling.SMOTE` is used to create a balanced dataset (`Xo`, `yo`).
        ```python
        from imblearn.over_sampling import SMOTE
        ros = SMOTE()
        Xo,yo = ros.fit_resample(X,y)
        # yo.value_counts() now shows a balanced distribution
        ```
      * **Evaluation:** A new KNN model (`modelC`) is trained on the oversampled data. This approach avoids the data loss of undersampling and typically leads to a more robust model with a balanced `classification_report`.

  * **3. Class Weight Management (with Logistic Regression):**

      * **Definition:** This method involves assigning a higher penalty (weight) to misclassifications of the minority class during training. It is a model-level approach that does not physically change the dataset.
      * **Method and Example:** The `LogisticRegression` model includes a `class_weight` parameter. Setting `class_weight={0:1 , 1:6}` assigns a weight of 6 to the minority class (`1`) and a weight of 1 to the majority class (`0`), forcing the model to prioritize correctly identifying the churned customers.

-----

#### 4\. Logistic Regression Model in Detail

**Definition:** Logistic Regression is a statistical model for **binary classification**. It calculates a linear combination of features and passes it through a **sigmoid activation function** to output a probability between 0 and 1. This probability is then used to classify an instance into one of two classes.

  * **Model Behavior (Sigmoid Function):** The sigmoid function, $f(z) = \\frac{1}{1 + e^{-z}}$, maps any real number `z` to a value between 0 and 1. The script demonstrates this with small `x` and `w` values.
    ```python
    x = 1
    w0 = 1e-2
    w1 = 1e-2
    z = 1/(1 + np.exp(-(w0 + w1*x)))
    print(z) # Output: ~0.505
    ```
    With larger values, the output approaches 1.
    ```python
    x = 2
    w0 = 4 * 1e-2
    w1 = 4 * 1e-2
    z = 1/(1 + np.exp(-(w0 + w1*x)))
    print(z) # Output: ~0.519
    ```
  * **Performance Evaluation:**
      * A `LogisticRegression` model (`model_log_A`) is trained on the original imbalanced data, and its `classification_report` shows the expected bias towards the majority class.
      * Another model (`model_log_B`) is trained on **oversampled data**, and its `classification_report` shows a significant improvement in identifying the minority class.
      * A third model (`model_log_D`) is trained with **class weights** as described above, and its `classification_report` also reflects a more balanced performance, demonstrating the effectiveness of this technique.
  * **Model Coefficients:** The `model_log_D.coef_` attribute provides the learned weights for each feature, which can be used to interpret the model and understand which features have the most influence on the prediction.
    ```python
    print(model_log_D.coef_)
    ```

### 1\. Logistic Regression on Imbalanced Data

This section of the notebook demonstrates how to build and evaluate a **Logistic Regression** model. The first model (`model_log_A`) is trained on the original, imbalanced dataset to establish a baseline performance.

  * **Data Preparation:** The script re-initializes the features `X` and target `y` from the preprocessed DataFrame `df2` and scales the features using `MinMaxScaler`.
    ```python
    X = df2.drop(columns = ['Exited'])
    y = df2['Exited']

    scaler = MinMaxScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)

    xtrain, xtest, ytrain, ytest = train_test_split(Xscaled,y,stratify=y)
    ```
  * **Model Training and Evaluation:** A `LogisticRegression` model is imported and trained on the scaled training data. Its accuracy is then evaluated on both the training and test sets.
    ```python
    from sklearn.linear_model import LogisticRegression
    model_log_A = LogisticRegression()
    model_log_A.fit(xtrain,ytrain)
    print("Model accuracy on training data -",model_log_A.score(xtrain,ytrain))
    print("Model accuracy on test data -",model_log_A.score(xtest,ytest))
    ```
  * **Performance Analysis:** The `classification_report` is used to reveal the model's performance on a per-class basis. The results would likely show high precision for the majority class (non-churned customers) but low recall for the minority class (churned customers), demonstrating the model's bias towards the more frequent outcome.

-----

### 2\. Logistic Regression with Oversampling (SMOTE)

To address the bias, the script demonstrates retraining the Logistic Regression model using oversampled data created with the SMOTE algorithm.

  * **SMOTE Implementation:** The `imblearn` library's `SMOTE` (Synthetic Minority Over-sampling Technique) is used to generate synthetic data points for the minority class, resulting in a balanced dataset (`Xo`, `yo`).
    ```python
    from imblearn.over_sampling import SMOTE
    ros = SMOTE()
    Xo,yo = ros.fit_resample(X,y)
    ```
  * **Model Training and Evaluation:** A new `LogisticRegression` model (`model_log_B`) is trained on the balanced data. The `classification_report` for this model is expected to show more balanced precision, recall, and f1-score for both classes, indicating a less biased and more robust model.
    ```python
    scaler = MinMaxScaler()
    scaler.fit(Xo)
    Xscaled = scaler.transform(Xo)
    xtrain, xtest, ytrain, ytest = train_test_split(Xscaled,yo,stratify=yo)

    model_log_B = LogisticRegression()
    model_log_B.fit(xtrain,ytrain)

    print(classification_report(ytest, model_log_B.predict(xtest)))
    ```

-----

### 3\. Logistic Regression with Class Weight Management

**Definition:** Logistic Regression is a statistical model for **binary classification**. It establishes a linear relationship between features and the log-odds of the target and uses the **sigmoid activation function** to predict a probability between 0 and 1.

  * **Mechanism (Sigmoid Function):** The sigmoid function, $f(z) = \\frac{1}{1 + e^{-z}}$, is key to logistic regression. It maps any real number input `z` (the linear combination of features) to a value in the range [0, 1], which can be interpreted as a probability.
      * **Example 1**: With small weights and input, the output is a probability close to 0.5.
        ```python
        x = 1; w0 = 1e-2; w1 = 1e-2
        z = 1/(1 + np.exp(-(w0 + w1*x)))
        print(z) # Output: ~0.505
        ```
      * **Example 2**: With larger weights and input, the output probability becomes higher.
        ```python
        x = 2; w0 = 4 * 1e-2; w1 = 4 * 1e-2
        z = 1/(1 + np.exp(-(w0 + w1*x)))
        print(z) # Output: ~0.519
        ```
  * **Class Weight Management:** This is a model-level technique to handle imbalanced data by assigning a higher penalty to misclassifying the minority class. The `LogisticRegression` model's `class_weight` parameter is used for this purpose. The script sets `class_weight={0:1 , 1:6}`, which tells the model to penalize errors on class `1` (the minority class) six times more than errors on class `0` (the majority class).
    ```python
    model_log_D = LogisticRegression(class_weight={0:1 , 1:6})
    model_log_D.fit(xtrain,ytrain)
    ```
  * **Evaluation and Interpretation:** The `classification_report` for `model_log_D` shows that this technique successfully improves **recall for the minority class** (exited customers), demonstrating the model is now better at identifying actual churn cases. The `model_log_D.coef_` attribute provides the learned weights for each feature, which can be analyzed to understand their impact on the prediction.
    ```python
    print(classification_report(ytest, model_log_D.predict(xtest)))
    print(model_log_D.coef_)
    ```
