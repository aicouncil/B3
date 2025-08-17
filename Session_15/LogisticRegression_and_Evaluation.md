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
