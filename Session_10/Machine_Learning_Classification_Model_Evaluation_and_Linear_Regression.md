# Machine Learning: Classification Model Evaluation and Linear Regression

This document provides a detailed explanation of key machine learning concepts and implementations using Python, drawing from `g3_d9_ml2.py` (Classification Model Evaluation) and `g3_d9_ml3(regression).py` (Linear Regression). It covers data preprocessing, model training, evaluation metrics, and cross-validation.

## Part 1: Classification Model Evaluation (from `g3_d9_ml2.py`)

This section demonstrates building and evaluating a K-Nearest Neighbors (KNN) classification model using the Wine dataset, emphasizing data scaling and cross-validation techniques.

### 1.1. Data Loading and Preparation

* **Load Wine Dataset:** The Wine dataset, containing various chemical properties and a 'Target' column indicating wine class, is loaded into a Pandas DataFrame.
    ```python
    import pandas as pd
    import numpy as np
    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/wine.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/wine.csv)')
    df.head()
    ```
    The 'Target' column contains unique values [0, 1, 2], representing different wine classes.
* **Feature and Target Separation:**
    * `X`: Features or independent variables, which are all columns except 'Target'.
    * `y`: The target or dependent variable, which is the 'Target' column.
    ```python
    X = df.drop(columns = ['Target'])
    y = df['Target']
    X.head()
    ```

### 1.2. Data Scaling (MinMaxScaler)

**Definition:** Data scaling is a preprocessing step where numerical feature values are transformed to a standard range. `MinMaxScaler` scales features to a given range, typically [0, 1], by transforming `x` to `(x - min(x)) / (max(x) - min(x))`.

* **Purpose/Benefit:** Scaling is crucial for distance-based algorithms like KNN because they are sensitive to the magnitude and units of features. Features with larger ranges might dominate the distance calculation, leading to biased results. Scaling ensures all features contribute proportionally.
* **Process:**
    1.  An instance of `MinMaxScaler` is created.
    2.  The `fit()` method calculates the minimum and maximum values for each feature in `X`.
    3.  The `transform()` method applies the scaling transformation to `X` using the calculated min/max values, resulting in `Xscaled` where all feature values range from 0 to 1.
    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    Xscaled = scaler.transform(X)
    Xscaled # Output: All data columns scaled to magnitude 0 to 1
    ```

### 1.3. Data Splitting for Training and Testing

* **Purpose:** To evaluate the model's performance on unseen data, the dataset is split into training and testing sets. The training set is used to train the model, and the test set is used to evaluate its generalization ability.
* **Method:** `train_test_split` from `sklearn.model_selection` is used. `random_state` ensures reproducibility of the split.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(Xscaled,y,random_state=2)
    ```

### 1.4. K-Nearest Neighbors (KNN) Model Training

* **Algorithm:** `KNeighborsClassifier` is imported from `sklearn.neighbors`.
* **Model Definition:** `modelA` is initialized with `n_neighbors=7`, meaning it will consider the 7 nearest data points for classification.
* **Model Training:** The `fit()` method trains the model using the `xtrain` (scaled features) and `ytrain` (target labels).
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    modelA = KNeighborsClassifier(n_neighbors=7)
    modelA.fit(xtrain,ytrain)
    ```

### 1.5. Model Evaluation

* **Predictions:**
    * `ytrainP`: Predictions made by `modelA` on the training data (`xtrain`).
    * `ytestP`: Predictions made by `modelA` on the test data (`xtest`).
    ```python
    ytrainP = modelA.predict(xtrain)
    ytestP = modelA.predict(xtest)
    ```
* **Accuracy Calculation:** Accuracy is calculated as the proportion of correctly predicted samples. It's evaluated for both training and test data.
    ```python
    print((ytrain == ytrainP).sum()/len(xtrain)) # Training accuracy
    print((ytest == ytestP).sum()/len(xtest))   # Test accuracy
    ```
* **Prediction for New Data:** The trained model can predict the class of a new, unseen data point (features for a new wine sample).
    ```python
    wine_feature = [13.24, 2.59, 2.87, 21.0, 118, 2.80, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735]
    print(modelA.predict([wine_feature]))
    ```

### 1.6. Cross-Validation (K-Fold Cross-Validation)

**Definition:** K-fold cross-validation is a technique to estimate the performance of a machine learning model more robustly. The dataset is split into `K` equal-sized folds. The model is trained `K` times, with each fold serving as the test set exactly once, and the remaining `K-1` folds as the training set.

* **Purpose/Benefit:**
    * **Robust Performance Estimation:** Provides a more reliable estimate of model performance on unseen data by averaging results over multiple train/test splits, reducing the impact of a single "lucky" or "unlucky" split.
    * **Better Hyperparameter Tuning:** Helps in selecting optimal hyperparameters by providing a less biased estimate of how different parameters affect generalization.
    * **Better Use of Data:** All data points eventually serve as both training and testing data.
* **Process (10-fold example):**
    The script demonstrates a 10-fold cross-validation by looping 10 times. In each iteration, `train_test_split` is called with a different `random_state` (controlled by `k` in the loop) to generate a new train-test split. A new `KNeighborsClassifier` model (`modelB`) is trained on each split, and its accuracy on both training and test sets is appended to respective lists (`train_accuracy`, `test_accuracy`).
    ```python
    train_accuracy = []
    test_accuracy = []

    for k in range(0,10): # Looping 10 times for 10 folds
      xtrain,xtest,ytrain,ytest = train_test_split(Xscaled,y,random_state=k)
      modelB = KNeighborsClassifier()
      modelB.fit(xtrain,ytrain)
      train_accuracy.append(modelB.score(xtrain,ytrain))
      test_accuracy.append(modelB.score(xtest,ytest))
    ```
* **Average Performance:** Finally, the average training and test accuracies across all 10 folds are calculated and printed using `np.mean()`.
    ```python
    print("Average train accuracy (10-fold)-" , np.mean(train_accuracy))
    print("Average test accuracy (10-fold)-" , np.mean(test_accuracy))
    ```
    * *Interpretation*: If the average test accuracy is high and stable across folds, it indicates a robust model. A large gap between average train and test accuracy might suggest overfitting.

## Part 2: Linear Regression (from `g3_d9_ml3(regression).py`)

This section delves into Linear Regression, a fundamental technique for predicting continuous numerical values, covering both manual implementation and usage with Scikit-learn, including simple and multiple linear regression.

### 2.1. Introduction to Regression

**Definition:** Regression in machine learning refers to predictive modeling techniques that aim to establish a relationship between a dependent variable (target) and one or more independent variables (features) to predict continuous numerical outcomes.

* **Simple Dataset:** A basic dataset with one feature `x` and one target `y` is used for demonstration.
    ```python
    x = np.array([3,6,4,9,2])   # Feature
    y = np.array([7,8,6,10,5])  # Target
    ```
* **Visualization:** A scatter plot helps visualize the relationship between `x` and `y`.
    ```python
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.show()
    ```

### 2.2. Linear Regression: Manual Implementation (Gradient Descent Concept)

This part illustrates the core mechanics of linear regression by manually implementing the gradient descent optimization algorithm. The goal is to find the "best possible linear relationship" ($y_h = w_0 + w_1 * x$) between features and target by minimizing error.

* **Model Parameters:**
    * `w0` (intercept): The value of `y` when `x` is 0.
    * `w1` (slope): The change in `y` for a one-unit change in `x`.
    * Initial values are set for `w0` and `w1` (e.g., `w0=1`, `w1=1.5`).
* **Prediction (`yh`):** The predicted target values (`yh`) are calculated using the current `w0`, `w1`, and `x`.
    ```python
    w0 = 1
    w1 = 1.5
    yh = w0 + w1*x # Initial prediction
    print(yh)
    ```
* **Error Calculation (Mean Squared Error - MSE):**
    **Definition:** MSE is a common loss function for regression, calculated as the mean of the squared differences between the actual target values (`y`) and the predicted values (`yh`). Squaring the error ensures positive values and penalizes larger errors more heavily.
    ```python
    error = ((y - yh)**2).mean()
    print(error)
    ```
* **Gradient Calculation (`dew0`, `dew1`):**
    These represent the partial derivatives of the MSE loss function with respect to `w0` and `w1`, respectively. They indicate the direction and magnitude to adjust `w0` and `w1` to reduce the error.
    ```python
    dew0 = -2*(y-yh).mean()
    dew1 = -2*((y-yh)*x).mean()
    print(dew0, dew1)
    ```
* **Weight Update (Gradient Descent Step):**
    **Definition:** Gradient Descent is an iterative optimization algorithm used to minimize a function (the loss function in ML). It works by taking steps proportional to the negative of the gradient of the function at the current point.
    * `lr` (learning rate): A small positive value that determines the step size at each iteration. A smaller `lr` leads to slower but potentially more precise convergence.
    * The weights `w0` and `w1` are updated by subtracting the product of `lr` and their respective gradients.
    ```python
    lr = 0.01
    w0 = w0 - lr * dew0
    w1 = w1 - lr * dew1
    print(w0 , w1)
    ```
* **Training Loop:** The entire process of prediction, error calculation, gradient calculation, and weight update is repeated for a fixed number of `epochs` (e.g., 500). In each epoch, the error is printed, showing its gradual reduction as the model learns.
    ```python
    for i in range(500): # Iterates 500 times
      yh = w0 + w1*x
      error = ((y - yh)**2).mean()
      dew0 = -2*(y-yh).mean()
      dew1 = -2*((y-yh)*x).mean()
      lr = 0.01
      w0 = w0 - lr * dew0
      w1 = w1 - lr * dew1
      print(error) # Shows error decreasing per epoch
    ```
* **Final Parameters and Predictions:** After the loop, the optimized `w0` and `w1` are obtained, and final predictions `yh` are made.
    ```python
    print(w0 , w1)
    yh = w0 + w1 * x
    print(yh)
    ```
* **Mean Absolute Error (MAE):**
    **Definition:** MAE is another common metric for regression, representing the average of the absolute differences between actual and predicted values. It gives a more intuitive measure of average error compared to MSE, as it's in the same unit as the target variable.
    ```python
    print("Mean absolute error" , abs(y - yh).mean())
    ```
* **Visualization of Regression Line:** The final step is to plot the original scatter points and overlay the learned regression line (`yh` vs. `x`) to visually assess the fit.
    ```python
    plt.scatter(x,y)
    plt.plot(x,yh,'r') # Plots the regression line in red
    plt.show()
    ```
* **Prediction for New Data:** The learned `w0` and `w1` can be used to predict `y` for a new `x_new` value.
    ```python
    x_new = 5
    y_new = w0 + w1 * x_new
    print(y_new)
    ```
* **Benefits (Manual Implementation):** Provides a deep understanding of how linear regression models learn and how optimization algorithms like gradient descent work.
* **Limitations (Manual Implementation):** Extremely tedious and prone to errors for larger datasets or more complex models (e.g., multiple features), making it impractical for real-world scenarios.

### 2.3. Linear Regression using Scikit-Learn

Scikit-learn provides a much simpler and efficient way to implement linear regression.

* **Data Preparation:** Features `x` and target `y` are reshaped to 2D arrays, which is a requirement for Scikit-learn's model fitting methods when dealing with single features.
    ```python
    x = np.array([3,6,4,9,2]).reshape(5,1)
    y = np.array([7,8,6,10,5]).reshape(5,1)
    ```
* **Import and Define Model:** `LinearRegression` algorithm is imported and an instance `modelA` is created.
    ```python
    from sklearn.linear_model import LinearRegression
    modelA = LinearRegression()
    ```
* **Training:** The `fit()` method automatically performs the optimization (finding `w0` and `w1`).
    ```python
    modelA.fit(x,y)
    ```
* **Trained Parameters:** The learned `w1` (coefficients) and `w0` (intercept) can be accessed directly.
    ```python
    print("w1",modelA.coef_)
    print("w0",modelA.intercept_)
    ```
* **Predictions:** The `predict()` method is used to get predictions.
    ```python
    yp = modelA.predict(x)
    print(yp)
    ```
* **Error (MAE):** MAE is calculated to assess model performance.
    ```python
    error = abs(y - yp).mean()
    print(error)
    ```
* **Prediction for New Data:** Predicting `y` for a new `x_new` value is straightforward.
    ```python
    x_new = 5
    y_new = modelA.predict([[x_new]])
    print(y_new)
    ```
* **Benefits (Scikit-learn):** Simplicity of implementation, efficiency for large datasets, and robustness due to optimized algorithms.

### 2.4. Multiple Linear Regression (Advertising Dataset)

Multiple linear regression extends simple linear regression to involve two or more independent variables (features) to predict a single continuous dependent variable.

* **Load Advertising Dataset:** The dataset contains advertising costs (TV, radio, newspaper) and corresponding sales.
    ```python
    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv)', index_col=0)
    df.head()
    ```
* **Simple Linear Regression Example (TV vs. Sales):**
    * `X` is 'TV' expense, `y` is 'sales'.
    * Data is split into train/test sets.
    * A `LinearRegression` model (`modelB`) is trained.
    * Mean Absolute Error (MAE) is calculated for both training and test data to evaluate performance.
        ```python
        X = df[['TV']]
        y = df['sales']
        from sklearn.model_selection import train_test_split
        xtrain, xtest , ytrain, ytest = train_test_split(X,y,random_state=0)
        from sklearn.linear_model import LinearRegression
        modelB = LinearRegression()
        modelB.fit(xtrain,ytrain)
        ytrainP = modelB.predict(xtrain)
        ytestP = modelB.predict(xtest)
        mae_train = abs(ytrain - ytrainP).mean()
        mae_test = abs(ytest - ytestP).mean()
        print(mae_train, mae_test)
        ```
    * Sales are predicted for a new TV expense.
        ```python
        tv_expense = [[34.5]]
        predicted_sales = modelB.predict(tv_expense)
        print(predicted_sales)
        ```
* **Multiple Linear Regression Setup (TV, Radio vs. Sales):**
    The script sets up the features `X` to include both 'TV' and 'radio' expenses to predict 'sales'. This is the starting point for building a multiple linear regression model. The subsequent steps would involve splitting the data, training a `LinearRegression` model, finding coefficients for both features, and evaluating its performance similar to the simple linear regression example.
    ```python
    X = df[['TV', 'radio']]
    y = df['sales']
    ```
* **Benefits (Multiple Linear Regression):** Can capture more complex relationships and potentially provide more accurate predictions by incorporating the influence of multiple predictor variables simultaneously.
* **Limitations (General Linear Regression):**
    * **Assumes Linearity:** Assumes a linear relationship between features and the target. If the relationship is non-linear, a linear model may perform poorly.
    * **Sensitivity to Outliers:** Outliers can significantly affect the learned regression line.
    * **Multicollinearity:** In multiple linear regression, if independent variables are highly correlated, it can lead to unstable coefficient estimates and make interpretation difficult.
    * **Homoscedasticity:** Assumes the variance of the errors is constant across all levels of the independent variables.
    * **Normality of Residuals:** Assumes the errors (residuals) are normally distributed.
