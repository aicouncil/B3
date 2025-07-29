# Linear and Non-Linear Regression: From Fundamentals to Advanced Implementations

This document provides a comprehensive explanation of linear and non-linear regression techniques in machine learning, covering manual implementations, Scikit-learn usage, model evaluation, feature importance, and polynomial regression. The content is derived from `g3_d9_ml3(regression).py` and `g3_d10_ml4(regression).py`.

## Part 1: Linear Regression Fundamentals (from `g3_d9_ml3(regression).py`)

This section introduces the core concepts of linear regression, starting with a manual implementation to build foundational understanding before moving to Scikit-learn's efficient tools.

### 1.1. Introduction to Regression

**Definition:** Regression in machine learning is a type of supervised learning focused on predicting a **continuous numerical value** (the target or dependent variable) based on one or more independent variables (features). It aims to establish a mathematical relationship, typically linear, between inputs and outputs.

* **Simple Dataset:** A basic dataset with a single feature `x` and a corresponding target `y` is used for illustration.
    ```python
    import numpy as np
    x = np.array([3,6,4,9,2])   # Feature
    y = np.array([7,8,6,10,5])  # Target
    print("x:", x) # Output: x: [3 6 4 9 2]
    print("y:", y) # Output: y: [ 7  8  6 10  5]
    ```
* **Visualization:** A scatter plot is used to visualize the linear relationship between `x` and `y`.
    ```python
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.show()
    ```

### 1.2. Linear Regression: Manual Implementation (Gradient Descent Concept)

This section demonstrates the underlying mathematical process of linear regression, particularly how its parameters (weights) are optimized using Gradient Descent. The goal is to "Establish a best possible linear relationship between given sets of features and target". The linear relationship is represented as $y_h = w_0 + w_1 \cdot x$, where $y_h$ is the hypothesized (predicted) value, $w_0$ is the intercept, and $w_1$ is the slope.

* **Initial Model Parameters:** `w0` (intercept) and `w1` (slope) are initialized with arbitrary values.
    ```python
    w0 = 1    # Initial intercept
    w1 = 1.5  # Initial slope
    ```
* **Initial Prediction (`yh`):** The predicted values are calculated using the initial `w0`, `w1`, and the feature `x`.
    ```python
    yh = w0 + w1*x
    print(yh) # Output: [8.5 10.0 7.0 14.5 4.0]
    ```
* **Error Calculation (Mean Squared Error - MSE):**
    **Definition:** Mean Squared Error (MSE) is a common metric used as a loss function in regression. It measures the average of the squares of the errors, where an error is the difference between the actual observed value and the predicted value.
    $$ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - y_{h_i})^2 $$
    ```python
    error = ((y - yh)**2).mean()
    print(error) # Output: 2.5
    ```
* **Gradient Calculation (`dew0`, `dew1`):**
    These are the partial derivatives of the MSE loss function with respect to $w_0$ and $w_1$. They indicate the direction and magnitude by which $w_0$ and $w_1$ should be adjusted to reduce the error.
    * `dew0`: Gradient for $w_0$.
    * `dew1`: Gradient for $w_1$.
    ```python
    dew0 = -2*(y-yh).mean()
    print("dew0:", dew0) # Output: dew0: -1.0
    dew1 = -2*((y-yh)*x).mean()
    print("dew1:", dew1) # Output: dew1: -8.0
    ```
* **Weight Update (Single Gradient Descent Step):**
    **Definition:** Gradient Descent is an iterative optimization algorithm used to find the local minimum of a function (in this case, the MSE loss function). It repeatedly adjusts model parameters (weights) in the direction of the steepest descent of the loss function.
    * `lr` (learning rate): A hyperparameter that controls the step size at each iteration. A small `lr` leads to slow convergence, while a large `lr` might overshoot the minimum.
    ```python
    lr = 0.01
    w0 = w0 - lr * dew0
    w1 = w1 - lr * dew1
    print(w0 , w1) # Output: 1.01 1.58
    ```
* **Training Loop (Multiple Epochs):** The prediction, error calculation, gradient calculation, and weight update steps are repeated iteratively for a fixed number of `epochs` (e.g., 500). This continuous adjustment of $w_0$ and $w_1$ gradually minimizes the `error`.
    ```python
    w0 = 1
    w1 = 1.5
    for i in range(500):
      yh = w0 + w1*x
      error = ((y - yh)**2).mean()
      dew0 = -2*(y-yh).mean()
      dew1 = -2*((y-yh)*x).mean()
      lr = 0.01
      w0 = w0 - lr * dew0
      w1 = w1 - lr * dew1
      print(error) # Shows error decreasing per epoch
    ```
* **Final Parameters and Predictions:** After 500 epochs, the optimized `w0` and `w1` values are obtained, and final predictions `yh` are made.
    ```python
    print(w0 , w1) # Output: 4.051948051948052 0.6558441558441558
    yh = w0 + w1 * x
    print(yh) # Output: [6.01753247 7.97012987 6.67337662 9.92571429 5.36168831]
    ```
* **Mean Absolute Error (MAE):**
    **Definition:** Mean Absolute Error (MAE) calculates the average of the absolute differences between predictions and actual values. Unlike MSE, MAE is in the same units as the target variable, making it more interpretable.
    ```python
    print("Mean absolute error" , abs(y - yh).mean()) # Output: Mean absolute error 0.6597402597402598
    ```
* **Visualization of Regression Line:** The learned linear relationship is plotted on the scatter plot to visually assess the model's fit to the data.
    ```python
    plt.scatter(x,y)
    plt.plot(x,yh,'r') # Plots the regression line in red
    plt.show()
    ```
* **Prediction for New Data:** The optimized `w0` and `w1` can be used to predict the target for a new, unseen feature value.
    ```python
    x_new = 5
    y_new = w0 + w1 * x_new
    print(y_new) # Output: 7.331168831168831
    ```
* **Benefits (Manual Implementation):** Provides a fundamental understanding of how linear regression and gradient descent work at a low level.
* **Limitations (Manual Implementation):** Highly tedious, inefficient, and impractical for real-world datasets with many features or large numbers of data points.

### 1.3. Linear Regression using Scikit-Learn

Scikit-learn provides a robust and efficient `LinearRegression` class that automates the complex calculations of manual implementation.

* **Data Preparation:** For `sklearn`, feature arrays typically need to be 2D (`.reshape(-1,1)` for a single feature).
    ```python
    x = np.array([3,6,4,9,2]).reshape(5,1)   # Feature (reshaped to 2D)
    y = np.array([7,8,6,10,5]).reshape(5,1)  # Target (reshaped to 2D)
    ```
* **Import and Model Definition:** The `LinearRegression` algorithm is imported from `sklearn.linear_model`, and a model instance `modelA` is created.
    ```python
    from sklearn.linear_model import LinearRegression
    modelA = LinearRegression()
    ```
* **Model Training:** The `fit()` method is called with the features `x` and target `y`. This trains the model, automatically finding the optimal `w0` and `w1`.
    ```python
    modelA.fit(x,y)
    ```
* **Trained Parameters:** The learned slope (`w1`) and intercept (`w0`) can be accessed directly from the `modelA` object.
    ```python
    print("w1",modelA.coef_)     # Output: w1 [[0.65584416]]
    print("w0",modelA.intercept_) # Output: w0 [4.05194805]
    ```
* **Predictions:** The `predict()` method uses the trained model to make predictions on the input features.
    ```python
    yp = modelA.predict(x)
    print(yp) # Output: [[6.01753247] [7.97012987] [6.67337662] [9.92571429] [5.36168831]]
    ```
* **Error (MAE):** MAE is calculated to quantify the model's prediction accuracy.
    ```python
    error = abs(y - yp).mean()
    print(error) # Output: 0.6597402597402598
    ```
* **Prediction for New Data:** Predictions for new, unseen data points are made by simply passing them to the `predict()` method.
    ```python
    x_new = 5
    y_new = modelA.predict([[x_new]]) # Note: input needs to be 2D array [[value]]
    print(y_new) # Output: [[7.33116883]]
    ```
* **Benefits (Scikit-learn):** Simplicity of code, efficiency for large datasets, robustness due to optimized algorithms, and wide applicability to various regression tasks.

## Part 2: Advanced Linear Regression and Feature Importance (from `g3_d10_ml4(regression).py`)

This section extends linear regression to multiple features and explores methods for understanding feature importance.

### 2.1. Introduction to Multiple Linear Regression

Multiple linear regression is an extension of simple linear regression where a target variable is predicted based on **two or more independent variables (features)**.

* **Load Advertising Dataset:** A dataset containing advertising expenditure on TV, radio, and newspaper, along with corresponding sales figures, is loaded. This dataset is suitable for predicting sales based on advertising costs.
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Advertising.csv)', index_col=0)
    df.head()
    ```

### 2.2. Simple Linear Regression (TV vs. Sales) - as a Baseline

Before implementing multiple linear regression, a simple linear regression model using only 'TV' advertising cost to predict 'sales' is built and evaluated as a baseline for comparison.

* **Feature and Target Selection:** `X` is the 'TV' column, and `y` is the 'sales' column.
    ```python
    X = df[['TV']]
    y = df['sales']
    ```
* **Model Training and Evaluation:** The data is split, a `LinearRegression` model (`modelB`) is trained, and its performance is evaluated using Mean Absolute Error (MAE) on both training and test sets.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest , ytrain, ytest = train_test_split(X,y,random_state=0)
    from sklearn.linear_model import LinearRegression
    modelB = LinearRegression()
    modelB.fit(xtrain,ytrain)
    ytrainP = modelB.predict(xtrain)
    ytestP = modelB.predict(xtest)
    mae_train = abs(ytrain - ytrainP).mean()
    mae_test = abs(ytest - ytestP).mean()
    print("MAE Train:", mae_train) # Example: 1.218...
    print("MAE Test:", mae_test)   # Example: 1.134...
    ```
* **Prediction Example:** Predicts sales for a new 'TV' expense.
    ```python
    tv_expense = [[34.5]]
    predicted_sales = modelB.predict(tv_expense)
    print("Predicted Sales for TV expense 34.5:", predicted_sales)
    ```

### 2.3. Multiple Linear Regression (TV & Radio vs. Sales)

This part builds a multiple linear regression model using 'TV' and 'radio' advertising costs to predict 'sales'.

* **Feature and Target Selection:** `X` now includes both 'TV' and 'radio' columns.
    ```python
    X = df[['TV', 'radio']]
    y = df['sales']
    ```
* **Model Training and Evaluation:** Data is split, a `LinearRegression` model (`modelA` in this file) is trained, and its coefficients and intercept are retrieved. MAE is used to evaluate performance.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size=0.75)
    from sklearn.linear_model import LinearRegression
    modelA = LinearRegression()
    modelA.fit(xtrain,ytrain)
    print("Coefficients:", modelA.coef_)    # Output: [0.046... 0.180...] (w1, w2 for TV, radio)
    print("Intercept:", modelA.intercept_)  # Output: 3.023... (w0)
    ```
* **Manual Prediction Example (using coefficients):** Demonstrates how sales can be predicted manually using the obtained coefficients and intercept.
    ```python
    tv_expense = 34.5
    radio_expense = 55.7
    # Calculated based on coefficients and intercept
    print(3.023 + 0.046 * tv_expense + 0.180 * radio_expense)
    ```
* **Scikit-learn Prediction Example:** Uses the `predict()` method of `modelA` for prediction.
    ```python
    print(modelA.predict([[tv_expense, radio_expense]]))
    ```
* **Evaluation:** MAE is calculated for both training and test sets.
    ```python
    ytrainP = modelA.predict(xtrain)
    ytestP = modelA.predict(xtest)
    maeTrain = abs(ytrain - ytrainP).mean()
    print("Mean Absolute Error, Train", maeTrain) # Example: 1.144...
    maeTest = abs(ytest - ytestP).mean()
    print("Mean Absolute Error, Test", maeTest)   # Example: 1.258...
    ```

### 2.4. Multiple Linear Regression (TV, Radio & Newspaper vs. Sales)

This section further extends the model to include all three advertising features: 'TV', 'radio', and 'newspaper' to predict 'sales'.

* **Feature and Target Selection:** `X` now includes 'TV', 'radio', and 'newspaper'.
    ```python
    X = df[['TV', 'radio' , 'newspaper']]
    y = df['sales']
    ```
* **Model Training and Evaluation:** A new `LinearRegression` model (`modelB` in this file) is trained, and its performance is evaluated using MAE on training and test sets.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size=0.75)
    from sklearn.linear_model import LinearRegression
    modelB = LinearRegression()
    modelB.fit(xtrain,ytrain)
    ytrainP = modelB.predict(xtrain)
    ytestP = modelB.predict(xtest)
    maeTrain = abs(ytrain - ytrainP).mean()
    print("Mean Absolute Error, Train", maeTrain) # Example: 1.127...
    maeTest = abs(ytest - ytestP).mean()
    print("Mean Absolute Error, Test", maeTest)   # Example: 1.341...
    ```
* **Coefficients:** The coefficients for each feature (TV, radio, newspaper) are displayed.
    ```python
    print(modelB.coef_) # Output: [0.046... 0.188... -0.001...]
    ```
* **Conclusion from File:** Based on the MAE values, the script suggests that the "Model build using only TV and radio expense is good here", implying that adding 'newspaper' expense might not significantly improve (or even slightly worsen) the model's test performance.

### 2.5. Feature Importance Analysis

To understand which features are most influential, methods like pairplots and correlation matrices are used.

* **Pairplot:**
    **Definition:** A pairplot (from Seaborn) plots pairwise relationships between variables in a dataset. It's useful for visualizing distributions of individual variables and relationships between pairs, helping to identify potential correlations or patterns.
    ```python
    import seaborn as sns
    sns.pairplot(df)
    ```
* **Correlation Matrix:**
    **Definition:** A correlation matrix displays the correlation coefficients between different variables in a dataset. The Pearson correlation coefficient ranges from -1 to +1, where +1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.
    ```python
    df.corr() # Displays the correlation matrix for all numerical columns in df
    ```
* **Manual Correlation Calculation:** The script demonstrates how to mathematically calculate the Pearson correlation coefficient between 'TV' expense and 'sales'.
    ```python
    x1 = df['TV']
    y = df['sales']
    n1 = ((x1 - x1.mean()) * (y - y.mean())).sum()
    d1 = (((x1 - x1.mean())**2).sum() * ((y - y.mean())**2).sum())**0.5
    c1 = n1/d1
    print(c1) # Output: 0.782... (strong positive correlation between TV and sales)
    ```
* **Use Cases (Feature Importance):**
    * **Feature Selection:** Identifying and removing irrelevant or redundant features to simplify models and improve performance.
    * **Model Interpretability:** Understanding which factors contribute most to the target variable's prediction.

## Part 3: Non-Linear Regression (from `g3_d10_ml4(regression).py`)

This section explores how to model non-linear relationships using linear regression by transforming features.

### 3.1. Introduction to Non-Linear Regression

Sometimes, the relationship between features and the target is not linear. Non-linear regression aims to model such relationships. Polynomial regression is a common approach to achieve this.

* **Simple Non-Linear Dataset:** A basic dataset with `x` and `y` where the relationship might not be perfectly linear is re-introduced.
    ```python
    x = np.array([3,6,4,9,2]).reshape(5,1)   # Feature
    y = np.array([7,8,6,10,5])      # Target
    plt.scatter(x,y)
    plt.show()
    ```

### 3.2. Polynomial Features Transformation

**Definition:** `PolynomialFeatures` is a Scikit-learn preprocessing class that generates polynomial and interaction features. For a feature `x`, it can create $x^2, x^3$, etc., enabling a linear model to fit a non-linear curve.

* **Manual Polynomial Transformation:** Demonstrates creating higher-order polynomial features (`x^2`, `x^3`) manually and stacking them with the original feature `x`.
    ```python
    xn = np.hstack((x , x**2 , x**3)) # Creates [x, x^2, x^3] for each data point
    print(xn)
    ```
* **Scikit-learn `PolynomialFeatures`:** A more automated way to generate polynomial features.
    * `degree=3`: Creates polynomial features up to the 3rd degree (e.g., $x, x^2, x^3$).
    * `include_bias=False`: Excludes the bias (intercept) term, as `LinearRegression` will add its own intercept.
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    pol = PolynomialFeatures(degree = 3, include_bias=False)
    pol.fit(X) # X = df[['TV']] from Advertising dataset
    Xt = pol.transform(X) # Transforms 'TV' feature into [TV, TV^2, TV^3]
    print(Xt)
    ```

### 3.3. Non-Linear Regression Model Training and Evaluation

After transforming the features into polynomial terms, a standard `LinearRegression` model can be trained on these transformed features.

* **Model Training on Transformed Data:** A `LinearRegression` model (`model_nl_A` or `model_nl_B`) is trained using the polynomial features (`xn` or `Xt`).
    ```python
    from sklearn.linear_model import LinearRegression
    model_nl_A = LinearRegression()
    model_nl_A.fit(xn, y) # For simple x,y data
    print("Coefficients (model_nl_A):", model_nl_A.coef_) # Output: [w1, w2, w3] for x, x^2, x^3
    ```
    ```python
    # For Advertising dataset's TV feature transformed
    xtrain, xtest, ytrain, ytest = train_test_split(Xt,y)
    model_nl_B = LinearRegression()
    model_nl_B.fit(xtrain , ytrain)
    print("Coefficients (model_nl_B):", model_nl_B.coef_)
    ```
* **Prediction for Visualization:** To visualize the non-linear curve, a new set of sequential data points (`xt` or `Xp`) spanning the range of the original feature is created, transformed into polynomial features, and then used for prediction.
    ```python
    xt = np.linspace(1,10,5).reshape(5,1) # Example for simple x,y data
    xt = np.hstack((xt, xt**2 , xt**3)) # Transform
    yn = model_nl_A.predict(xt) # Predict
    ```
    ```python
    Xp = np.linspace(0,300,100).reshape(100,1) # Example for Advertising TV
    Xps = np.hstack((Xp, Xp**2 , Xp**3)) # Transform
    Yps = model_nl_B.predict(Xps) # Predict
    ```
* **Visualization of Non-Linear Curve:** The original scatter plot is overlaid with the predicted non-linear regression curve.
    ```python
    plt.scatter(x,y) # For simple x,y data
    plt.plot(xt[:,0:1] , yn) # Plotting curve
    plt.show()
    ```
    ```python
    plt.scatter(xtrain[:,0:1], ytrain) # For Advertising TV data
    plt.plot(Xp, Yps, 'r') # Plotting curve
    plt.show()
    ```
* **Evaluation:** MAE is used to evaluate the performance of the non-linear model on training and test data.
    ```python
    ytrainP = model_nl_B.predict(xtrain)
    ytestP = model_nl_B.predict(xtest)
    mae_train = abs(ytrain - ytrainP).mean()
    mae_test = abs(ytest - ytestP).mean()
    print("MAE Train:", mae_train, "MAE Test:", mae_test)
    ```

### 3.4. Unimplemented Task

The file concludes with a task to implement and train a non-linear regression model using two features (TV and radio expenses) to predict sales, implying the application of `PolynomialFeatures` on multiple input columns.

* **Limitations (General Linear Regression):**
    * **Assumes Linearity:** Basic linear regression assumes a linear relationship; non-linear relationships require feature transformations (e.g., polynomial features).
    * **Sensitivity to Outliers:** Outliers can significantly skew the regression line or curve.
    * **Multicollinearity:** In multiple linear regression, high correlation between independent variables can lead to unstable and uninterpretable coefficients.
    * **Homoscedasticity:** Assumes the variance of residuals is constant across all levels of predictors.
    * **Normality of Residuals:** Assumes that the errors (residuals) are normally distributed.
