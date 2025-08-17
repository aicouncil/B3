### Machine Learning Model Building and Evaluation

This document provides a detailed explanation of the machine learning model building process, from data splitting and initial model training to advanced hyperparameter tuning and ensemble methods. The content is based on the `g3_d16_classification(decisiontreeclassifier).py` script, which aims to build a loan approval prediction model.

-----

#### 1\. Data Preparation for Modeling

The script assumes that data cleaning and preprocessing have already been completed. This section focuses on preparing the data for the machine learning model.

  * **Feature and Target Selection:** The preprocessed DataFrame `df2` is split into features (`X`) and the target variable (`y`, which is `Loan_Status`). This is a standard step in supervised learning.
    ```python
    X = df2.drop(columns = 'Loan_Status')
    y = df2['Loan_Status']
    ```
  * **Handling Imbalanced Data:** The script identifies that the target variable `y` is imbalanced (more loan approvals than rejections). To address this bias, **Oversampling** is performed using **SMOTE** (Synthetic Minority Over-sampling Technique) from `imblearn`. This technique creates synthetic data points for the minority class to balance the dataset.
    ```python
    from imblearn.over_sampling import SMOTE
    ros = SMOTE()
    Xr,yr = ros.fit_resample(X,y)
    # The output of yr.value_counts() shows a balanced distribution
    ```
  * **Data Splitting:** The oversampled features (`Xr`) and target (`yr`) are split into training and test sets using `train_test_split`.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(Xr,yr)
    ```

-----

#### 2\. Decision Tree Classifier

A **Decision Tree** is a supervised machine learning algorithm that uses a tree-like model of decisions and their possible consequences. It splits the data based on feature values to create a hierarchical set of rules.

  * **Model Parameters:** The model is initialized with several hyperparameters to control its behavior and prevent overfitting.
      * `criterion = 'entropy'`: The function to measure the quality of a split. `entropy` measures the impurity or randomness of the data, and the model seeks to minimize this.
      * `max_depth = 3`: Limits the depth of the tree to prevent it from becoming too complex and overfitting the training data.
      * `class_weight = {0:2 , 1:1}`: Assigns a weight to each class. This is another way to handle imbalanced data, by penalizing misclassifications of one class more heavily than the other.
      * `max_leaf_nodes = 8`: Limits the total number of leaf nodes, which also helps control the tree's complexity.
  * **Model Training and Evaluation:** The model is trained using the training data, and its performance is evaluated using the `score()` method on both the training and test sets.
    ```python
    from sklearn.tree import DecisionTreeClassifier
    modelA = DecisionTreeClassifier(criterion = 'entropy' , max_depth=3 , class_weight = {0:2 , 1:1} , max_leaf_nodes=8)
    modelA.fit(xtrain, ytrain)
    print("Train Score:", modelA.score(xtrain,ytrain))
    print("Test Score:", modelA.score(xtest,ytest))
    ```
  * **Model Visualization:** The `plot_tree` function is used to visualize the trained decision tree, making it easy to understand the model's decision-making process.
    ```python
    from sklearn.tree import plot_tree
    plt.figure(figsize = (16,14))
    plot_tree(modelA, feature_names=X.columns)
    plt.show()
    ```
  * **Classification Report:** The `classification_report` is used to get a detailed breakdown of the model's performance on a per-class basis, including precision, recall, and F1-score.
  * **Cross-Validation:** To get a more robust estimate of the model's performance, **K-fold cross-validation** is performed. The `cross_val_score` function splits the data into 10 folds (`cv=10`), trains the model 10 times, and returns a list of accuracy scores. The mean of these scores gives a more reliable performance metric.
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(modelB, X, y, cv=10)
    print("Average scores", scores.mean())
    ```
  * **Hyperparameter Tuning (`GridSearchCV`):**
      * **Definition:** `GridSearchCV` is a method for systematically searching through a predefined grid of hyperparameter values to find the best combination that results in the optimal model performance.
      * **Implementation:** A dictionary `params` is defined with various hyperparameter values to test. `GridSearchCV` trains a model for each combination of these parameters and identifies the best one.
    <!-- end list -->
    ```python
    from sklearn.model_selection import GridSearchCV
    params = {
        "criterion" : ["gini", "entropy", "log_loss"],
        "splitter" : ["best", "random"],
        # ... other parameters ...
    }
    modelC = DecisionTreeClassifier()
    gridmodel = GridSearchCV(modelC, params)
    gridmodel.fit(X,y)
    print("Best Parameters:", gridmodel.best_params_)
    print("Best Estimator:", gridmodel.best_estimator_)
    ```
  * **Prediction:** The final, optimized model (`model_after_grid`) is used to make a prediction on a single sample from the test set.

-----

#### 3\. Random Forest Classifier

**Definition:** A **Random Forest** is an ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes (for classification) or the mean prediction (for regression) of the individual trees. This ensemble approach helps reduce overfitting and improves model accuracy and robustness.

  * **Model Parameters:** The `RandomForestClassifier` is initialized with `n_estimators=100`, which means it will build a forest of 100 decision trees. Other parameters, such as `criterion` and `max_depth`, control the behavior of the individual trees within the forest.
  * **Model Training and Evaluation:** The model is trained, and its performance is evaluated on both training and test data using `score()` and `classification_report`.
    ```python
    from sklearn.ensemble import RandomForestClassifier
    model_r = RandomForestClassifier(n_estimators=100 , criterion = 'gini' , max_depth=5)
    model_r.fit(xtrain,ytrain)
    print("Train Score:", model_r.score(xtrain,ytrain))
    print("Test Score:", model_r.score(xtest,ytest))
    ```
  * **Visualization:** The `plot_tree` function is used to visualize a single tree from the forest (specifically, the 99th tree in the list of estimators).
  * **Prediction:** A prediction is made on a single sample from the test set, and then the script demonstrates the ensemble nature of the model by showing how the predictions from all 100 trees are collected to determine the final, majority-voted prediction.
    ```python
    all_predictions = []
    for i in range(0,100):
      all_predictions.append(model_r[i].predict([xtest.values[10]]))
    pd.Series(all_predictions).value_counts()
    ```
    This `value_counts()` output shows the final prediction by majority vote.
