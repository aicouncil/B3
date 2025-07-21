# Machine Learning Concepts and Implementations

This document provides a comprehensive overview of Machine Learning, its categories, and practical implementations using Python's `numpy`, `pandas`, `matplotlib`, and `scikit-learn` libraries, based on the provided `g3_d9_ml1.py` file.

## 1. Introduction to Machine Learning

**Machine Learning** is a branch of artificial intelligence that empowers systems to learn from data, allowing them to improve performance on specific tasks without explicit programming.

## 2. Types of Machine Learning

Machine Learning is broadly categorized into three types:

### 2.1. Supervised Learning

In Supervised Learning, models are trained on **labeled datasets**, where each input is associated with a corresponding correct output. The model's objective is to learn a mapping function from inputs to outputs to make accurate predictions on new, unseen data.

#### Sub-types of Supervised Learning:

* **Classification:** Used when the goal is to predict a discrete category or class from a fixed number of categories.
    * **Common Classification Algorithms:**
        * K-Nearest Neighbors (KNN)
        * Logistic Regression
        * Decision Tree
        * Random Forest
        * Support Vector Machine (SVM)
        * Naive Bayes
* **Regression:** Used for predicting continuous numerical values.

### 2.2. Unsupervised Learning

Unsupervised Learning involves training models on **unlabeled datasets**. The models' task is to discover hidden patterns, structures, or relationships within the data without any prior output guidance.

### 2.3. Reinforcement Learning

Reinforcement Learning involves an agent learning to make sequential decisions by interacting with an environment. The agent receives rewards or penalties based on its actions, aiming to maximize cumulative reward over time through trial and error.

## 3. Practical Implementation Examples (Python)

The notebook provides several Python examples to illustrate data handling and machine learning model building.

### 3.1. Basic Data Generation and Visualization for Classification

This section demonstrates how to create a synthetic dataset and visualize it to understand the concept of data points belonging to different classes.

* **Importing Libraries:**
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    ```
* **Data Generation:** Three sets of random 2D integer arrays (`n1`, `n2`, `n3`) are generated, representing different clusters of data. These are then vertically stacked and converted into a Pandas DataFrame `d1` with columns 'A' and 'B'. A 'Label' column is added manually to categorize these data points into three classes (0, 1, 2).
    ```python
    n1 = np.random.randint(1,20,(5,2))
    n2 = np.random.randint(17,40,(5,2))
    n3 = np.random.randint(37,60,(5,2))

    d1 = pd.DataFrame(np.vstack((n1,n2,n3)) , columns = ['A', 'B'])
    d1['Label'] = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    # d1 will show the combined DataFrame with assigned labels
    ```
* **Data Visualization:** A scatter plot is generated to visualize the data points in the 'A' vs. 'B' plane, with points colored according to their 'Label'. This visually separates the data into distinct clusters.
    ```python
    plt.scatter(d1['A'] , d1['B'] , c = d1['Label'])
    plt.show()
    ```

### 3.2. K-Nearest Neighbors (KNN) - Manual Prediction Example

This section illustrates the core idea behind the KNN algorithm by manually predicting the label for a new data point `x_new = [25, 65]`.

* **Visualize New Point:** The new point `x_new` is added to the existing scatter plot for visual context.
    ```python
    plt.scatter(d1['A'] , d1['B'] , c = d1['Label'])
    plt.scatter(25,65,marker='*',s=100,c='r')
    plt.show()
    ```
* **Calculate Distances:** Euclidean distance from `x_new` to all existing data points in `d1` is calculated and stored in a new 'distance' column.
    ```python
    x_new = [25 , 65]
    d1['distance'] = ((d1['A'] - x_new[0])**2 + (d1['B'] - x_new[1])**2)**0.5
    ```
* **Identify K-Nearest Neighbors:** The DataFrame is sorted by 'distance', and the top 5 (assuming K=5) nearest data points are identified.
    ```python
    d1.sort_values('distance').head(5)
    ```
    * **Prediction Logic:** Based on the labels of the 5 nearest data points, the new point's label is determined by majority vote.
        * **Conclusion:** "Out of 5 nearest data points from [25,65], 4 data points are of category 2, so, [25, 65] will be classified as category 2."

### 3.3. K-Neighbors Classifier using Scikit-Learn (Synthetic Data)

This demonstrates using `scikit-learn`'s `KNeighborsClassifier` for a more streamlined approach to classification.

* **Data Preparation (re-generated for clean slate):** The synthetic data `d1` (features 'A', 'B' and 'Label') is re-generated.
    ```python
    n1 = np.random.randint(1,20,(5,2))
    n2 = np.random.randint(17,40,(5,2))
    n3 = np.random.randint(37,60,(5,2))

    d1 = pd.DataFrame(np.vstack((n1,n2,n3)) , columns = ['A', 'B'])
    d1['Label'] = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    ```
* **Feature and Label Separation:**
    * `X`: Features or independent variables (`d1[['A','B']]`).
    * `y`: Labels or dependent variables (`d1['Label']`).
    ```python
    X = d1[['A','B']]
    y = d1['Label']
    ```
* **Model Definition and Training:**
    * `KNeighborsClassifier` is imported.
    * A model instance `modelA` is created with `n_neighbors=5`.
    * The model is trained using the `fit()` method with `X` and `y`.
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    modelA = KNeighborsClassifier(n_neighbors=5)
    modelA.fit(X,y)
    ```
* **Prediction and Accuracy Evaluation:**
    * Predictions (`Yp`) are made on the training data `X`.
    * Model accuracy is calculated by comparing predicted labels `Yp` with real labels `y`.
    ```python
    Yp = modelA.predict(X)
    (y == Yp).sum()/len(X) # Prints the accuracy
    ```
* **Predicting New Data Point:** The trained `modelA` predicts the label for `x_new = [25, 65]`.
    ```python
    print(modelA.predict([[25,65]])) # Prints the predicted class for [25,65]
    ```

### 3.4. Classification Model for Predicting Flower Species (Iris Dataset)

This section provides a complete example of building and evaluating a classification model using the well-known Iris dataset.

* **Load Iris Dataset:**
    ```python
    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Iris.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/Iris.csv)')
    df.head() # Displays first few rows of Iris dataset
    ```
* **Explore Unique Species:** Identifies the unique classes in the 'Species' column.
    ```python
    df['Species'].unique() # Output: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    ```
* **Separate Features and Labels:**
    * `X`: Features (all columns except 'Id' and 'Species').
    * `y`: Labels ('Species' column).
    ```python
    X = df.drop(columns = ['Id','Species'])
    y = df['Species']
    ```
* **Split Data into Training and Test Sets:** The data is split into 75% for training and 25% for evaluation/validation using `train_test_split`.
    ```python
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size=0.75)
    ```
    * **Verify Shapes of Splits:** Prints the dimensions of the original and split datasets.
        ```python
        print(X.shape , ' ' , xtrain.shape , ' ' , xtest.shape)
        print(y.shape , ' ' , ytrain.shape , ' ' , ytest.shape)
        ```
* **Model Training:** A `KNeighborsClassifier` model (`modelB`) is initialized with `n_neighbors=5` and trained on the `xtrain` and `ytrain` data.
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    modelB = KNeighborsClassifier(n_neighbors=5)
    modelB.fit(xtrain, ytrain)
    ```
* **Evaluate Model Accuracy:** Predictions are made on both the training and test sets, and their respective accuracies are calculated by comparing predicted labels with real labels.
    ```python
    ytrainPred = modelB.predict(xtrain)
    ytestPred = modelB.predict(xtest)

    print("Model accuracy on training data-")
    print((ytrain == ytrainPred).sum()/len(xtrain))

    print()

    print("Model accuracy on test data-")
    print((ytest == ytestPred).sum()/len(xtest))
    ```
* **Predicting New Iris Flower Species:** Demonstrates how `modelB` can predict the species of a new Iris flower given its measurements.
    ```python
    # Example prediction with arbitrary values, actual values should match Iris features
    print(modelB.predict([[2.3,3.2,3.9,3.4]]))
    ```

### 3.5. Wine Class Classification

The file introduces the Wine dataset, setting up for a similar classification task.
* **Load Wine Dataset:**
    ```python
    df = pd.read_csv('[https://github.com/bipulshahi/Dataset/raw/refs/heads/main/wine.csv](https://github.com/bipulshahi/Dataset/raw/refs/heads/main/wine.csv)')
    df.head() # Displays first few rows of Wine dataset
    ```
* **Explore Unique Target Classes:** Shows the unique classes present in the 'Target' column of the Wine dataset.
    ```python
    df['Target'].unique() # Typically shows the different wine classes (e.g., [0, 1, 2])
    ```
This section sets the stage for building and evaluating a predictive model for Wine class classification, similar to the Iris example, using the provided data.