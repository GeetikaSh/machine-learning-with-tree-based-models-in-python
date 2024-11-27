# Machine Learning with Tree-Based Models in Python

Tree-based models are a family of supervised learning algorithms that create models in the form of a tree structure. 
These models are powerful and versatile, excelling in both regression and classification tasks.

This repository consists of the Tree-Based Model notes. I am also going to attach DataCamp solutions to it as well. 
The aim is to create a basic notes file that you can review or read before implementing any tree-based model to help you get started with modeling.

---

## Table of Contents
1. **Introduction**
   - Overview of Tree-Based Models
   - Applications of Tree Models
2. **Types of Tree-Based Models**
   - Decision Trees
   - Ensemble Models
     - Random Forests
     - Gradient Boosting Machines (GBM)
     - XGBoost, LightGBM, and CatBoost
3. **Core Concepts**
   - Splitting Criteria (Gini, Entropy, MSE)
   - Overfitting and Pruning
   - Feature Importance
4. **Model Training and Evaluation**
   - Train-Test Split
   - Cross-Validation
   - Metrics: Accuracy, RMSE, AUC
5. **Advanced Topics**
   - Hyperparameter Tuning
   - Handling Imbalanced Data
   - Feature Engineering
   - Interpretability of Models
6. **Case Study**
   - End-to-End Pipeline: Predicting Customer Churn
   - Data Preprocessing
   - Model Building
   - Model Evaluation
7. **Best Practices**
   - Avoiding Overfitting
   - Ensuring Generalizability
8. **Resources and References**
   - Recommended Reading
   - Courses and Tutorials

---

## 1. Introduction

Tree-based models mimic human decision-making by learning decision rules from data. These models can handle categorical and
numerical data and are non-parametric, making them flexible for a wide range of datasets.

---

## 2. Types of Tree-Based Models

### Decision Trees
- Single tree structure.
- Greedy algorithm to split data at each node.

### Ensemble Models
#### Random Forests
Random Forest is a powerful and versatile machine learning algorithm that belongs to the family of ensemble learning methods. 
It works by constructing multiple decision trees during training and 
outputting the mode of the classes (classification) or mean prediction (regression) from all the trees.

---

##### Key Characteristics of Random Forest
1. **Ensemble Method**:
   - Combines multiple decision trees to improve predictive accuracy.
2. **Bagging (Bootstrap Aggregating)**:
   - Uses random subsets of the data to train individual trees, ensuring diversity.
3. **Feature Randomness**:
   - Each split considers a random subset of features to further reduce correlation among trees.
4. **Reduces Overfitting**:
   - By averaging the results of multiple trees, Random Forest prevents overfitting that is common with individual decision trees.

---

##### Advantages of Random Forest
- **Handles Missing Data**:
  - Can maintain accuracy with missing values.
- **Works with Both Classification and Regression**:
  - Versatile and widely applicable.
- **Robust to Outliers**:
  - Can handle noisy and unbalanced datasets effectively.
- **Feature Importance**:
  - Provides a measure of the relative importance of each feature.

---

##### Disadvantages of Random Forest
- **Slower Prediction**:
  - Due to the need to average multiple trees, it can be slower compared to single-tree models.
- **Memory Intensive**:
  - Requires more memory to store multiple trees.
- **Less Interpretable**:
  - Compared to single decision trees, Random Forests are more of a "black-box" model.

---

##### How Random Forest Works

1. **Data Sampling**:
   - Random subsets of the training dataset are sampled with replacement (bootstrap sampling).
2. **Tree Building**:
   - Each tree is trained on a different subset of the data.
   - Random subsets of features are considered at each split.
3. **Prediction**:
   - For classification, predictions are made by majority voting among the trees.
   - For regression, the predictions are averaged.

---

##### Implementation in Python

###### Example: Classification with Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

###### Example: Regression with Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)

# Evaluate
y_pred = rf_reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

#### Gradient Boosting Machines
- Sequentially builds trees to minimize loss.
- More prone to overfitting but highly accurate.

#### XGBoost, LightGBM, and CatBoost
- Popular implementations of gradient boosting with optimizations for speed and performance.

---

## 3. Core Concepts

### Splitting Criteria
- **Classification**: Gini Impurity, Entropy.
- **Regression**: Mean Squared Error (MSE).

### Overfitting and Pruning
- **Overfitting**: Trees growing too deep, capturing noise.
- **Pruning**: Reducing tree depth for simplicity.

### Feature Importance
- Helps in understanding the contribution of each feature.

---

## 4. Model Training and Evaluation

### Key Metrics
- **Classification**: Accuracy, Precision, Recall, AUC-ROC.
- **Regression**: Mean Squared Error (MSE), R-squared.

### Cross-Validation
- K-fold validation to assess model robustness.

---

## 5. Advanced Topics

### Hyperparameter Tuning
- **Techniques**:
  - Grid Search
  - Random Search
  - Bayesian Optimization
- **Parameters to Tune**:
  - Depth
  - Learning Rate
  - Number of Trees

### Handling Imbalanced Data
- **Techniques**:
  - Resampling (Over-sampling, Under-sampling)
  - Weighted Loss Functions

### Feature Engineering
- One-hot encoding for categorical variables.
- Standardization for numerical features (if required).

### Model Interpretability
- Tools like SHAP and LIME for understanding model predictions.

---

## 6. Best Practices

- Use ensemble methods for better performance.
- Regularize models to avoid overfitting.
- Validate models thoroughly before deployment.

---

## 7. Resources and References

### Books:
- *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* by Aurélien Géron.
- *"Introduction to Statistical Learning"* by James, Witten, Hastie, and Tibshirani.

### Online Tutorials:
- [scikit-learn documentation](https://scikit-learn.org)
