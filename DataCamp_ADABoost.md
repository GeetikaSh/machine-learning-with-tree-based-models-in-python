# Table of Content
- [Understanding AdaBoost (Adaptive Boosting)](#understanding-adaboost-adaptive-boosting)
  - [Explanation](#explanation)
  - [Key Concepts](#key-concepts)
  - [Algorithm Steps](#algorithm-steps)
  - [Advantages](#advantages)
  - [Limitations](#limitations)
  - [DataCamp Solutions](#datacamp-solutions)
  - [Interview Questions on AdaBoost](#interview-questions-on-adaboost)

---

# Understanding AdaBoost (Adaptive Boosting)

AdaBoost is a popular boosting algorithm used in machine learning to improve the performance of weak classifiers
It combines multiple weak learners to create a strong learner capable of achieving higher accuracy.

![A generalized workflow of the AdaBoost algorithm](https://miro.medium.com/v2/resize:fit:828/format:webp/1*eosJ6Yg0epLuH5kHsMaPWQ.png)

**Source:** [Boosting — Adaboost, Gradient Boost and XGBoost](https://medium.com/@pingsubhak/boosting-adaboost-gradient-boost-and-xgboost-bdda87eed44e)

---

## Explanation:

1. **Initial Data Weighting**:
   All data points start with equal weights.

2. **Weak Learner Training**:
   - Train a weak learner (e.g., a decision stump) on the weighted dataset.
   - Misclassified samples are given higher weights.

3. **Sequential Learning**:
   Each new weak learner is trained on the updated weights, focusing on the difficult-to-classify examples.

4. **Final Aggregation**:
   Combine all weak learners using a weighted majority vote, where each learner’s weight is proportional to its accuracy.

This diagram provides a clear and structured view of the AdaBoost process, helping visualize how the algorithm progressively improves performance.


## Key Concepts

1. **Boosting**: 
   Boosting is a technique that converts weak learners into strong ones by sequentially applying them to weighted versions of the data.

2. **Weak Learner**: 
   A weak learner is a model that performs slightly better than random guessing, such as a decision stump (a decision tree with one split).

3. **Weighted Data**:
   AdaBoost assigns weights to training data points. Misclassified points get higher weights in the next iteration, ensuring the algorithm focuses more on difficult examples.

---

## Algorithm Steps

1. **Initialization**:
   Assign equal weights to all training data points:  
   $w_i = \frac{1}{N}, \; \forall \; i \in \{1, 2, ..., N\}$  
   where $N$ is the total number of samples.

2. **Iterative Training**:
   - For each iteration $t$ (total $T$ iterations):
     1. Train a weak learner on the weighted dataset.
     2. Compute the weighted error rate $\epsilon_t$:  
        $\epsilon_t = \frac{\sum_{i=1}^N w_i \cdot \mathbb{I}(y_i \neq h_t(x_i))}{\sum_{i=1}^N w_i}$  
        where $\mathbb{I}$ is an indicator function, $y_i$ is the true label, $x_i$ is the feature vector, and $h_t(x_i)$ is the predicted label.
     3. Compute the model's weight:  
        $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$
     4. Update the weights of the data points:  
        $w_i \leftarrow w_i \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)$
     5. Normalize the weights to ensure they sum to 1.

3. **Final Model**:
   Combine the weak learners using a weighted majority vote:  
   $H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \cdot h_t(x)\right)$

---

## Advantages

- Can significantly improve the accuracy of weak learners.
- Works well with diverse datasets and models.

## Limitations

- Sensitive to noisy data and outliers, as it assigns higher weights to misclassified points.
- Requires careful tuning of hyperparameters (e.g., number of iterations).

---

## DataCamp Solutions

- **Define ADA Boost Classifier**
```python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth= 2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator= dt, n_estimators= 180, random_state=1)
```
- **Train the AdaBoost classifier**
``` python
# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]
```
- **Evaluate the AdaBoost classifier**
``` python
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
```

## Interview Questions On AdaBoost
**1. How does AdaBoost minimize exponential loss?**

AdaBoost minimizes an exponential loss function by focusing on misclassified points. Each weak learner is optimized to reduce this loss iteratively.
The exponential loss penalizes large errors more, ensuring misclassified examples receive higher attention in subsequent rounds.

**2. What happens if all weak learners in AdaBoost are perfect classifiers?**

If all weak learners are perfect classifiers, AdaBoost would achieve 100% accuracy on the training dataset in fewer iterations.
However, overfitting might occur, especially if the dataset has noise.

**3.. How does AdaBoost differ from Bagging?**

AdaBoost: Focuses on reweighting data points to handle hard-to-classify examples and combines weak learners sequentially.
Bagging: Trains multiple models independently on different bootstrapped datasets and averages their predictions. Bagging reduces variance, whereas AdaBoost reduces bias.

**4. How would you address overfitting in AdaBoost?**

To address overfitting:
- Limit the number of iterations (estimators).
- Use simpler base learners like decision stumps.
- Add regularization to the weak learners.
- Reduce the learning rate.

**5. Explain how AdaBoost can handle class imbalance.**

AdaBoost inherently reweights data points, giving more importance to misclassified examples.
In a class-imbalanced dataset, it can focus on the minority class by assigning higher weights to its misclassified samples.
However, additional techniques like stratified sampling or modifying the loss function may further improve performance.
