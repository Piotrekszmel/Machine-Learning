# Machine-Learning
Machine Learning algorithms implementation from scratch. 

The purpose of this repository is to learn about how these algorithms work and learn how to work with larger projects in python.
### Quick example
```python
from MachineLearning.utils.data_manipulation import train_test_split, normalize
from MachineLearning.supervised_learning.LogisticRegression import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### What have been implemented so far? (1.04.2020)

### supervised_learning/
1) regression.py:
  - l1_regularization
  - l2_regularization
  - l1_l2_regularization
  - Linear Regression
  - Lasso Regression
  - Polynomial Regression
  - Ridge Regression
  - PolynomialRidge Regression
  - ElasticNet

2) LogisticRegression.py:
  - LogisticRegression
 
3) decision_tree.py:
  - Regression Tree
  - Classification Tree

4) naive_bayes:
  - NaiveBayes

### deep_learning/
1) activation_functions.py: 
  - Sigmoid
  - Softmax
  - Tanh
  - ReLU
  
2) layers.py:
  - Dense
