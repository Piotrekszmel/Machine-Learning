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

## What have been implemented so far? (17.04.2020)

### supervised_learning/

#### regression.py:
1) l1_regularization
2) l2_regularization
3) l1_l2_regularization
4) Linear Regression
5) Lasso Regression
6) Polynomial Regression
7) Ridge Regression
8) PolynomialRidge Regression
9) ElasticNet

#### LogisticRegression.py:
1) LogisticRegression
 
#### decision_tree.py:
1) Regression Tree
2) Classification Tree

#### naive_bayes:
1) NaiveBayes

### deep_learning/

#### activation_functions.py: 
1) Sigmoid
2) Softmax
3) Tanh
4) ReLU
  
#### layers.py:
1) Dense
2) Activation

#### neural_network.py:
1) NeuralNetwork

#### loss_functions.py:
1) SquareLoss
2) CrossEntropy

#### optimizers.py
1) SGD
2) NesterovAcceleratedGradient