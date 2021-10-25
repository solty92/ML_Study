# import libraries
import mglearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn import tree # plot tree
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################################################################
# practice - synthetic dataset (for regression)
######################################################################
## wave dataset
X, y = mglearn.datasets.make_wave(n_samples = 40)

# plot regression line on wave dataset
plt.scatter(X[:, 0], y, marker = 'o', c = 'steelblue', s = 100) # c = color
plt.ylabel('Target')
plt.xlabel('Feature')
plt.show()

## split dataset into training and test
# the ratio of the test dataset is 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# print result
print('*' * 200)
print(X_train.shape, y_train.shape)
print('*' * 200)
print(X_test.shape, y_test.shape)
print('*' * 200)
print(y_test)

# build DT model
depth = 2
reg = DecisionTreeRegressor(max_depth = depth)
reg.fit(X_train, y_train)

# predict test set
y_test_hat = reg.predict(X_test)
print("실제 값: ", y_test)
print("예측 값: ", y_test_hat)

# compute test performance
# test RMSE(= Root Mean Squared Error)
y_test_hat = reg.predict(X_test)
print('test RMSE: ', round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3))