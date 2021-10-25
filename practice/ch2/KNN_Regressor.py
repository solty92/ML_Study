# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##########################################
# practice - synthetic dataset
##########################################
## wave dataset(for regression)
X, y = mglearn.datasets.make_wave(n_samples = 40)

# plot dataset
mglearn.discrete_scatter(X[:, 0],y)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape : ", X.shape)

## split dataset into training and test
# the ratio of the test dataset is 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# print result
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(y_test)

# build KNN model
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(X_train, y_train)

# predict test set
y_test_hat = reg.predict(X_test)
print("실제 값 : ", y_test)
print("예측 값 : ", y_test_hat)

# compute test performance
# test RMSE(= Root Mean Squared Error)
y_test_hat = reg.predict(X_test)    # 생략가능
print('test RMSE : ', round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3))