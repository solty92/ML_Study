# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

########################################
# practice - synthetic dataset (fro classification)
########################################
## forge dataset

X, y = mglearn.datasets.make_forge()

# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape :", X.shape)

## split dataset into training and test
# the ratio of test dataset is 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# StandardScaler
scaler = MinMaxScaler()

# scale the input data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('training set without scaling')
print(np.round(X_train, 3))
print('*' * 200)
print('training set with scaling')
print(np.round(X_train_scaled), 3)

# build KNN model
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train_scaled, y_train)

# compute train/test performance
# train accuarcy
y_train_hat = clf.predict(X_train_scaled)
print('train accuarcy : ', round(accuracy_score(y_train, y_train_hat), 3))

#test accuracy
y_test_hat = clf.predict(X_test_scaled)
print('test accuracy : ', round(accuracy_score(y_test, y_test_hat), 3))

# print result
print('*' * 200)
print(X_train.shape, y_train.shape)
print('*' * 200)
print(X_test.shape, y_test.shape)
print('*' * 200)
print(y_test)

# build KNN model
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)

# predict test set
y_test_hat = clf.predict(X_test)

print('*' * 200)
print(y_test)     # 실제 class
print('*' * 200)
print(y_test_hat) # 예측 class

# compute train/test performance
# train accuracy
y_train_hat = clf.predict(X_train)
print('train accuracy : ', round(accuracy_score(y_train, y_train_hat), 3))

# test accuracy
y_test_hat = clf.predict(X_test)
print('test accuracy : ', round(accuracy_score(y_test, y_test_hat), 3))