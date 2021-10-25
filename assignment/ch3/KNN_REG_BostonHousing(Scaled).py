# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##################################################
# practice - real world dataset (for regression)
##################################################
## boston dataset
from sklearn.datasets import load_boston
boston = load_boston()

# divide boston dataset into X(= input feature) and y(= target)1
X = boston.data
y = boston.target

# Scaling
scaler = MinMaxScaler()
# scaler = StandardScaler()

# scaler.fit(X)
# X = scaler.transform(X)
X = scaler.fit_transform(X)

# show result
X_df = pd.DataFrame(X)
print(X_df)
print(y)

# split dataset into traning and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# split training set into training1 and validation set
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)

# identify the size of training/validation/test set
print(X_train1.shape, y_train1.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# build KNN model
for i in range(1, 11):
  reg = KNeighborsRegressor(n_neighbors = i)
  reg.fit(X_train1, y_train1)

  # predict valid set
  y_valid_hat = reg.predict(X_valid)

  print('K가 ', i, '일 때')
  # print('실제 class : \n', y_valid)
  # print('예측 class : \n', y_valid_hat)

  # compute valid performance
  # test RMSE(= Root Mean Squared Error)
  print('test RMSE : ', round(mean_squared_error(y_valid, y_valid_hat) ** 0.5, 3))
  print('*' * 20)

# evolve KNN model
reg = KNeighborsRegressor(n_neighbors = 2)
reg.fit(X_train, y_train)

# predict test set
y_test_hat = reg.predict(X_test)
print("실제 값 : ", y_test)
print("예측 값 : ", y_test_hat)

# compute test performance
# test RMSE(= Root Mean Squared Error)
y_test_hat = reg.predict(X_test)  # 생략가능
print('test RMSE : ', round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3))