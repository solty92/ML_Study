# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################################
# practice - real world dataset (for classification)
###################################################
## breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# divide cancer dataset into X(= input feature) and y(= target)
X = cancer.data
y = cancer.target

# show result
X_df = pd.DataFrame(X)
print(X_df)
print(y)

# split dataset into traning and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 0)

# split training set into training1 and validation set
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size = 0.25, stratify = y_train, random_state = 0)

# Identify the size of training1/validation/test set
print(X_train1.shape, y_train1.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# train_acc_list = []
# valid_acc_list = []
# k_list         = range(1, 11)
# build KNN model
for i in range(10):
  clf = KNeighborsClassifier(n_neighbors = i+1)
  clf.fit(X_train1, y_train1)

  #predict valid set
  y_valid_hat = clf.predict(X_valid)

  print('K가 ', i+1, '일 때')
  # print('실제 class : \n', y_valid)
  # print('예측 class : \n', y_valid_hat)

  # compute train1/valid performance
  # train1 accuracy
  y_train1_hat = clf.predict(X_train1)
  train_acc    = round(accuracy_score(y_train1, y_train1_hat), 3)
  print('train accuracy : ', train_acc)
  # train_acc_list.append(train_acc)

  # valid accuracy
  y_valid_hat = clf.predict(X_valid)    #생략가능
  valid_acc   = round(accuracy_score(y_valid, y_valid_hat), 3)
  print('valid accuracy : ', valid_acc)
  # valid_acc_list.append(valid_acc)

  print('*' * 25)

# list comprehension in python
# optimal_k = [i for i, acc in enumerate(valid_acc_list) if acc == max(valid_acc_list)]

# evolve KNN model
clf = KNeighborsClassifier(n_neighbors = 9)
clf.fit(X_train, y_train)

# compute train/valid performance
# train accuracy
y_train_hat = clf.predict(X_train)
print('train accuracy : ', round(accuracy_score(y_train, y_train_hat), 3))

# test accuracy
y_test_hat = clf.predict(X_test)    #생략가능
print('test accuracy : ', round(accuracy_score(y_test, y_test_hat), 3))

# evolve KNN model
clf.fit(X, y)

# compute entire performance
# entrie accuracy
y_hat = clf.predict(X)
print('entire accuracy : ', round(accuracy_score(y, y_hat), 3))