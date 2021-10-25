# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import random

np.random.seed(0)
random.seed(0)

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
# the ratio of test dataset is 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 0)

# split training set into training1 and validation set
# the ratio of valid dataset is 0.25
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size = 0.25, stratify = y_train, random_state = 0)

# Identify the size of training1/validation/test set
print('*' * 200)
print(X_train1.shape, y_train1.shape)
print('*' * 200)
print(X_valid.shape, y_valid.shape)
print('*' * 200)
print(X_test.shape, y_test.shape)

# build DT model
for i in range(1,11):
  clf = DecisionTreeClassifier(max_depth = i, random_state=0)  # random_state로 결과값 고정
  clf.fit(X_train1, y_train1)

  #predict valid set
  y_valid_hat = clf.predict(X_valid)

  print('Depth가 ', i, '일 때')
  # print('실제 class : \n', y_valid)
  # print('예측 class : \n', y_valid_hat)

  # compute train1/valid performance
  # train1 accuracy
  y_train1_hat = clf.predict(X_train1)
  train_acc    = round(accuracy_score(y_train1, y_train1_hat), 3)
  print('train accuracy : ', train_acc)
  # train_acc_list.append(train_acc)

  # valid accuracy
  # y_valid_hat = clf.predict(X_valid)    #생략가능
  valid_acc   = round(accuracy_score(y_valid, y_valid_hat), 3)
  print('valid accuracy : ', valid_acc)
  # valid_acc_list.append(valid_acc)

  print('*' * 25)

# evolve DT model
clf = DecisionTreeClassifier(max_depth = 5)
clf.fit(X_train, y_train)

# compute train/valid performance
# train accuracy
y_train_hat = clf.predict(X_train)
print('train accuracy : ', round(accuracy_score(y_train, y_train_hat), 3))

# test accuracy
y_test_hat = clf.predict(X_test)    #생략가능
print('test accuracy : ', round(accuracy_score(y_test, y_test_hat), 3))

# evolve DT model
clf.fit(X, y)

# compute entire performance
# entrie accuracy
y_hat = clf.predict(X)
print('entire accuracy : ', round(accuracy_score(y, y_hat), 3))