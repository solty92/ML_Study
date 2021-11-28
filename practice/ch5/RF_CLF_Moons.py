# import libraries
import mglearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################################################################
# practice - synthetic dataset (for classification)
###################################################################################
## forge dataset
X, y = make_moons(n_samples= 100, noise= 0.25, random_state= 3)

## split dataset into training and test
# the ratio of test dataset is 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

# build RF model (random_state가 바뀌면 split에서 선택하는 feature가 바뀐다.)
clf = RandomForestClassifier(n_estimators= 5, random_state= 2)
clf.fit(X_train, y_train)

# predict test set
y_test_hat = clf.predict(X_test)

# compute train/test performance
# train accuracy
y_train_hat = clf.predict(X_train)
print('train accuracy: ', round(accuracy_score(y_train, y_train_hat), 3))

# test accuracy
y_test_hat - clf.predict(X_test)
print('train accuracy: ', round(accuracy_score(y_test, y_test_hat), 3))
