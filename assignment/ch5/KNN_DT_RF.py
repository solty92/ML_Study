# import libraries
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##################################################
# practice - real world dataset
##################################################
## breast cancer dataset (for classification)
from sklearn.datasets import load_breast_cancer
# cancer = None

## boston dataset (for regression)
from sklearn.datasets import load_boston
# boston = None

X = y = None
X_train = y_train = X_train1 = y_train1 = X_valid = y_valid =  X_test = y_test = None

KNNCLF = KNNREG = DTCLF = DTREG = RFCLF = RFREG = 0


# Scaling
scaler = MinMaxScaler()

# data split
def split(X, y):
    # split dataset into traning and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # split training set into training1 and validation set
    X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)

    return [X_train, y_train, X_train1, y_train1, X_valid, y_valid, X_test, y_test, X, y]


####################################
# algorithms
####################################
def KNN(TYPE):
    global X_train
    global y_train
    global X_train1
    global y_train1
    global X_valid
    global y_valid
    global X_test
    global y_test
    global X
    global y
    global KNNCLF
    global KNNREG

    if TYPE == 'clf':
        valid_accuracy = []
        for HP in range(1, 31):
            clf = KNeighborsClassifier(n_neighbors = HP)
            clf.fit(X_train1, y_train1)

            #predict valid set
            y_valid_hat = clf.predict(X_valid)

            # valid accuracy
            valid_acc   = round(accuracy_score(y_valid, y_valid_hat), 3)
            valid_accuracy.append(valid_acc)

        BESTPARAM = valid_accuracy.index(np.max(valid_accuracy)) + 1

        # evolve KNN model
        clf = KNeighborsClassifier(n_neighbors = BESTPARAM)
        clf.fit(X_train, y_train)

        # evolve KNN model
        clf.fit(X, y)
        # compute entire performance
        # entrie accuracy
        y_hat = clf.predict(X)
        KNNCLF = round(accuracy_score(y, y_hat), 3)


    elif TYPE == 'reg' :
        valid_accuracy = []
        for HP in range(1, 31):
            reg = KNeighborsRegressor(n_neighbors = HP)
            reg.fit(X_train1, y_train1)

            # predict valid set
            y_valid_hat = reg.predict(X_valid)

            # compute valid performance
            # test RMSE(= Root Mean Squared Error)
            valid_accuracy.append(round(mean_squared_error(y_valid, y_valid_hat) ** 0.5, 3))

        BESTPARAM = valid_accuracy.index(np.min(valid_accuracy)) + 1

        # evolve KNN model
        reg = KNeighborsRegressor(n_neighbors = BESTPARAM)
        reg.fit(X_train, y_train)

        # compute test performance
        # test RMSE(= Root Mean Squared Error)
        y_test_hat = reg.predict(X_test)  # 생략가능
        KNNREG = round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3)

def DT(TYPE):
    global X_train
    global y_train
    global X_train1
    global y_train1
    global X_valid
    global y_valid
    global X_test
    global y_test
    global X
    global y
    global DTCLF
    global DTREG

    if TYPE == 'clf' :
        valid_accuracy = []
        for HP in range(1, 31):
            clf = DecisionTreeClassifier(max_depth = HP, random_state=0)  # random_state로 결과값 고정
            clf.fit(X_train1, y_train1)

            #predict valid set
            y_valid_hat = clf.predict(X_valid)

            # valid accuracy
            valid_acc = round(accuracy_score(y_valid, y_valid_hat), 3)
            valid_accuracy.append(valid_acc)

        BESTPARAM = valid_accuracy.index(np.max(valid_accuracy)) + 1

        # evolve DT model
        clf = DecisionTreeClassifier(max_depth = 5)
        clf.fit(X_train, y_train)

        # evolve DT model
        clf.fit(X, y)

        # compute entire performance
        # entrie accuracy
        y_hat = clf.predict(X)
        DTCLF = round(accuracy_score(y, y_hat), 3)

    elif TYPE == 'reg' :
        valid_accuracy = []
        for HP in range(1, 31):
            reg = DecisionTreeRegressor(max_depth = HP, random_state=0)  # random_state로 결과값 고정
            reg.fit(X_train1, y_train1)

            # predict valid set
            y_valid_hat = reg.predict(X_valid)

            # compute valid performance
            # test RMSE(= Root Mean Squared Error)
            valid_accuracy.append(round(mean_squared_error(y_valid, y_valid_hat) ** 0.5, 3))

        BESTPARAM = valid_accuracy.index(np.min(valid_accuracy)) + 1

        reg = DecisionTreeRegressor(max_depth = BESTPARAM)
        reg.fit(X_train, y_train)

        # compute test performance
        # test RMSE(= Root Mean Squared Error)
        y_test_hat = reg.predict(X_test)
        DTREG = round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3)

def RF(TYPE):
    global X_train
    global y_train
    global X_train1
    global y_train1
    global X_valid
    global y_valid
    global X_test
    global y_test
    global X
    global y
    global RFCLF
    global RFREG

    if TYPE == 'clf' :
        valid_accuracy = []
        for HP in range(1, 31):
            clf = RandomForestClassifier(n_estimators= HP, random_state= 0)
            clf.fit(X_train1, y_train1)

            #predict valid set
            y_valid_hat = clf.predict(X_valid)

            # valid accuracy
            valid_acc = round(accuracy_score(y_valid, y_valid_hat), 3)
            valid_accuracy.append(valid_acc)

        BESTPARAM = valid_accuracy.index(np.max(valid_accuracy)) + 1

        clf = RandomForestClassifier(n_estimators= BESTPARAM, random_state= 0)
        clf.fit(X_train, y_train)

        # evolve DT model
        clf.fit(X, y)

        # compute entire performance
        # entrie accuracy
        y_hat = clf.predict(X)
        RFCLF = round(accuracy_score(y, y_hat), 3)

    elif TYPE == 'reg' :
        valid_accuracy = []
        for HP in range(1, 31):
            reg = RandomForestRegressor(n_estimators = HP, random_state = 2)
            reg.fit(X_train1, y_train1)

            # predict valid set
            y_valid_hat = reg.predict(X_valid)

            # compute valid performance
            # test RMSE(= Root Mean Squared Error)
            valid_accuracy.append(round(mean_squared_error(y_valid, y_valid_hat) ** 0.5, 3))

        BESTPARAM = valid_accuracy.index(np.min(valid_accuracy)) + 1

        reg = RandomForestRegressor(n_estimators = BESTPARAM, random_state = 2)
        reg.fit(X_train, y_train)

        # compute test performance
        # test RMSE(= Root Mean Squared Error)
        y_test_hat = reg.predict(X_test)
        RFREG = round(mean_squared_error(y_test, y_test_hat) ** 0.5, 3)



####################################
# TYPE
####################################
# classification
def clf(ALGORITHM):
    global X_train
    global y_train
    global X_train1
    global y_train1
    global X_valid
    global y_valid
    global X_test
    global y_test
    global X
    global y

    cancer = load_breast_cancer()

    # divide cancer dataset into X(= input feature) and y(= target)
    X = cancer.data
    y = cancer.target

    SPLITLIST = split(X, y)
    X_train = SPLITLIST[0]
    y_train = SPLITLIST[1]
    X_train1 = SPLITLIST[2]
    y_train1 = SPLITLIST[3]
    X_valid = SPLITLIST[4]
    y_valid = SPLITLIST[5]
    X_test = SPLITLIST[6]
    y_test = SPLITLIST[7]
    X = SPLITLIST[8]
    y = SPLITLIST[9]

    # scaling
    if(ALGORITHM == 'KNN'):
        X_train1 = scaler.fit_transform(X_train1)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    if( ALGORITHM == 'KNN' ):
        KNN('clf')
    elif( ALGORITHM == 'DT' ):
        DT('clf')
    elif ( ALGORITHM == 'RF' ):
        RF('clf')


# regression
def reg(ALGORITHM):
    global X_train
    global y_train
    global X_train1
    global y_train1
    global X_valid
    global y_valid
    global X_test
    global y_test
    global X
    global y
    boston = load_boston()

    # divide boston dataset into X(= input feature) and y(= target)1
    X = boston.data
    y = boston.target

    SPLITLIST = split(X, y)
    X_train = SPLITLIST[0]
    y_train = SPLITLIST[1]
    X_train1 = SPLITLIST[2]
    y_train1 = SPLITLIST[3]
    X_valid = SPLITLIST[4]
    y_valid = SPLITLIST[5]
    X_test = SPLITLIST[6]
    y_test = SPLITLIST[7]
    X = SPLITLIST[8]
    y = SPLITLIST[9]

    # scaling
    if ALGORITHM == 'KNN':
        X_train1 = scaler.fit_transform(X_train1)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    if ALGORITHM == 'KNN':
        KNN('reg')
    elif ALGORITHM == 'DT':
        DT('reg')
    elif ALGORITHM == 'RF':
        RF('reg')



dataType = ['cancer', 'boston']
ALGORITHM = ['KNN', 'DT', 'RF']

for i in dataType:
    if i == 'cancer':
        for i in ALGORITHM:
            clf(i)
    elif i == 'boston':
        for i in ALGORITHM:
            reg(i)

print('KNNCLF : ', KNNCLF)
print('KNNREG : ', KNNREG)
print('DTCLF : ', DTCLF)
print('DTREG : ', DTREG)
print('RFCLF : ', RFCLF)
print('RFREG : ', RFREG)
