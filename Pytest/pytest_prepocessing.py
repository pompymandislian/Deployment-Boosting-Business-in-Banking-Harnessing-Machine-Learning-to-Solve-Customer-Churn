import pytest
import numpy as np
import joblib
import pandas as pd
from encodermodule import Encoder
from feature_data import Scaling, imbalance


X_train = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/X_train.csv")
y_train = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/y_train.csv")
X_test = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/X_test.csv")
y_test = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/y_test.csv")

def test_encoder_ohe():

    # split category columns
    category_cols_train = X_train.select_dtypes(include=['object'])
    category_cols_test = X_test.select_dtypes(include=['object'])

    #split numeric columns
    numeric_cols_train = X_train.select_dtypes(include=['number'])
    numeric_cols_test = X_test.select_dtypes(include=['number'])

    # encoder ohe
    ohe_train = Encoder(category_cols_train)
    data_ohe_train = ohe_train.Ohe_encoder()

    ohe_test = Encoder(category_cols_test)
    data_ohe_test = ohe_test.Ohe_encoder()

    # concat data numeric and category
    x_train = pd.concat([data_ohe_train, numeric_cols_train], axis=1)
    x_test = pd.concat([data_ohe_test, numeric_cols_test], axis=1)

    # testing
    assert data_ohe_train.shape[1] == 8
    assert data_ohe_test.shape[1] == 8
    assert numeric_cols_train.shape[1] == 6
    assert numeric_cols_test.shape[1] == 6
    assert x_train.shape[1] == 14
    assert x_test.shape[1] == 14


# split category columns
category_cols_train = X_train.select_dtypes(include=['object'])
category_cols_test = X_test.select_dtypes(include=['object'])

#split numeric columns
numeric_cols_train = X_train.select_dtypes(include=['number'])
numeric_cols_test = X_test.select_dtypes(include=['number'])

# encoder ohe
ohe_train = Encoder(category_cols_train)
data_ohe_train = ohe_train.Ohe_encoder()

ohe_test = Encoder(category_cols_test)
data_ohe_test = ohe_test.Ohe_encoder()

# concat data numeric and category
x_train = pd.concat([data_ohe_train, numeric_cols_train], axis=1)
x_test = pd.concat([data_ohe_test, numeric_cols_test], axis=1)

def test_balance_data():
    
    balance = imbalance()
    X_train_smote, y_train_smote = balance.perform_smote(x_train, y_train)
    X_test_smote, y_test_smote = balance.perform_smote(x_test, y_test)

    # Assertion untuk memastikan operasi SMOTE berhasil dilakukan
    assert len(X_train_smote) == len(y_train_smote)
    assert len(X_test_smote) == len(y_test_smote)

    # Assertion untuk memastikan data telah seimbang
    assert sum(y_train_smote) == len(y_train_smote) / 2
    assert sum(y_test_smote) == len(y_test_smote) / 2
    
balance = imbalance()
X_train_smote, y_train_smote = balance.perform_smote(x_train, y_train)
X_test_smote, y_test_smote = balance.perform_smote(x_test, y_test)

def test_standardization():
    # split data
    columns_feature = ['balance','credit_score','estimated_salary']

    # drop columns
    drop_smote_train = X_train_smote.drop(columns=['balance', 'credit_score', 'estimated_salary'])
    drop_smote_test = X_test_smote.drop(columns=['balance', 'credit_score', 'estimated_salary'])

    # Scaling
    scaling = Scaling()
    feature_smote_train, scaler = scaling.standardizeData(X_train_smote[columns_feature])
    feature_smote_test, scaler = scaling.standardizeData(X_test_smote[columns_feature])

    # join data after scaling
    X_train_smote_stand = pd.concat([drop_smote_train, feature_smote_train], axis=1)
    X_test_smote_stand = pd.concat([drop_smote_test, feature_smote_test], axis=1)

    # test data
    assert X_train_smote_stand.shape[1] == 14
    assert X_test_smote_stand.shape[1] == 14