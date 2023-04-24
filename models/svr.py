#Written by Debojyoti Das, TTU                                                                                                                                    

#import the python libraries
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random as rn
np.random.seed(1234)
rn.seed(1254)


y_data_array = []
X_test_array = []
y_train_array = []
y_test_array = []
model_fit_array = []
y_hat_array = []
y_hat_array_train = []
y_predicted = []
MAE_array = []
MAE_array_train = []
MSE_array = []
MSE_array_train = []
kernel_list_array = []
kernels_list = []
r2_array = []
pear_coeff = []

target = ['theta_pr','b_pr']

def svr(data):
    features = ['Alpha', 'Beta', 'Gamma']
    X_data = data[features].values.reshape(-1, len(features))
    kernel_array = ["linear", "rbf", "sigmoid"]
    for t in range(0, len(target)):
        y_data = data[target[t]].values
        y_data_array.append(y_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_array[t], test_size=0.20, random_state=1)
        X_test_array.append(X_test)
        y_train_array.append(y_train)
        y_test_array.append(y_test)
        y_hat_array1 = []
        y_hat_array1_train = []
        mae_array1 = []
        mae_array1_train = []
        mse_array1 = []
        mse_array1_train = []
        r2_array_test = []
        pear_r = []
        for k in kernel_array:
            svr = GridSearchCV(
            SVR(kernel= k),
            param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, return_train_score=True
            )
            svr.fit(X_train, y_train)
            print(k, svr.best_params_)
            y_hat = svr.predict(X_test)
            y_hat_train = svr.predict(X_train)
            y_hat_array1.append(y_hat)
            y_hat_array1_train.append(y_hat_train)
            mae_test = mae(y_test,y_hat).round(2)
            mae_train = mae(y_train,y_hat_train).round(2)
            mae_array1.append(mae_test)
            mae_array1_train.append(mae_train)
            mse_test = sqrt(mse(y_test,y_hat).round(2))
            mse_train = (sqrt(mse(y_train,y_hat_train)))#.round(2)
            mse_array1.append(mse_test)
            mse_array1_train.append(mse_train)
            r_2 = r2_score(y_test,y_hat)
            r2_array_test.append(r_2)
            pear_rcoeff = pearsonr(y_test,y_hat)[0]
            pear_r.append(pear_rcoeff)
            y_hat_array.append(y_hat_array1)
        y_hat_array_train.append(y_hat_array1_train)
        MAE_array_train.append(mae_array1_train)
        MAE_array.append(mae_array1)
        MSE_array_train.append(mse_array1_train)
        MSE_array.append(mse_array1)
        r2_array.append(r2_array_test)
        pear_coeff.append(pear_r)
        kernels = kernel_array
        kernel_list_array.append(kernels)
        res = dict(zip(kernel_list_array[0], MAE_array[t]))
        for i, v in res.items():
            if v == min(MAE_array[t]):
                m = kernel_array.index(i)
                print(m)
                kernels_list.append(i)
                y_predicted.append(y_hat_array[t][m])




