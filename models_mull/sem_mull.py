#Written by Debojyoti Das, TTU                                                                                                                                    

#import the python libraries
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn.ensemble import StackingRegressor
from numpy import dstack
import random as rn
PYTHONHASHSEED=0
np.random.seed(1234)
rn.seed(1254)

y_data_array = []
X_test_array = []
y_train_array = []
y_test_array = []
model_fit_array = []
y_predicted = []
MAE_array = []
MSE_array = []
r2_array = []
pear_coeff = []

target = ['Mull_pop_P'] # #use this target for Mulliken Population prediction

#stacked ensemble model
def sem_mull(data):
    features = ['Alpha', 'Beta', 'Gamma', 'b']
    X_data = data[features].values.reshape(-1, len(features))
    #kernel_array = ["linear", "rbf","laplacian", "chi2"]
    for t in range(0, len(target)):
        y_data = data[target[t]].values
        y_data_array.append(y_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_array[t], test_size=0.20, random_state=1)
        X_test_array.append(X_test)
        y_train_array.append(y_train)
        y_test_array.append(y_test)
        level0 = list()
        level0.append(('knn', KNeighborsRegressor()))
        level0.append(('cart', DecisionTreeRegressor()))
        level0.append(('svm', SVR()))
        level0.append(('KRR', KernelRidge(alpha=1.0)))
        level0.append(('XGB', xgb.XGBRegressor()))
        level0.append(('ExtraTree', ExtraTreesRegressor(n_estimators = 20, random_state = 0)))
        level1 = ExtraTreesRegressor(n_estimators = 20, random_state = 0)
        # define the stacking ensemble
        model_stack = StackingRegressor(estimators=level0, final_estimator=level1, cv=10)
        # fit the model on all available data
        model_stack.fit(X_train, y_train)
        # make a prediction for one example
        y_hat = model_stack.predict(X_test) #prediction_from_testing_dataset
        y_predicted.append(y_hat)
        MSE = sqrt(mean_squared_error(y_test,y_hat).round(2))
        MAE = mae(y_test,y_hat).round(2)
        r_2 = r2_score(y_test,y_hat)
        MAE_array.append(MAE)
        MSE_array.append(MSE)
        r2_array.append(r_2)
        pear_rcoeff = pearsonr(y_test,y_hat)[0]
        pear_coeff.append(pear_rcoeff)



