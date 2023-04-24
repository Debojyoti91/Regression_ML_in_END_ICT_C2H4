#Written by Debojyoti Das, TTU                                                                                                                                    

#import python libraries
from pandas import read_csv                                                                                                                                       
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from math import sqrt
from sklearn.ensemble import ExtraTreesRegressor



y_data_array = []
X_test_array = []
y_train_array = []
y_test_array = []
model_fit_array = []
y_hat_array = []
MAE_array = []
MSE_array = []
r2_array = []
pear_coeff = []

target = ['theta_pr','b_pr']
#target = ['theta_pr','b_pr']

            
def ET_reg(data):
    features = ['Alpha', 'Beta', 'Gamma']
    X_data = data[features].values.reshape(-1, len(features))
    kernel_array = ["linear", "rbf","laplacian", "chi2"]
    for t in range(0, len(target)):
        y_data = data[target[t]].values
        y_data_array.append(y_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_array[t], test_size=0.20, random_state=1)
        X_test_array.append(X_test)
        y_train_array.append(y_train)
        y_test_array.append(y_test)
        ET = ExtraTreesRegressor(n_estimators=20, random_state=0)
        model = ET.fit(X_train, y_train_array[t])
        y_hat = model.predict(X_test)
        y_hat_array.append(y_hat)
        MSE = sqrt(mean_squared_error(y_test,y_hat).round(2))
        MAE = mae(y_test,y_hat).round(2)
        r_2 = r2_score(y_test,y_hat)
        MAE_array.append(MAE)
        MSE_array.append(MSE)
        r2_array.append(r_2)
        pear_rcoeff = pearsonr(y_test,y_hat)[0]
        pear_coeff.append(pear_rcoeff)
