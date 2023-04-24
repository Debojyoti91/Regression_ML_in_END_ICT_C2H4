#Written by Debojyoti Das, TTU  

#import python libraries
from pandas import read_csv                                                                                                                                       
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from math import sqrt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV



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


params = {
    "colsample_bytree": np.arange(0.01, 1, 100),                                                                                                                        
    "gamma": np.arange(0.01, 1, 100),
    "learning_rate": np.arange(0.03, 0.3, 100), # default 0.1
    "max_depth": np.arange(2, 6), # default 3
    "n_estimators": np.arange(100, 150), # default 100
    "subsample": np.arange(0.2, 0.4, 10)
}

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

xgb_model = xgb.XGBRegressor()

target = ['theta_pr','b_pr']

#xgboost model
def xgboost(data):
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
        search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1, n_jobs=1, return_train_score=True)
        model = search.fit(X_train, y_train_array[t])
        report_best_scores(model.cv_results_, 1)
        model_fit_array.append(model)
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
