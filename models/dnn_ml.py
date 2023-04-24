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
from scipy.stats import pearsonr
from math import sqrt
from matplotlib import pyplot
from sklearn.ensemble import StackingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from numpy import dstack
from scipy.stats import pearsonr
import random as rn
PYTHONHASHSEED=0
tf.random.set_seed(1234)
np.random.seed(1234)
rn.seed(1254)

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

#NN model
def dnn(data):
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
        #NN = model()
        #NN_model = NN.fit(X_train, y_train_array[t], epochs=200, verbose=0)
        model5 = Sequential()
        model5.add(Dense(128,activation = 'relu',input_dim = 3))
        model5.add(Dense(64,activation = 'relu'))
        model5.add(Dense(50,activation = 'relu'))
        model5.add(Dense(25,activation = 'relu'))
        model5.add(Dropout(0.1))
        model5.add(Dense(10,activation = 'relu'))
        model5.add(Dense(1,activation = 'relu'))
        model5.compile(loss=tf.keras.losses.mae, optimizer='adam', metrics=['mae'])
        model5.fit(X_train, y_train, epochs=200, verbose=0)
        y_hat = model5.predict(X_test)
        y_hat_array.append(y_hat)
        MSE = sqrt(mean_squared_error(y_test,y_hat).round(2))
        MAE = mae(y_test,y_hat).round(2)
        r_2 = r2_score(y_test,y_hat)
        MAE_array.append(MAE)
        MSE_array.append(MSE)
        r2_array.append(r_2)
        pear_rcoeff = pearsonr(y_test,y_hat)[0]
        pear_coeff.append(pear_rcoeff)


