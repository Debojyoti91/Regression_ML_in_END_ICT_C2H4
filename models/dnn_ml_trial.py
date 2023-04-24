from pandas import read_csv                                                                                                                                       
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd                                                                                                                                               
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.ensemble import StackingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from numpy import dstack
from scipy.stats import pearsonr
import random as rn
PYTHONHASHSEED=0
#tf.random.set_seed(1234)
tf.random.set_seed(1234)
np.random.seed(1234)
rn.seed(1254)

data = pd.read_csv("../c2h4_rainbow_scat_data.csv")
#data = pd.read_csv("c2h4_data_smogn_final.csv")

#data = data.sort_values(
#       by="theta_pr",
#       kind="mergesort"
#      )
#
#data = data.reset_index()
#
#data = data.drop(columns=['index'])

features = ['Alpha', 'Beta', 'Gamma']
#features = ['Alpha', 'Beta', 'Gamma', 'b']

X_data = data[features].values.reshape(-1, len(features))
y_data = data['theta_pr'].values
#y_data = data['b_pr'].values

#train_test_splitting
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=1)

def dnn():
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
    return y_hat
