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
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from math import sqrt
import random as rn
PYTHONHASHSEED=0
np.random.seed(1234)
rn.seed(1254)

def ordinary_least_squares(X, y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_hat

def make_design_matrix(x, order):
  """Create the design matrix of inputs for use in polynomial regression
  Args:                                                                                                                                                         
    x (ndarray): input vector of shape (samples,)
    order (scalar): polynomial regression order
  Returns:
    ndarray: design matrix for polynomial regression of shape (samples, order+1)
  """

  # Broadcast to shape (n x 1) so dimensions work
  if x.ndim == 1:
    x = x[:, None]

  #if x has more than one feature, we don't want multiple columns of ones so we assign
  # x^0 here
  design_matrix = np.ones((x.shape[0], 1))

  # Loop through rest of degrees and stack columns (hint: np.hstack)
  for degree in range(1, order + 1):
      design_matrix = np.hstack((design_matrix, x**degree))

  return design_matrix

def solve_poly_reg(x, y, max_order):
  """Fit a polynomial regression model for each order 0 through max_order.
  Args:
    x (ndarray): input vector of shape (n_samples)
    y (ndarray): vector of measurements of shape (n_samples)
    max_order (scalar): max order for polynomial fits
  Returns:
    dict: fitted weights for each polynomial model (dict key is order)
  """

  # Create a dictionary with polynomial order as keys,
  # and np array of theta_hat (weights) as the values
  theta_hats = {}

  # Loop over polynomial orders from 0 through max_order
  for order in range(1, (max_order + 1)):
    print(order)
    # Create design matrix
    X_design = make_design_matrix(x, order)

    # Fit polynomial model
    this_theta = ordinary_least_squares(X_design, y)

    theta_hats[order] = this_theta

  return theta_hats

max_order = 5

mae_list = []
mse_list = []
r2_list = []
pear_coeff = []
y_hat_array = []
order_list = list(range(1, (max_order + 1)))
order_list_array = []
y_predicted = []
order_min = []
y_data_array = []
X_test_array = []
y_train_array = []
y_test_array = []
theta_hat_array = []

target = ['theta_pr','b_pr']

#polynomial regression model
def poly_reg(data):
    features = ['Alpha', 'Beta', 'Gamma']  #these 3 are inputs
    X_data = data[features].values.reshape(-1, len(features))
    for t in range(0, len(target)):
        y_data = data[target[t]].values
        y_data_array.append(y_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_array[t], test_size=0.20, random_state=1)
        X_test_array.append(X_test)
        y_train_array.append(y_train)
        y_test_array.append(y_test)
        theta_hats = solve_poly_reg(X_train, y_train_array[t], max_order)
        theta_hat_array.append(theta_hats)
        y_hat_array_list = []
        mae_list_array = []
        mse_list_array = []
        r2_list_array = []
        pear_r = []
        for order in order_list:
            X_design = make_design_matrix(X_test, order)
            y_hat = X_design @ theta_hat_array[t][order]
            y_hat_array_list.append(y_hat)
            mse_test = sqrt((np.std((y_test_array[t] - y_hat)**2)).round(2))
            mae_test = mae(y_test, y_hat).round(2)
            mae_list_array.append(mae_test)
            mse_list_array.append(mse_test)
            r_2 = r2_score(y_test,y_hat)
            r2_list_array.append(r_2)
            pear_rcoeff = pearsonr(y_test,y_hat)[0]
            pear_r.append(pear_rcoeff)
        mae_list.append(mae_list_array)
        mse_list.append(mse_list_array)
        r2_list.append(r2_list_array)
        pear_coeff.append(pear_r)
        y_hat_array.append(y_hat_array_list)
        order = order_list
        order_list_array.append(order)
        res = dict(zip(order_list_array[0], mae_list[t]))
        for k, v in res.items():
            if v == min(mae_list[t]):
               m = k-1
               order_min.append(k)
               y_predicted.append(y_hat_array[t][m])
