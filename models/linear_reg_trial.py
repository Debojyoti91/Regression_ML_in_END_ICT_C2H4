#linear regression model
def linear_reg():
    for t in range(0, len(target)):
         y_data = data[target[t]].values
         y_data_array.append(y_data)
         X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_array[t], test_size=0.20, random_state=1)
         X_test_array.append(X_test)
         y_train_array.append(y_train)
         y_test_array.append(y_test)
         ols = linear_model.LinearRegression()  #linear model for ordinary least square model
         model = ols.fit(X_train, y_train_array[t])
         model_fit_array.append(model)
         y_hat = model.predict(X_test)
         y_hat_array.append(y_hat)
         MAE = mae(y_test,y_hat).round(2)
         MAE_array.append(MAE)
         mse_test = sqrt(mse(y_test,y_hat).round(2))
         MSE_array.append(mse_test)
         r_2 = r2_score(y_test,y_hat)
         r2_array.append(r_2)
         pear_r = pearsonr(y_test,y_hat)[0]
         pear_coeff.append(pear_r)

