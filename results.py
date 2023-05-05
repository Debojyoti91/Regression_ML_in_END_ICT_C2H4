from krr import *    #importing the model (Model should be the one is being implemented *** one should notice that a model(function) is called as the name of the python file without the .py extention)
from plot import *

#data preparation
data = pd.read_csv("../data/c2h4_rainbow_scat_data.csv") #the dataset contains 120 data points (initial one)

krr(data)

std_data = []
std_predict = []

for y in range(0, len(target)):
    std = np.std(y_data_array[y])
    std_yhat = np.std(y_hat_array[y])
    std_data.append(std)
    std_predict.append(std_yhat)

deviations = pd.DataFrame({'Targets' : target, 'std_data' : std_data, 'std_prediction' : std_predict})
print(deviations)

#this section is for svr and krr models
results = pd.DataFrame({'Targets' : target, 'Kernel' : kernel_list_array,  'MAE' : MAE_array,  'RMSE' : MSE_array, "r_2" : r2_array, 'pear_coeff' : pear_coeff})
print(results)

#this section is for polynomial regression
#results = pd.DataFrame({'Targets' : target, 'Orders' : order_list_array,  'MAE' : mae_list, 'MSE': mse_list, 'r_2' : r2_list, 'pear_coeff' : pear_coeff})
#print(results)

# for all other models
#results = pd.DataFrame({'targets' : target, 'MAE' : MAE_array, 'MSE' : MSE_array, 'r_2' : r2_array, 'pear_coeff' : pear_coeff})
#print(results)

plot_scat_angle(y_test_array[0], y_predicted[0])
plot_imp_para(y_test_array[1], y_predicted[1])

#Plotting Mulliken population
#plot_mull_pop(y_test_array[0], y_predicted[0])
