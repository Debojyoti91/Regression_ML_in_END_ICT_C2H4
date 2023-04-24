from krr import *    #importing the model (Model should be the one is being implemented *** one should notice that a model(function) is called as the name of the python file without the .py extention)
from plot import *

#data preparation
data = pd.read_csv("../data/c2h4_rainbow_scat_data.csv") #the dataset contains 120 data points (initial one)
#data = pd.read_csv("../c2h4_data_smogn_final.csv") #Smogn data are already generated
##data = pd.read_csv("c2h4_mull_pop_phi1.csv") #use this datasets for Mulliken Population prediction
data = data.reset_index()
data = data.drop(columns=['index'])
features = ['Alpha', 'Beta', 'Gamma']  #these 3 are inputs for scatt. angle and impact parameters
#features = ['Alpha', 'Beta', 'Gamma', 'b']  #these 3 are inputs for Mulliken population
#target = ['theta_pr','b_pr'] #these  are targets for scatt. angle and impact parameters
#target = ['Mull_pop_P'] # #use this target for Mulliken Population prediction 
X_data = data[features].values.reshape(-1, len(features))

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
