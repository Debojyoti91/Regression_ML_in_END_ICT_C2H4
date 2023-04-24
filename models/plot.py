import numpy as np
import matplotlib.pyplot as plt

def plot_scat_angle(array1, array2):
    target_prop = ['Scattering Angle, \u03B8 (degrees)', 'Impact Parameter, b (a.u.)']
    x = np.arange(0, len(array1))  #generates_an_array_of_values_each_number_corresponding_to_an_orientation
    plt.plot(x, array1, marker='o', color="r", label=" actual data")
    plt.plot(x, array2, marker='o',  color="black", label="ML prediction")
    plt.ylim(0, 16)
    plt.title('%s vs X for SMOGN dataset $\u03C6_{3}$ (KRR)' %target_prop[0])  #change the title of the plot if needed
    #plt.title('x vs Mulliken population')
    #plt.xticks(x)
    plt.xlabel('X (numerical label of an orientation)')
    plt.ylabel('%s' %target_prop[0])
    #plt.ylabel('Mulliken population')
    plt.legend()
    #plt.savefig("plot_krrreg_120_for_phi1_%s" %target[0])     #this is to save the figure
    #plt.savefig("plot_krrreg_smogn_theta_for_phi3_%s" %target[0])
    #plt.savefig("plot_krrreg_mul_pop_for_phi1_%s" %target[p])
    #plt.savefig("plot_krrreg_smogn_for_phi1_mull_pop")
    plt.show()


def plot_imp_para(array1, array2):
    target_prop = ['Scattering Angle, \u03B8 (degrees)', 'Impact Parameter, b (a.u.)']
    x = np.arange(0, len(array1))  #generates_an_array_of_values_each_number_corresponding_to_an_orientation
    plt.plot(x, array1, marker='o', color="r", label=" actual data")
    plt.plot(x, array2, marker='o',  color="black", label="ML prediction")
    plt.ylim(0, 9.0)
    plt.title('%s vs X for SOMGN dataset $\u03C6_{3}$ (KRR)' %target_prop[1])
    #plt.title('x vs Mulliken population')
    plt.xlabel('X (numerical label of an orientation)')
    plt.ylabel('%s' %target_prop[1])
    #plt.ylabel('Mulliken population')
    plt.legend()
    #plt.savefig("plot_krrreg_120_for_phi1_%s" %target[1])
    #plt.savefig("plot_krrreg_smogn_bpr_for_phi3_%s" %target[1])
    #plt.savefig("plot_krrreg_mul_pop_for_phi1_%s" %target[p])
    #plt.savefig("plot_krrreg_smogn_for_phi1_mull_pop")
    plt.show() 
