import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots

def get_arr(fname):
    arr = np.loadtxt(fname)
    a = arr[:,0]
    b = arr[:,1]
    c = arr[:,2]
    b = 10*np.log10(b)
    return a[500:len(a)-1], b[500:len(a)-1], c[500:len(a)-1]


if __name__ == "__main__":
    #file_name = 'train_result.txt'
    file_name_non = "non-win-DDPG-20dB-0.01-R=5-LTAT_LTABLER.txt"
    file_name = 'DDPG-W=300-20dB-0.01-R=5-LTAT_LTABLER.txt'
    file_name_3000 = 'DDPG-W=3000-20dB-0.01-R=5-LTAT_LTABLER.txt'
    file_name_100 = 'DDPG-W=100-20dB-0.01-R=5-LTAT_LTABLER.txt'
    #file_name_1000 = 'DDPG-W=1000-13dB-0.01-R=5-LTAT_LTABLER.txt'
    T = [i for i in range(501, 100000+1)]
    non_LTAT, non_power, non_BLER = get_arr(file_name_non)
    LTAT, power, BLER = get_arr(file_name)
    LTAT_3000, power_3000, BLER_3000 = get_arr(file_name_3000)
    LTAT_100, power_100, BLER_100 = get_arr(file_name_100)
    #LTAT_1000, power_1000, BLER_1000 = get_arr(file_name_1000)
    #print(len(LTAT), len(non_LTAT))
    power_constrain = [20 for i in range(501, 100000+1)]
    BLER_constrain = [0.01 for i in range(501, 100000+1)]
    LTAT_bound = [3.6175246 for i in range(501, 100000+1)]
    plt.style.use(['science', 'ieee', 'grid'])
    #'''
    plt.plot(T, non_LTAT, color = 'blue', label = r'No Trun.', marker = 'x', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, LTAT_3000, color = 'green', linestyle = '-', marker = '^', label = r'Trun.-$W=3000$', markevery =3000, markersize = '3')
    plt.plot(T, LTAT, color = 'red', label = r'Trun.-$W=300$', marker = 'o', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, LTAT_100, color = 'black', linestyle = '-', marker = '>', label = r'Trun.-$W=100$', markevery =3000, markersize = '3')
    plt.plot(T, LTAT_bound, color = 'black', linestyle = '--', label = r'GP')
    plt.legend(loc = 'best')
    plt.xlim(0,100000)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'LTAT $\eta$ [bps/Hz]')
    plt.savefig('LTAT_compare.jpg', dpi = 600, bbox_inches = 'tight')
    #'''

    '''
    plt.plot(T, non_power, color = 'blue', label = r'No Trun.', marker = 'x', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, power_3000, color = 'green', label = r'Trun.-$W=3000$', marker = '^', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, power, color = 'red', label = r'Trun.-$W=300$', marker = 'o', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, power_100, color = 'black', label = r'Trun.-$W=100$', marker = '>', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, power_constrain, color = 'black', linestyle = '--', label = r'Bound')
    plt.legend(loc = 'best')
    plt.xlim(0,100000)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Average total power [dB]')
    plt.savefig('LTA_Power_compare.jpg', dpi = 600, bbox_inches = 'tight')
    '''

    '''
    plt.plot(T, non_BLER, color = 'blue', label = r'No Trun.', marker = 'x', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, BLER_3000, color = 'green', label = r'Trun.-$W=3000$', marker = '^', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, BLER, color = 'red', label = r'Trun.', marker = 'o', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, BLER_100, color = 'black', label = r'Trun.-$W=100$', marker = '>', markevery=3000, markersize = '3', linestyle = '-')
    plt.plot(T, BLER_constrain, color = 'black', linestyle = '--', label = r'Bound')
    plt.legend(loc = 'best')
    plt.xlim(0,100000)
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Average BLER')
    plt.savefig('LTA_BLER_compare.jpg', dpi = 600, bbox_inches = 'tight')
    '''
