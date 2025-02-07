import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots
    
if __name__ == "__main__":
    #file_name = 'train_result.txt'
    file1 = r'6dB-0.01-R=5.txt'
    file2 = r'non-win-10dB-0.01-R=5.txt'
    improve = np.loadtxt(file1, dtype=np.float32, delimiter=' ')
    non_improve = np.loadtxt(file2, dtype=np.float32, delimiter=' ')
    bar_Power = 10**(6*0.1)
    t = []
    LTA_Power = []
    LTA_Power_Compare = []
    
    LTA_BLER = []
    LTA_BLER_Compare = []
    
    for a, b, c, d in improve:
        t.append(a)
        LTA_Power.append(d)
        LTA_BLER.append(c)
        
    for a, b, c, d in non_improve:
        LTA_Power_Compare.append(d)
        LTA_BLER_Compare.append(c)
    '''
    plt.style.use(['science', 'ieee', 'grid'])
    plt.plot(t, LTA_BLER, color = 'blue', linestyle = '-',markersize = 2,markevery = 50, label = r'Trun.')
    plt.plot(t, LTA_BLER_Compare, color = 'red',marker = '+',markersize = 5, linestyle = '--',markevery = 50, label = r'Non-Trun.')
    plt.axhline(0.01, color = 'black', linestyle = '--', label = r'Constraint')
    plt.legend(loc = 'best')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'LTA BLER')
    plt.savefig('LTA_BLER_compare.jpg', dpi = 600, bbox_inches = 'tight')
    '''
    plt.style.use(['science', 'ieee', 'grid'])
    plt.plot(t, LTA_Power, color = 'blue', linestyle = '-',markersize = 2,markevery = 50, label = r'Trun.')
    #plt.plot(t, LTA_Power_Compare, color = 'red',marker = '+',markersize = 5, linestyle = '--',markevery = 50, label = r'Non-Trun.')
    plt.axhline(bar_Power, color = 'black', linestyle = '--', label = r'Constraint')
    plt.legend(loc = 'best')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'LTA Power')
    plt.savefig('LTA_Power_compare.jpg', dpi = 600, bbox_inches = 'tight')
    
    #save_array = np.array([Constrain_Power, LTAT])
    #np.savetxt(file_name, save_array, delimiter = ' ')

