import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots
from matplotlib.lines import Line2D
def DDPG_get_throughput(bar_Power_dB, R):
    file_name = "DDPG-" + str(bar_Power_dB) + 'dB-0.01-R='+str(R)+'-LTAT_LTABLER.txt'
    save_file_jpg = str(bar_Power_dB) + 'dB.jpg'
    txt_array = np.loadtxt(file_name, delimiter=' ')
    #print(txt_array)
    
    return txt_array[-2][0]

def GP_throught(R):
    file_name = 'R=' + str(R) + ',GP_throughput_FB.txt'
    arr = np.loadtxt(file_name)
    power_dB_arr = []
    throughput_arr = []
    for _, power_dB, throughput,_ in arr:
        power_dB_arr.append(int(power_dB))
        throughput_arr.append(throughput)
    return power_dB_arr, throughput_arr

def GP_throught_Shannon(R):
    file_name = 'R=' + str(R) + ',GP_throughput_Shannon.txt'
    arr = np.loadtxt(file_name)
    power_dB_arr = []
    throughput_arr = []
    for _, power_dB, throughput,_ in arr:
        power_dB_arr.append(int(power_dB))
        throughput_arr.append(throughput)
    return power_dB_arr, throughput_arr

if __name__ == "__main__":
    #file_name = 'train_result.txt'
    R = 3
    DDPG_throughput_R3 = []
    GP_power_dB_R3, GP_throughput_R3 = GP_throught(R)
    GP_power_dB_R3_Shannon, GP_throughput_R3_Shannon = GP_throught_Shannon(R)
    DDPG_power_dB_R3 = [i for i in range(9, 41)]
    for constrain_Power in DDPG_power_dB_R3:
        DDPG_throughput_R3.append(DDPG_get_throughput(constrain_Power, R))

    R = 5
    DDPG_throughput_R5 = []
    GP_power_dB_R5, GP_throughput_R5 = GP_throught(R)
    GP_power_dB_R5_Shannon, GP_throughput_R5_Shannon = GP_throught_Shannon(R)
    DDPG_power_dB_R5 = [i for i in range(13, 41)]
    for constrain_Power in DDPG_power_dB_R5:
        DDPG_throughput_R5.append(DDPG_get_throughput(constrain_Power, R))
    
    plt.style.use(['science', 'ieee', 'grid'])
    
    plt.plot(DDPG_power_dB_R3, DDPG_throughput_R3, color = 'black', linestyle = ':', label=r"DRL-$R=3$ bps/Hz")
    plt.plot(GP_power_dB_R3, GP_throughput_R3, color = 'red', linestyle = '-.',label=r"GP-$R=3$ bps/Hz")
    #plt.plot(GP_power_dB_R3_Shannon, GP_throughput_R3_Shannon, color = 'blue', linestyle = ' ', marker = 'x', markersize = 3, label=r"GP(Shannon)-$R=3$ bps/Hz")
    #plt.text(35, 2.6, r'$R=3$')
    #plt.plot(DDPG_power_dB_R4, DDPG_throughput_R4, color = 'blue', marker = '^', markersize = 3,linestyle = ':',)
    #plt.plot(GP_power_dB_R4, GP_throughput_R4, color = 'blue', linestyle = '-')
    #plt.text(35, 3.6, r'$R=4$')
    plt.plot(DDPG_power_dB_R5, DDPG_throughput_R5, color = 'black',linestyle = '--', label=r"DRL-$R=5$ bps/Hz")
    plt.plot(GP_power_dB_R5, GP_throughput_R5, color = 'red', linestyle = '-', label=r"GP-$R=5$ bps/Hz")
    #plt.plot(GP_power_dB_R5_Shannon, GP_throughput_R5_Shannon, color = 'blue', linestyle = ' ', marker = 'x', markersize = 3, label=r"GP(Shannon)-$R=5$ bps/Hz")
    #plt.text(35, 4.6, r'$R=5$')
    '''
    custom_legend = [
    Line2D([0], [0], linestyle = '-', color = 'black', label='GP'),
    #Line2D([0], [0], marker='', linestyle = '', color='blue', label=r'\text{DRL}', markersize = 3),
    Line2D([0], [0], marker='^', linestyle = '', color='green', label=r'DRL($R=5$)', markersize=3),
    Line2D([0], [0], marker='x', linestyle = '', color='blue', label=r'DRL($R=4$)', markersize = 3),
    Line2D([0], [0], marker='o', linestyle = '', color='red', label=r'DRL($R=3$)', markersize = 3),
    ]
    '''
    #plt.legend(handles=custom_legend, fontsize='small')
    plt.legend(loc = 'best', fontsize=6)
    plt.xlim(9, 40)
    plt.xlabel(r'Maximum allowable power $\bar P$ [dB]')
    plt.ylabel(r'Maximum LTAT $\eta$ [bps/Hz]')
    plt.savefig('Throughput.jpg', dpi = 600, bbox_inches = 'tight')
    #save_array = np.array([Constrain_Power, LTAT])
    #np.savetxt(file_name, save_array, delimiter = ' ')
    #plt.show()

