import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle
import matplotlib.ticker as ticker
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



def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.1,y_ratio=0.1):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    #print(zone_left)
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right+1] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)


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
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.plot(DDPG_power_dB_R5, DDPG_throughput_R5, color = 'black', linestyle = ':', label=r"DRL-$R=5$ bps/Hz")
    ax.plot(GP_power_dB_R5, GP_throughput_R5, color = 'red', linestyle = '-', label=r"GP-$R=5$ bps/Hz")
    ax.plot(GP_power_dB_R5_Shannon, GP_throughput_R5_Shannon, color = 'blue', linestyle = ' ', marker = 'x', markersize = 3, label=r"GP(Shannon)-$R=5$ bps/Hz")
    ax.plot(DDPG_power_dB_R3, DDPG_throughput_R3, color = 'black', linestyle = ':', label=r"DRL-$R=3$ bps/Hz")
    ax.plot(GP_power_dB_R3, GP_throughput_R3, color = 'red', linestyle = '-', label=r"GP-$R=3$ bps/Hz")
    ax.plot(GP_power_dB_R3_Shannon, GP_throughput_R3_Shannon, color = 'blue', linestyle = ' ', marker = 'x', markersize = 3, label=r"GP(Shannon)-$R=3$ bps/Hz")
    
    ax.legend(loc = 'lower right', fontsize=6)
    #ax.set_xlim(15, 40)
    #plt.legend(handles=custom_legend, fontsize='small')
    #print(GP_throughput_R5[1], GP_throughput_R5_Shannon[1])
    axins = ax.inset_axes((0.65, 0.58, 0.3, 0.3))
    axins.plot(GP_power_dB_R5, GP_throughput_R5, color = 'red', linestyle = '-')
    axins.plot(GP_power_dB_R5_Shannon, GP_throughput_R5_Shannon, color = 'blue', linestyle = ' ', marker = 'x', markersize = 3)
    zone_and_linked(ax, axins, 0, 1, GP_power_dB_R5, [GP_throughput_R5, GP_throughput_R5_Shannon], 'top')
    #axins.set_xlim(GP_power_dB_R5[0]-0.1, GP_power_dB_R5[1]+0.1)
    #axins.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #axins.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    ax.set_xlabel(r'Maximum allowable power $\bar P$ [dB]')
    ax.set_ylabel(r'Maximum LTAT $\eta$ [bps/Hz]')
    plt.savefig('Throughput_revised.jpg', dpi = 600, bbox_inches = 'tight')
    #save_array = np.array([Constrain_Power, LTAT])
    #np.savetxt(file_name, save_array, delimiter = ' ')
    #plt.show()

