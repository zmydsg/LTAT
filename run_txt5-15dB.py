import numpy as np
import matplotlib.pyplot as plt
#import scienceplots

def main(bar_Power_dB):
    file_name_dB = str(bar_Power_dB) + 'dB-BLER.txt'
    train_process_file = str(bar_Power_dB) + 'dB.txt'
    out = open(train_process_file, 'w')
    save_file_jpg = str(bar_Power_dB) + 'dB.jpg'
    txt_array = np.loadtxt(file_name_dB, dtype=np.float32, delimiter=' ')
    #print(txt_array)
    X = []
    Y = []
    Z = []
    BLER = 0
    for x, y, z in txt_array:
        output_t = str(x) + ' '
        output_LTAT = str(y) + '\n'
        out.writelines(output_t)
        out.writelines(output_LTAT)
        out.flush()
    out.close()
        #if x > 200000:
        #    BLER += z * 500
        #    average_BLER = BLER / (x - 200000)
        #    Z.append(average_BLER)
    #plt.style.use(['science', 'ieee', 'grid'])
    #plt.legend(loc = 'best')
    #plt.xlabel(str(bar_Power_dB) + 'dB')
    #plt.ylabel(r'Throughput $\eta$')
    #plt.savefig(save_file_jpg, dpi = 600, bbox_inches = 'tight')
    #plt.close()
    #print(Z[-1])
    
if __name__ == "__main__":
    
    for x in range(5, 15):
        main(x)
    

