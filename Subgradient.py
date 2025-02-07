import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from scipy.special import erfc

def subgradient_optimization(objective_function, gradient_function, x0, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER, step_size=0.1, num_iterations=1):
    x = np.array(x0, dtype=float)

    for i in range(num_iterations):
        g = np.array(gradient_function(x, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER))
        x[0] -= step_size * g[0] * 0.1
        x[1] -= step_size * g[1] * 100
        x = np.maximum(x,0)
    
    f_opt = objective_function(x, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER)

    return x, f_opt

def objective_function(x, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER): # 目标函数也就是奖励
    Lambda = x[0]
    mu = x[1]
    #print(indicator_fail)
    return LTAT + Lambda * (bar_Power - LTA_Power) + mu * (bar_BLER - LTA_BLER)# 奖励

def gradient_function(x, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER):
    Lambda = x[0]
    mu = x[1]
    #print(bar_Power - LTA_Power, bar_BLER - LTA_BLER)
    #if T % 1000 == 0:
    #    print('Power_Constrain_TD:%.3f' % (bar_Power - P_sum / (T - fail_M_1)))
    #    print('BLER_Constrain_TD:%.3f' % (bar_BLER - fail_M / (T - fail_M_1)))
    return [bar_Power - LTA_Power, bar_BLER - LTA_BLER] # [dr/dLambda, dr/dmu]

def subgradient(Lambda, mu, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER):
    x0 = [Lambda, mu]
    x_opt, f_opt = subgradient_optimization(objective_function, gradient_function, x0, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER)
    #print("Optimized solution:", x_opt)
    #print("Optimal function value:", f_opt)
    return x_opt, f_opt
