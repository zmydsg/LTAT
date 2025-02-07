import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from scipy.special import erfc
from Model import ReplayBuffer
from TD3_model import TD3
from Subgradient import subgradient

def Q_fun(x):
    return erfc(x / 2**0.5) * 0.5

def epsilon_m(m, g, gamma, R, l):
    C_sum = 0
    V_sum = 0
    #print(l)
    for i in range(m + 1):
        C_sum += np.log2(1 + gamma[i] * g[i])
        V_sum += 1.0 - 1.0 / ((1.0 + gamma[i] * g[i])**2)
        #V_sum = 1
        #print(gamma[i], g[i])
    if V_sum == 0:
        #print(g)
        #print(gamma)
        return 1.0
    return Q_fun((C_sum - R) / (np.log2(np.exp(1)) * np.sqrt(V_sum / l)))

def main(bar_Power, bar_BLER):
    K = 100.0 # K个信息
    M = 5# 重传M次
    l = 50.0 # 码长
    R = 5 #传输速率

    # Initialize DDPG agent
    state_dim = 1
    action_dim = 1
    max_action = 100000
    
    agent = TD3(state_dim, action_dim, max_action)
    
    #file = '1'
    flag = 'train'
    if flag == 'train':
        # Initialize replay buffer
        buffer_size = 2000
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        next_state = np.zeros(state_dim)
        batch_size = 128
        
        # Train agent
        T = 100000000 # T次传输

        throughput_sum = 0 # 吞吐量
        Power_sum = 0 # 计算前t次的传输功率总和
        indicator_fail_sum = np.zeros(M) # 记录第m次传输失败的总和次数
        N = 0

        Lambda = 1.0 # 功率约束变量
        mu = 1.0 # BLER约束变量
        
        g = np.random.exponential(size = M)
        m = 0 # m:0 ~ M-1代表第1~M次传输
        next_state = np.random.uniform(size = 1) * 10000
        #Times = 0
        for t in range(1, T+1):
            #Times += 1
            #if t % 100 ==0:
            #    print(t)

            state = next_state
            action = agent.select_action(state) # 选择功率
            reward = -abs(action - Q_fun(state))
            next_state = np.random.uniform(size = 1)
            done = 1
            print(reward)
            replay_buffer.add(state, next_state, action, reward, done) # 存储长期平均吞吐量而不是奖励
            
            
            # Train agent
            if len(replay_buffer) > batch_size:
                iterations = 100
                agent.train(replay_buffer, iterations)

                
                
            
        return T

if __name__ == "__main__":
    x = main(100, 0.01)


