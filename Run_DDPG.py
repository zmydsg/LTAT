import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from scipy.special import erfc
from Model import ReplayBuffer, DDPG
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
        
    return Q_fun((C_sum - R) / (np.log2(np.exp(1)) * np.sqrt(V_sum / l)))

def Enivorment(state, action, m, g, M, R, l):
    #power = np.zeros(M)
    state_dim = len(state)
    next_state = state.copy()

    power = np.append(state, 0)
    power[m] = action # 1~m次的功率

    # 判断本次传输是否成功
    epsilon = epsilon_m(m, g, power, R, l)
    v = np.random.uniform()
    Flag = 0 # 判断该数据包是否传输成功
    #if action == 0:
    #    v = 0
    if v <= epsilon: # 本次传输失败
        reward = 0
        if m == M - 1: # 达到最大传输次数也就是该数据包传输失败
            #print('Max')
            next_g = np.random.exponential(size = M)
            next_state = np.zeros(state_dim)
            next_m = 0
            done = True
            Flag = 1

        else: # 没有达到最大传输次数
            #print('no Max')
            next_g = g
            next_state[m] = action
            #print(next_state)
            next_m = m + 1
            done = False

    else: # 本次传输成功
        #print('sucess')
        reward = R
        next_g = np.random.exponential(size = M)
        next_state = np.zeros(state_dim)
        next_m = 0
        done = True

    return next_state, reward, done, next_g, next_m, Flag

def insert_element(trun_array, index, m, flag, W, M):
    #W = 1000
    trun_array[index] = (m, flag)
    #print(index)
    Average_BLER = np.zeros(M)
    for i in range(0, M):
        count = np.sum([elem == (i, 1) for elem in trun_array])
        Average_BLER[i] = count
    return Average_BLER

def main(bar_Power_dB, bar_BLER, R):
    
    train_times = 100000
    save_all = np.zeros((train_times + 1, 3))
    all_Power_sum = 0
    save_R_BLER_name = 'DDPG-W=100-' + str(bar_Power_dB) + 'dB-0.01-R=' + str(R) + '-LTAT_LTABLER.txt'
    
    K = 100.0 # K个信息
    M = 5# 重传M次
    save_fail = np.zeros(M+1)
    l = 50.0 # 码长
    #R = 5 #传输速率
    subgradient_time = 100
    bar_Power = 10**(bar_Power_dB * 0.1)
    # Initialize DDPG agent
    state_dim = M - 1
    action_dim = 1
    max_action = bar_Power*1.5
    W = 100 #截断的大小
    net_file = 'DDPG-' + str(bar_Power_dB) + 'dB-0.01-R=' + str(R)
    train_process_file = str(bar_Power_dB) + 'dB-0.01-R=' + str(R) + '.txt'
    out = open(train_process_file, 'w')
    agent = DDPG(state_dim, action_dim, max_action)
    #agent.load(net_file)
    bar_M = 1

    operation = 'train'
    if operation == 'train':
        trun_array = np.empty(W, dtype=object)  # 创建一个大小为3000的空数组
        index = 0
        # Initialize replay buffer
        buffer_size = 3000
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        next_state = np.zeros(state_dim)
        state = np.zeros(state_dim)
        batch_size = 32
        
        # Train agent
        throughput_array = np.zeros(W)
        power_array = np.zeros(W)
        throughput_sum = 0 # 吞吐量
        all_Power_sum = 0
        all_throughput_sum = 0
        #Power_sum = 0 # 计算前t次的传输功率总和
        #indicator_fail_sum = np.zeros(M) # 记录第m次传输失败的总和次数
        N = 0

        Lambda = 0 # 功率约束变量
        mu = 0 # BLER约束变量
        #Power_500 = np.zerors(500)
        g = np.random.exponential(size = M)
        
        next_m = 0
        m = 0
        t = 1
        while t <= train_times:
            
            state = next_state
            m = next_m
            action = agent.select_action(state)[0] # 选择功率
            
            #if m == 0:
            #    action = np.clip(action, 0, bar_Power)
            #else:
            #    action = np.clip(action, 0, 2 * bar_Power - sum(state))
            # TD3变量
            next_state, reward, done, g, next_m, Flag = Enivorment(state, action, m, g, M, R, l)
            
            # 次梯度变量
            throughput_array[t%W] = reward
            power_array[t%W] = action
            #print(W, t, power_array[t%W], throughput_array[t%W])
            #Power_sum += action
            #throughput_sum += reward
            if reward == 0:
                indicator_fail_sum = insert_element(trun_array, index, m, 1, W, M)
                save_fail[m] += 1
            else:
                indicator_fail_sum = insert_element(trun_array, index, m, 0, W, M)
            index = (index + 1) % W  # 使用取余运算实现循环存储
            Temp_Power = action
            T = min(W, t)
            N = T - np.sum(indicator_fail_sum[:M-1])
            if N!=0:
                bar_M = T/N

            all_Power_sum += action 
            all_throughput_sum += reward
            save_all[t-1][0] = all_throughput_sum / t
            save_all[t-1][1] = all_Power_sum / (t - np.sum(save_fail[:M-1]))
            if t > 5:
                save_all[t-1][2] = save_fail[M-1] / (t - np.sum(save_fail[:M-1]))
            reward += Lambda * (bar_Power - Temp_Power * bar_M) + mu * (bar_BLER- Flag * bar_M)
            action = np.array([action])
            reward = np.array([reward])
            done = np.array([done])

            replay_buffer.add(state, next_state, action, reward, done) # 存储长期平均吞吐量而不是奖励

            #loss = 0
            # Train agent
            if len(replay_buffer) > batch_size:
                iterations = 10
                agent.train(replay_buffer, iterations)

                #if reward >= 0: # 强化学习训练好了
                if t % subgradient_time == 0:

                    #T = (t - 1) % times + 1
                    #N = T - sum(indicator_fail_sum[:M-1])
                    LTA_Power = np.sum(power_array) / N
                    LTA_BLER = indicator_fail_sum[M-1] / N
                    LTAT = np.sum(throughput_array) / T
                    #bar_M = 
                    #replay_buffer = ReplayBuffer(state_dim, action_dim)
                    x, _ = subgradient(Lambda, mu, LTAT, LTA_Power, LTA_BLER, bar_Power, bar_BLER)
                    Lambda, mu = x
                    #if LTA_Power < bar_Power:
                    #    Lambda = 0
            #print('ok')
            #print(reward, Flag, m)
            if t % subgradient_time == 0:
                #T = (t - 1) % times + 1
                #N = T - sum(indicator_fail_sum[:M-1])
                LTA_Power = np.sum(power_array) / N
                LTA_BLER = indicator_fail_sum[M-1] / N
                LTAT = np.sum(throughput_array) / T
                print('t:%d' % t)
                print('index:%d' % index)
                print('Power:%.3f' % action)
                print('Lambda:%.5f' % Lambda)
                print('mu:%.5f' % mu)
                print('g:' + str(g))
                print('m:%d' % m)
                print('next_m:%d' % next_m)
                print('state:' + str(state))
                print('next_state:' + str(next_state))
                #print(str(LTA_BLER))
                print('bar_M:%.3f' % bar_M)
                print('trun_BLER:%.5f' % LTA_BLER)
                print('trun_Power:%.3f' % LTA_Power)
                print('bar_Power:%.3f' % bar_Power)
                print('bar_Power_dB:%.3f' % bar_Power_dB)
                print('trun_throughput:%.3f' % (LTAT))
                print('reward:%.3f' % reward)
                print('LTA-Power:%.5f'% save_all[t-1][1])
                print('LTAT:%.5f' % (all_throughput_sum / t))
                #print('loss:%.5f' % loss)
                print()
                
                agent.save(net_file)
                output_t = str(t) + ' '
                output_LTAT = str(LTAT) + ' '
                output_BLER = str(LTA_BLER) + ' '
                output_Power = str(LTA_Power) + '\n'
                #print(str(output_LTAT))
                out.writelines(output_t)
                out.writelines(output_LTAT)
                out.writelines(output_BLER)
                out.writelines(output_Power)
                out.flush()
            t += 1 # 现在进行第t轮传输
        np.savetxt(save_R_BLER_name, save_all)

if __name__ == "__main__":
    #Bar_Power = 10**(30*0.1)
    #x = 21
    #while x <= 40:
    
    #R_arr = [3, 4, 5]
    #for R in R_arr:
    #for x in range(9, 41):
    #main(16, 0.01, 3)
    
    main(20, 0.01, 5)
