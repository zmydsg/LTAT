import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.special import erfc


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
    if action == 0:
        v = 0
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

def select_action(state):
    #GP_action = [16.736843092240440,4.328125760203530,3.441638108862318,4.042963132103546,8.271524889638322] #15dB
    #GP_action = [29.196738576998932,7.657510399474178,5.270804774821871,3.878583895639333,1.824114121702676] #16dB
    #GP_action = [39.999650158084656,10.795511203655833,7.552300491995668,5.426777689888123,0.471091575981651] #17dB
    #GP_action = [53.242185517725670,14.643276835642043,10.411168507899912,7.715960005568470,0.133130511773947] #18dB
    #GP_action = [69.767040941092290,19.431982599197617,13.960470639395176,10.629042894871018,0.041495463875289] #19dB
    GP_action = [90.472256670235540, 25.423400063653162, 18.388516538192317, 14.282505403168178, 0.014007775187703] #20dB
    m = 0
    M = 5
    while state[m] != 0 and m < M - 2:
        m += 1
    return GP_action[m]

def main(bar_Power_dB, bar_BLER):
    K = 100.0 # K个信息
    M = 5# 重传M次
    l = 50.0 # 码长
    R = 5 #传输速率
    bar_Power = 10**(bar_Power_dB * 0.1)
    # Initialize DDPG agent
    state_dim = M - 1
    action_dim = 1
    max_action = bar_Power
    next_state = np.zeros(state_dim)
    
    operation = 'train'


    # Train agent
    throughput_sum = 0 # 吞吐量
    Power_sum = 0 # 计算前t次的传输功率总和
    indicator_fail_sum = np.zeros(M) # 记录第m次传输失败的总和次数
    N = 0

    Lambda = 0 # 功率约束变量
    mu = 0 # BLER约束变量
    
    g = np.random.exponential(size = M)
    
    next_m = 0
    m = 0
    t = 0
    while t < 300000:
        t += 1 # 现在进行第t轮传输
        state = next_state.copy()
        m = next_m
        action = select_action(state) # 选择功率
        
        if m == 0:
            action = min(action, bar_Power)
        
        # TD3变量
        next_state, reward, done, g, next_m, Flag = Enivorment(state, action, m, g, M, R, l)
        
        # 次梯度变量
        Power_sum += action
        throughput_sum += reward
        if reward == 0:
            indicator_fail_sum[m] += 1
        
        Temp_Power = action
        reward -= Lambda * Temp_Power + mu * Flag
        #print(m, Flag, reward, action)
        action = np.array([action])
        reward = np.array([reward])
        done = np.array([done])


        #loss = 0
        # Train agent
        #if len(replay_buffer) > batch_size:
        #    iterations = 100
        #    agent.train(replay_buffer, iterations)

            #if reward >= 0: # 强化学习训练好了
        #    if t % 500 == 0:

        #print('ok')
        #print(reward, Flag, m)
        if t % 10000 == 0:
            N = t - sum(indicator_fail_sum[:M-1])
            LTA_Power = Power_sum / N
            LTA_BLER = indicator_fail_sum[M-1] / N
            LTAT = throughput_sum / t
            print('t:%d' % t)
            print('Power:%.3f' % action)
            
            #print('Origin_Power:%.3f' % A)
            #print('Temp_Power:%.3f' % Temp_Power)
            print('Lambda:%.3f' % Lambda)
            print('mu:%.3f' % mu)
            print('g:' + str(g))
            print('m:%d' % m)
            print('next_m:%d' % next_m)
            print('state:' + str(state))
            print('next_state:' + str(next_state))
            print('LTA_BLER:%.5f' % LTA_BLER)
            print('LTA_Power:%.3f' % LTA_Power)
            print('LTAT:%.3f' % (throughput_sum / t))
            print('reward:%.3f' % reward)
            #print('loss:%.5f' % loss)
            print()
            #agent.save('DDPG-30dB')


if __name__ == "__main__":
    #Bar_Power = 10**(30*0.1)
    main(20, 0.01)

