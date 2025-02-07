import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from scipy.special import erfc

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        
        self.layer_2 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        
        self.layer_3 = nn.Linear(32, action_dim)
        nn.init.normal_(self.layer_3.weight, 0., 0.1)
        nn.init.constant_(self.layer_3.bias, 0.1)
        
        self.max_action = torch.tensor(max_action)
    
    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        a = self.max_action * torch.sigmoid(self.layer_3(a))
        
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 64)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)
        
        self.layer_2 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)
        
        self.layer_3 = nn.Linear(32, 1)
        nn.init.normal_(self.layer_3.weight, 0., 0.1)
        nn.init.constant_(self.layer_3.bias, 0.1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.relu(self.layer_1(sa))
        q = torch.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, next_state, action, reward, done):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idx],
            self.next_state[idx],
            self.action[idx],
            self.reward[idx],
            self.done[idx]
        )
    def __len__(self):
        return self.size
'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 佛祖保佑       调参一次过
 '''

# Define the DDPG class
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-7)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-7)
        
        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
    def train(self, replay_buffer, iterations, batch_size=32, discount=0.9, tau=0.005):
        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)
            next_state = torch.FloatTensor(y)
            done = torch.FloatTensor(1 - d)
            reward = torch.FloatTensor(r)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
    
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))


