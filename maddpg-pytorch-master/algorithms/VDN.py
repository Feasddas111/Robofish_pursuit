import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
from utils.buffer import ReplayBuffer
from gym.spaces import Box, Discrete
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Q_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        q = self.fc2(h)
        return q

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

class VDN:
    # ...
    def __init__(self, env, num_agents, state_size, action_size, seed, hidden_size=64, lr=0.01, gamma=0.99, batch_size=64):
        self.num_agents = num_agents
        self.nets = [Q_Network(state_size, hidden_size, action_size) if _ != 5 else Q_Network(18, hidden_size, action_size) for _ in range(num_agents)]  # 需要将agnet 3的input维度设置为14
        self.target_nets = [Q_Network(state_size, hidden_size, action_size) if _ != 5 else Q_Network(18, hidden_size, action_size) for _ in range(num_agents)]  # 需要将agnet 3的input维度设置为14
        self.optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.nets]
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = OrnsteinUhlenbeckNoise(action_size, seed)

    def get_actions(self, states):
        actions = []
        for i, state in enumerate(states):
            q_values = self.nets[i](torch.tensor(state))
            noise = self.noise.sample()
            q_values_with_noise = q_values + torch.tensor(noise)
            #action = torch.argmax(q_values_with_noise).item()
            action = q_values_with_noise
            actions.append(action)
        return actions

    def update(self, sample,  batch_size=64, gamma=0.99):
        experiences = sample
        states, actions, rewards, next_states, dones = experiences

        total_loss = 0
        for i, (net, target_net, optimizer) in enumerate(zip(self.nets, self.target_nets, self.optimizers)):
            q_values = net(torch.tensor(states[i]))
            next_q_values = target_net(torch.tensor(next_states[i]))
            target_q_values = torch.tensor(rewards[i]) + self.gamma * next_q_values.max() * (1 - torch.tensor(dones[i]))
            loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / self.num_agents

    def update_target(self):
        for net, target_net in zip(self.nets, self.target_nets):
            target_net.load_state_dict(net.state_dict())