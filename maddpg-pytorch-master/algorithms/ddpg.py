# encoding: utf-8
"""
@Version: Python3.x
@Author: FengYuKai
@Contact: fengyukai2021@ia.ac.cn
@Software: PyCharm
@Filename: ddpg
@CreatTime: 7/11/2023上午 2:09
@Description：简单描述
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, seed):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

        self.noise = OrnsteinUhlenbeckNoise(action_dim, seed)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = self.noise.sample()
        return [np.clip(action + noise, -self.max_action, self.max_action)]

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
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

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), str(filename) + "_critic")
        torch.save(self.critic_optimizer.state_dict(), str(filename) + "_critic_optimizer")

        torch.save(self.actor.state_dict(), str(filename) + "_actor")
        torch.save(self.actor_optimizer.state_dict(), str(filename) + "_actor_optimizer")