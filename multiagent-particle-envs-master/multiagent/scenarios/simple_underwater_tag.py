# encoding: utf-8
"""
@Version: Python3.x
@Author: FengYuKai
@Contact: fengyukai2021@ia.ac.cn
@Software: PyCharm
@Filename: 111
@CreatTime: 2023/6/820:23
@Description：简单描述
"""

import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        num_walls = 4
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.action_callback = None if i < num_adversaries else 1
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.2 if agent.adversary else 0.05
            agent.frequency = 2.5 if agent.adversary else 1.3
            agent.bias = 1.047 if agent.adversary else 1.06
            agent.max_speed = 0.4 if agent.adversary else 0.6
            agent.max_omega = 160/180 * math.pi
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        world.walls = [Wall() for i in range(num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = i
            wall.endpoints = (-2, 2)
            if i == 0:
                wall.orient = 'H'
                wall.axis_pos = -2.0
            elif i == 1:
                wall.orient = 'V'
                wall.axis_pos = -2.0
            elif i == 2:
                wall.orient = 'H'
                wall.axis_pos = 2.0
            elif i == 3:
                wall.orient = 'V'
                wall.axis_pos = 2.0
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.p_omega = np.zeros(world.dim_y)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.array([2, 2])
                landmark.state.p_vel = np.zeros(world.dim_p)
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        main_reward = self.adversary_reward(agent, world)
        return main_reward

    def adversary_reward(self, agent, world):
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew -= 0.05 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        def bound(x):
            if x < 1.5:
                return 0
            if x < 2.0:
                return (x - 1.5) * 2
            return min(np.exp(2 * x - 2), 10)

        for adv in adversaries:
            for p in range(world.dim_p):
                x = abs(adv.state.p_pos[p])
                rew -= bound(x)
        return rew

    def observation(self, agent, world):
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

