import numpy as np
import seaborn as sns
import copy
import math
import matlab.engine

class EntityState(object):
    def __init__(self):
        self.p_pos = None
        self.p_vel = None
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.c = None
        self.p_omega = None
class Action(object):
    def __init__(self):
        self.u = None
        self.c = None
        self.m = None
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.05,
                 hard=True):
        self.orient = orient
        self.axis_pos = axis_pos
        self.endpoints = np.array(endpoints)
        self.width = width
        self.hard = hard
        self.color = np.array([0.25, 0.25, 0.25])
class Entity(object):
    def __init__(self):
        self.i = 0
        self.name = ''
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.ghost = False
        self.density = 25.0
        self.color = None
        self.max_speed = None
        self.accel = None
        self.state = EntityState()
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.c_noise = None
        self.u_range = 1.0
        self.m_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None

class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.walls = []
        self.dim_c = 0
        self.dim_p = 2
        self.dim_y = 1
        self.dim_color = 3
        self.damping = 0.25
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        self.use_force = True
        self.use_cpg = False

    @property
    def entities(self):
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        for agent in self.agents:
            self.call_matlab(agent)

    def call_matlab(self, agent):
        eng = matlab.engine.start_matlab()
        matlab_function_path = r"Robofish_dynamic"
        full_path = eng.genpath(matlab_function_path, nargout=1)
        eng.addpath(full_path, nargout=0)
        def to_matlab_double(ndarray):
            return matlab.double(ndarray.tolist())
        p_pos_matlab = to_matlab_double(agent.state.p_pos)
        p_vel_matlab = to_matlab_double(agent.state.p_vel)
        p_omega_matlab = to_matlab_double(agent.state.p_omega)
        u_matlab = to_matlab_double(agent.action.u)
        position, velocity, angvel = eng.blogmodel_m(p_pos_matlab, p_vel_matlab, p_omega_matlab, u_matlab, nargout=3)
        agent.state.p_vel[0] = np.array(velocity)[0]
        agent.state.p_vel[1] = np.array(velocity)[1]
        agent.state.p_pos[0] = np.array(position)[0]
        agent.state.p_pos[1] = np.array(position)[1]
        agent.state.p_omega[0] = np.array(angvel)
        if agent.max_speed is not None:
            speed = np.sqrt(
                np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
            if speed > agent.max_speed:
                agent.state.p_vel[0] = agent.state.p_vel[0] / np.sqrt(np.square(agent.state.p_vel[0]) +
                                                                  np.square(agent.state.p_vel[1])) * agent.max_speed
                agent.state.p_vel[1] = agent.state.p_vel[1] / np.sqrt(np.square(agent.state.p_vel[0]) +
                                                                  np.square(agent.state.p_vel[1])) * agent.max_speed

        eng.quit()
        return