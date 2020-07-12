import numpy as np
import random
from IHT import *


action_space = [-1, 0, 1]
max_velocity = 0.07
min_velocity = -0.07
max_position = 0.6
min_position = -1.2


class MountainCar(object):
    def __init__(self):
        self.max_position=max_position
        self.min_position=min_position
        self.max_velocity=max_velocity
        self.min_velocity=min_velocity
        self.goal_position=max_position

    def step(self,position,velocity,action):
        new_velocity = velocity+0.001*action-0.0025*np.cos(3*position)
        new_velocity = np.clip(new_velocity,self.min_velocity,self.max_velocity)
        new_position = position + new_velocity
        new_position = np.clip(new_position,self.min_position,self.max_position)

        if (new_position==self.min_position and new_velocity<0): new_velocity=0

        flag = bool(new_position>=self.goal_position)
        reward = -1
        return new_position,new_velocity,reward,flag

    def reset(self):
        start_position = -0.5
        start_velocity = 0
        flag = False
        return start_position,start_velocity,flag


def eps_greedy(position,velocity,value,epsilon):
    if np.random.binomial(1,epsilon) == 1:
        return np.random.choice(action_space)
    values = [value.get_value(position,velocity,ii) for ii in action_space]
    return action_space[np.argmax(values)]


class SARSA_value:
    def __init__(self, step, tilings = 8, maxSize = 2048):
        self.hashtable = IHT(maxSize)
        self.numtilings = tilings
        self.weights = np.zeros(maxSize)
        self.positionscale = self.numtilings / (max_position - min_position)
        self.velocityscale = self.numtilings / (max_velocity - min_velocity)
        self.stepsize = step/tilings

    def getTiles(self, position, velocity, action):
        return tiles(self.hashtable, self.numtilings,
                     [self.positionscale*position, self.velocityscale*velocity],[action])

    def get_value(self, position, velocity, action):
        if position == max_position:
            return 0
        tiles = self.getTiles( position, velocity, action)
        return np.sum(self.weights[tiles])

    def update_value(self, position, velocity, action, returns):
        tiles = self.getTiles( position, velocity, action)
        estimation = np.sum(self.weights[tiles])
        delta = self.stepsize * (returns - estimation)
        for i in tiles:
            self.weights[i] += delta

    def get_cost(self, position, velocity):
        costs = [self.get_value(position, velocity, ii) for ii in action_space]
        return -np.max(costs)
