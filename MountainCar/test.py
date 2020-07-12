import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MountainCar import *


figureIndex = 0
def print3d(valuefunction, index):
    global figureIndex
    gridsize = 40
    positionstep = (max_position - min_position)/gridsize
    positions = np.arange(min_position, max_position + positionstep, positionstep)
    velocitystep = (max_velocity - min_velocity)/gridsize
    velocities = np.arange(min_velocity, max_velocity + velocitystep, velocitystep)
    x = []
    y = []
    z = []
    for i in positions:
        for j in velocities:
            x.append(i)
            y.append(j)
            z.append(valuefunction.get_cost(i, j))
    fig = plt.figure(figureIndex)
    figureIndex += 1
    title = 'Episode:'+str(index)
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('cost to go')
    plt.show()


'''
fig = plt.figure('base')
title = 'gravity potential'
fig.suptitle(title)
plt.xlim((min_position,max_position))
ax=fig.add_subplot(111)
xx=np.linspace(min_position,max_position,10000)
yy=0.0025*np.cos(3*xx)
ax.plot(xx,yy)
ax.set_xlabel('position')
ax.set_ylabel('potential')
plt.show()


'''
alpha = 0.3
gamma = 1
eps = 0
episodes = 9000
targetEpisodes = [1, 10, 100, 1000, 9000]
valuefunction = SARSA_value(alpha)
for i in range(episodes):
    Car=MountainCar()
    position,velocity,flag=Car.reset()
    action = eps_greedy(position,velocity,valuefunction, eps)
    while not flag:
        new_position, new_velocity, reward , flag = Car.step(position,velocity,action)
        new_action = eps_greedy(new_position, new_velocity, valuefunction, eps)
        returns = reward + gamma * valuefunction.get_value( new_position, new_velocity, new_action )
        if position != max_position:
            valuefunction.update_value( position, velocity, action, returns )
        position = new_position
        velocity = new_velocity
        action = new_action
    if i+1 in targetEpisodes:
        print3d(valuefunction, i+1)
