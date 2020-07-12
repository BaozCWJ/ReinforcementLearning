import numpy as np
import matplotlib.pyplot as plt
from WindyGrid import *


max_episode = 500000

GridA = WindyGrid(stochastic=True)
Qvalue = value(gamma=1,epsilon=0.1,alpha=0.2)

#timestep = Q_learning(GridA,Qvalue,max_episode)
timestep = SARSA(GridA,Qvalue,max_episode)
'''
traj = []
while len(traj)!=9:
    traj = []
    flag = False
    state = GridA.reset()
    action = Qvalue.epsilon_greedy(state)
    traj.append(state)
    while not flag:
        new_state,reward,flag = GridA.step(state,action)
        state = new_state
        action = Qvalue.epsilon_greedy(state)
        traj.append(state)
print(traj)
'''

print(Qvalue.inference())

'''
plt.xlim(-50,max_episode)
#plt.ylim(0,80)
plt.plot(timestep)
plt.hlines(8, -50, max_episode+50,color='black',linestyle="--")
plt.text(max_episode, 8, '8', ha='left', va='center')
plt.xlabel('# Episode')
plt.ylabel('Time steps')
plt.title('Q_learning - Stochastic windy gridworld')
#plt.title('SARSA - Stochastic windy gridworld')
plt.show()
'''
