import numpy as np
import random


action_map={0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)}
action_space=[0,1,2,3]
x_size = 10
y_size = 7

class WindyGrid(object):
    def __init__(self,stochastic):
        self.stoch_wind = stochastic
        self.max_x = x_size-1
        self.max_y = y_size-1
        self.goal = np.array([7,3])
    def reset(self):
        state = np.array([0,3])
        self.time = 0
        return state
    def step(self,state,action_id):
        x,y = state
        action = action_map[action_id]
        wind = 0
        if self.stoch_wind and x in [3,4,5,6,7,8]:
            wind = random.choice([-1,0,1])
        if x in [3,4,5,8]:
            wind = 1 + wind
        elif x in [6,7]:
            wind = 2 + wind
        new_state = state + action + (0,wind)
        new_state = np.array(new_state)
        new_state[0] = np.clip(new_state[0],0,self.max_x)
        new_state[1] = np.clip(new_state[1],0,self.max_y)
        flag = int((new_state == self.goal).all())
        reward = flag -1
        self.time += 1
        return new_state,reward,flag





class value:
    def __init__(self,gamma,epsilon,alpha):
        self.discount = gamma
        self.lr = alpha
        self.eps = epsilon
        self.QvalueTable=np.zeros((x_size,y_size,len(action_space)))

    def get_Qvalue(self,state,action):
        return self.QvalueTable[state[0],state[1],action]

    def epsilon_greedy(self,state):
        if random.random() > self.eps:
            Qvalue=[self.get_Qvalue(state,ii) for ii in action_space]
            return np.argmax(Qvalue)
        else:
            return random.choice(action_space)

    def Q_learning(self,state,action,reward,new_state):
        Q=self.get_Qvalue(state,action)
        Qvalue=[self.get_Qvalue(new_state,ii) for ii in action_space]
        delta = reward + self.discount*np.max(Qvalue) - Q
        self.QvalueTable[state[0],state[1],action]= Q + self.lr*delta

    def SARSA(self,state,action,reward,new_state,new_action):
        Q=self.get_Qvalue(state,action)
        Qvalue=self.get_Qvalue(new_state,new_action)
        delta = reward + self.discount*Qvalue - Q
        self.QvalueTable[state[0],state[1],action]= Q + self.lr*delta

    def inference(self):
        return np.argmax(self.QvalueTable,axis=2).T[::-1,:]


def Q_learning(GridWorld,Qvalue,max_episode):
    TimeStep = []
    for ii in range(max_episode):
        flag = False
        state = GridWorld.reset()
        action = Qvalue.epsilon_greedy(state)
        while not flag:
            new_state,reward,flag = GridWorld.step(state,action)
            Qvalue.Q_learning(state,action,reward,new_state)
            state = new_state
            action = Qvalue.epsilon_greedy(state)
        TimeStep.append(GridWorld.time)
    return TimeStep


def SARSA(GridWorld,Qvalue,max_episode):
    TimeStep = []
    for ii in range(max_episode):
        flag = False
        state = GridWorld.reset()
        action = Qvalue.epsilon_greedy(state)
        while not flag:
            new_state,reward,flag = GridWorld.step(state,action)
            new_action = Qvalue.epsilon_greedy(new_state)
            Qvalue.SARSA(state,action,reward,new_state,new_action)
            state = new_state
            action = new_action
        TimeStep.append(GridWorld.time)
    return TimeStep
