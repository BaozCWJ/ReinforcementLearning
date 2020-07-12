import numpy as np
import random

action_map={0:(1,0),1:(-1,0),2:(0,1),3:(0,-1)}
action_space=[0,1,2,3]


class GridWorld(object):
    def __init__(self,maxSize):
        self.max_x=maxSize-1
        self.max_y=maxSize-1
    def step(self,state,action):
        new_state=state+np.array(action_map[action])
        reward=0
        if (state==(0,1)).all():
            new_state=np.array([4,1])
            reward=10
        elif (state==(0,3)).all():
            new_state=np.array([2,3])
            reward=5
        if (new_state!=np.clip(new_state,0,self.max_x)).any():
            new_state=np.clip(new_state,0,self.max_x)
            reward=-1
        return new_state,reward
    def reset(self):
        start_x,start_y=np.random.randint(low=0,high=self.max_x,size=2)
        state =np.array([start_x,start_y])
        return state

class value:
    def __init__(self,gamma,maxSize):
        self.discount=gamma
        self.QvalueTable=np.zeros((maxSize,maxSize,len(action_space)))
        self.deviation=np.ones((maxSize,maxSize,len(action_space)))
    def get_Qvalue(self,state,action):
        return self.QvalueTable[state[0],state[1],action]
    def update_Qvalue(self,state,action,reward,new_state,):
        Q=self.get_Qvalue(state,action)
        Qvalue=[self.get_Qvalue(new_state,ii) for ii in action_space]
        new_Q=reward+self.discount*np.sum(Qvalue)*0.25
        #new_Q=reward+self.discount*np.max(Qvalue)
        self.QvalueTable[state[0],state[1],action]=new_Q
        self.deviation[state[0],state[1],action]=abs(Q-new_Q)
    def check_convergence(self,eps):
        flag = bool(np.max(self.deviation)<=eps)
        return flag
    def get_value(self,state):
        Qvalue = [self.get_Qvalue(state,ii) for ii in action_space]
        return np.sum(Qvalue)*0.25
        #return np.max(Qvalue)
    def estimation(self):
        return np.sum(self.QvalueTable,axis=2)*0.25
        #return np.max(self.QvalueTable,axis=2)
    def inference(self):
        return np.argmax(self.QvalueTable,axis=2)
