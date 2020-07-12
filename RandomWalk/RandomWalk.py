import numpy as np
import random

class RandomWalk(object):
    def __init__(self,size):
        self.size=size
        self.state=0
    def reset(self):
        self.state =0
    def step(self):
        if random.random()>0.5:
            self.state+=1
        else:
            self.state-=1
        return int(self.state==self.size),(abs(self.state)==self.size)
    def run(self,gamma):
        flag = False
        diff = 0
        k = 0
        while not flag:
            state = self.state + self.size
            reward,flag=self.step()
            new_state = self.state + self.size
            diff += (gamma**k) * (reward + gamma*self.value[new_state]-self.value[state])
            k += 1
        return diff
    def TD(self,episode,alpha,gamma):
        self.value = np.ones((2*self.size+1))/2
        self.value[0]=0
        self.value[2*self.size]=0

        for ii in range(episode):
            self.reset()
            flag = False
            while not flag:
                state = self.state + self.size
                reward,flag = self.step()
                new_state = self.state + self.size
                diff = reward + gamma*self.value[new_state]-self.value[state]
                self.value[state] += alpha*diff
        return self.value

    def MC(self,episode,alpha,gamma):
        self.value = np.ones((2*self.size+1))/2
        self.value[0]=0
        self.value[2*self.size]=0

        for ii in range(episode):
            self.reset()
            flag = False
            traj = []
            while not flag:
                state = self.state + self.size
                traj.append(state)
                reward,flag = self.step()
            for ss in traj:
                self.value[ss] += alpha*(reward - self.value[ss])
        return self.value
