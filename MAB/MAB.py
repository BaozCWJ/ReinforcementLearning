import numpy as np
import random
import math


class MultiArmedBandit(object):
    def __init__(self,n,reward):
        self.number=n
        self.action=[]
        self.value=[]
        self.reward=reward
    def epsilon_greedy(self,epsilon,times):
        random.seed(1.5) #reproducing result
        assert (epsilon <=1) and (epsilon>=0)
        count=np.zeros(self.number)
        Q=np.zeros(self.number)
        value = 0
        for ii in range(times):
            if random.random()>1-epsilon:
                action = random.choice(range(self.number))
            else:
                action = np.argmax(Q)
            count[action] += 1
            reward = self.reward(action)
            Q[action] +=(reward-Q[action])/count[action]
            value += reward
            self.action.append(action)
            self.value.append(value)
        return self.action,self.value
    def UpperBound(self,c,times):
        epsilon=1e-8
        count = np.zeros(self.number)
        Q = np.zeros(self.number)
        value = 0
        for ii in range(times):
            J = [c * np.sqrt(np.log(ii + 1)) / np.sqrt(count[x]+epsilon) for x in range(self.number)]
            action = np.argmax(Q+J)
            count[action] += 1
            reward = self.reward(action)
            Q[action] += (reward - Q[action]) / count[action]
            value += reward
            self.action.append(action)
            self.value.append(value)
        return self.action, self.value
    def R_UpperBound(self,c,times,regulizer):
        count = np.zeros(self.number)
        Q = np.zeros(self.number)
        value = 0
        for ii in range(times):
            J = [regulizer(ii,x) for x in range(self.number)]
            action = np.argmax(Q + c*J)
            count[action] += 1
            reward = self.reward(action)
            Q[action] += (reward - Q[action]) / count[action]
            value += reward
            self.action.append(action)
            self.value.append(value)
        return self.action, self.value
    def Gradient(self,c,times):
        H=np.ones(self.number)
        value = 0
        random.seed(1.5)
        for ii in range(times):
            pi=[math.exp(h) for h in H]
            Z=sum(pi)
            index = random.random()*Z
            action =0
            index -= pi[action]
            while index>=0 and action <(self.number-1):
                action += 1
                index -=  pi[action]
            reward = self.reward(action)
            step = c * (reward - value / max(ii,1))
            H = [H[x] + step * ((action == x) - pi[x] / Z) for x in range(self.number)]
            value += reward
            self.action.append(action)
            self.value.append(value)
        return self.action, self.value





def action_analysis(optimal,action_traj):
    optimal_count=0
    optimal_rate=[]
    for ii in range(len(action_traj)):
        if action_traj[ii]==optimal:
            optimal_count +=1
        optimal_rate.append(optimal_count/(ii+1))
    return optimal_rate

def value_analysis(value_traj):
    value_average = []
    for ii in range(len(value_traj)):
        value_average.append(value_traj[ii] / (ii + 1))
    return value_average






