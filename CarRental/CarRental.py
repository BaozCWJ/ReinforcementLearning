import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CarRental(object):
    def __init__(self,max_car):
        self.max_car = max_car
        self.max_move= 5
        self.price = 10
        self.cost = 2
        self.reward1,self.T1 = self.probability(3,3)
        self.reward2,self.T2 = self.probability(4,2)
        self.value = np.zeros((self.max_car+1,self.max_car+1))
        self.policy = np.zeros((self.max_car+1,self.max_car+1))

    def probability(self,e_rental,e_return):
        rental_p = np.zeros(self.max_car+1)
        return_p = np.zeros(self.max_car+1)
        for kk in range(self.max_car):
                return_p[kk] = np.power(e_return,kk)/math.factorial(kk)*np.exp(-e_return)
                rental_p[kk] = np.power(e_rental,kk)/math.factorial(kk)*np.exp(-e_rental)
        return_p[self.max_car]=1-sum(return_p)
        rental_p[self.max_car]=1-sum(rental_p)

        T = np.zeros((self.max_car+1,self.max_car+1))
        reward = np.zeros(self.max_car+1)

        for morning_n in range(self.max_car+1):
            for rental_car in range(self.max_car+1):
                real_rental = min(morning_n,rental_car)
                reward[morning_n] += real_rental*self.price*rental_p[rental_car]
                for return_car in range(self.max_car+1):
                    n = max(morning_n-rental_car,0)
                    n = min(n+return_car,self.max_car)
                    T[morning_n,n] += return_p[return_car]*rental_p[rental_car]
        return reward,T

    def step(self,n1,n2,a):
        decay = 0.9
        a_min = max(-self.max_move,-n2,n1-self.max_car)
        a_max = min(+self.max_move,+n1,self.max_car-n2)
        a = np.clip(a,a_min,a_max)
        morning_n1 = int(n1 - a)
        morning_n2 = int(n2 + a)
        val = -self.cost*np.abs(a)+self.reward1[morning_n1]+self.reward2[morning_n2]
        for new_n1 in range(self.max_car+1):
            for new_n2 in range(self.max_car+1):
                val += self.T1[morning_n1,new_n1]*self.T2[morning_n2,new_n2]*(decay*self.value[new_n1,new_n2])
        return val

    def policyEval(self):
        diff = 0.1
        eps = 1e-6
        while (diff > eps):
            diff =0
            for n1 in range(self.max_car+1):
                for n2 in range(self.max_car+1):
                    temp_value = self.value[n1,n2]
                    action = self.policy[n1,n2]
                    self.value[n1,n2] = self.step(n1,n2,action)
                    diff = max(diff,np.abs(self.value[n1,n2]-temp_value))

    def policyImprove(self):
        flag = True
        for n1 in range(self.max_car+1):
            for n2 in range(self.max_car+1):
                b = self.policy[n1,n2]
                self.policy[n1,n2] = self.greedy(n1,n2)
                if b != self.policy[n1,n2]:
                    flag = False
        return flag

    def greedy(self,n1,n2):
        a_min = max(-self.max_move,-n2,n1-self.max_car)
        a_max = min(+self.max_move,+n1,self.max_car-n2)
        value = [self.step(n1,n2,ii) for ii in range(a_min,a_max+1)]
        return np.argmax(value)+a_min


    def run(self):
        flag = False
        count = 0
        while not flag:
            self.policyEval()
            flag=self.policyImprove()
            count+=1
        self.printValue(count)
        self.printPolicy(count)

    def printValue(self,index):
        x = []
        y = []
        z = []
        for i in range(self.max_car+1):
            for j in range(self.max_car+1):
                x.append(i)
                y.append(j)
                z.append(self.value[i,j])
        fig = plt.figure()
        title = 'Episode:'+str(index)
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel('A numb of car')
        ax.set_ylabel('B numb of car')
        ax.set_zlabel('value')
        plt.show()

    def printPolicy(self,index):
        fig = plt.figure(figsize=(15, 6), dpi=200)
        grid = plt.GridSpec(nrows=1, ncols=16, figure=fig)
        ACTIONS = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
        ax = fig.add_subplot(grid[0, 9:-1])
        cax = fig.add_subplot(grid[0, -1])
        im = ax.matshow(self.policy.T, origin='lower', cmap=plt.cm.get_cmap(name='RdBu', lut=len(ACTIONS)))
        ax.title.set_y(y=1.0)
        ax.xaxis.tick_bottom()
        cb = fig.colorbar(mappable=im, cax=cax, label='Policy Function')
        im.set_clim(vmin=-5.5, vmax=5.5)
        cb.set_ticks(ticks=ACTIONS)
        ax.set_xticks(np.arange(self.policy.shape[0] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.policy.shape[1] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel(xlabel='Number of Cars at Location 1')
        ax.set_ylabel(ylabel='Number of Cars at Location 2')
        ax.set_title(label=f'Policy Function at {index} Iterations')
        plt.show()
