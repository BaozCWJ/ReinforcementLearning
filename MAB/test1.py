import numpy as np
import matplotlib.pyplot as plt
from MAB import *

n=15
times=10000
epsilon_set=[0,0.1,0.01]
r_min=1
r_max=15
np.random.seed(1)
mu=np.random.permutation(range(1,16))
optimal_option=np.argmax(mu)
print('optimal action=',optimal_option)
print('optimal average reward=',max(mu))



def GaussianReward(i):
    return np.random.normal(mu[i],1)



MAB1=MultiArmedBandit(n,GaussianReward)
action_traj1,value_traj1=MAB1.epsilon_greedy(epsilon_set[0],times)
rate1=action_analysis(optimal_option,action_traj1)
val1=value_analysis(value_traj1)
MAB2=MultiArmedBandit(n,GaussianReward)
action_traj2,value_traj2=MAB2.epsilon_greedy(epsilon_set[1],times)
rate2=action_analysis(optimal_option,action_traj2)
val2=value_analysis(value_traj2)
MAB3=MultiArmedBandit(n,GaussianReward)
action_traj3,value_traj3=MAB3.epsilon_greedy(epsilon_set[2],times)
rate3=action_analysis(optimal_option,action_traj3)
val3=value_analysis(value_traj3)


plt.figure()
r1,=plt.plot(range(times),rate1)
r2,=plt.plot(range(times),rate2)
r3,=plt.plot(range(times),rate3)

plt.legend(handles=[r1,r2,r3],labels=epsilon_set)
plt.ylabel('optimal action/%')
plt.xlabel('times')
plt.show()

plt.figure()
v1,=plt.plot(range(times),val1)
v2,=plt.plot(range(times),val2)
v3,=plt.plot(range(times),val3)

plt.legend(handles=[v1,v2,v3],labels=epsilon_set)
plt.xlabel('times')
plt.ylabel('average reward')
plt.show()
