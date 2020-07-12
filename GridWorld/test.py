import numpy as np
from GridWorld import *



gamma=0.9
eps=1e-4
maxSize=5
W=GridWorld(maxSize)
state=W.reset()
action=random.choice(action_space)
valuefunction=value(gamma,maxSize)
flag=valuefunction.check_convergence(eps)
while not flag:
    new_state,reward=W.step(state,action)
    valuefunction.update_Qvalue(state,action,reward,new_state)
    flag=valuefunction.check_convergence(eps)
    state = new_state
    action = random.choice(action_space)
print('v=')
print(valuefunction.estimation())
print(valuefunction.inference())
