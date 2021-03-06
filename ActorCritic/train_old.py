'''
from data import *
from model import DeepQNetwork
'''
import numpy as np
import collections
import random
import matplotlib.pyplot as plt

def train():

    memory = []
    Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state"])

    model=DeepQNetwork(
        flags.n_actions,
        flags.n_features,
        flags.lr,
        flags.gamma,
        flags.epsilon_max,
        []
    )

    loss_his=[]
    reward_his=[]
    step_his=[]

    for ii in range(flags.max_epoch):
        state = env.reset()

        reward_all = 0
        done = False
        steps = 0
        loss = 0

        while not done:
            action = model.choose_action(state)
            next_state , reward , done , _ = env.step(action)
            reward_all += reward
            steps += 1

            if len(memory) > flags.memory_size:
                memory.pop(0)
            memory.append(Transition(state, action , reward , next_state))

            if len(memory) > flags.batch_size * 2:
                batch_transition = random.sample(memory , flags.batch_size)
                batch_state, batch_action, batch_reward, batch_next_state = map(np.array , zip(*batch_transition))
                loss = model.train(state = batch_state ,
                            action = batch_action ,
                            reward = batch_reward ,
                            state_ = batch_next_state
                             )

            if (ii+1) % flags.replace_target_freq == 0:
                model.replace_target()
                model.decay_epsilon()

            state = next_state

        if loss>0:
            loss_his.append(loss)
            reward_his.append(reward_all)
            step_his.append(steps)
            print("epoch=",ii,"/loss=",loss,"/reward_all=",reward_all,"/steps=",steps)

    return loss_his,reward_his,step_his


def plot_his(his,his_name,smooth):
    x=np.arange(len(his))
    if smooth:
        k=50
        y=np.zeros(len(his)-k)
        for ii in range(len(his)-k):
            for jj in range(k):
                y[ii] += his[ii+jj]
            y[ii]=y[ii]/k
        plt.plot(x[:len(his)-k],y)
    else:
        plt.plot(x,his)
    plt.ylabel(his_name)
    plt.xlabel("episode")
    plt.title(his_name+"-episode")
    plt.savefig(his_name+".png")
    plt.show()

#flags = FLAGS()
#loss_his,reward_his,step_his=train()
#np.savetxt('data',(loss_his,reward_his,step_his))

his = np.loadtxt("data")
plot_his(his[0],"loss",True)
plot_his(his[1],"entropy",False)
plot_his(his[2],"reward",True)
