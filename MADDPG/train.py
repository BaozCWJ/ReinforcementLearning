from data import *
from model import *
import collections
import random
import time

def train():

    positive_memory = []
    negative_memory = []
    zero_memory = []
    Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state"])

    agent1=MADDPG(
        'agent1',
        flags.n_actions,
        flags.n_features,
        flags.n_agents,
        flags.lr_C,
        flags.lr_A,
        flags.gamma,
        []
    )

    agent2=MADDPG(
        'agent2',
        flags.n_actions,
        flags.n_features,
        flags.n_agents,
        flags.lr_C,
        flags.lr_A,
        flags.gamma,
        []
    )

    agent3=MADDPG(
        'agent3',
        flags.n_actions,
        flags.n_features,
        flags.n_agents,
        flags.lr_C,
        flags.lr_A,
        flags.gamma,
        []
    )

    entropy_his=[]
    reward_his = []
    epsilon = 0.1
    for ii in range(flags.max_epoch):
        trajectory = []

        state = env.reset()
        reward_all = 0
        done = False
        steps = 0
        loss1 = 0
        t_start=time.time()

        if (ii+1)%flags.decay_freq==0:
            epsilon = epsilon*0.99
        action1 = agent1.choose_action(state[0,:],epsilon)
        action2 = agent2.choose_action(state[1,:],epsilon)
        action3 = agent3.choose_action(state[2,:],epsilon)
        action = [action1,action2,action3]

        while not done:
            next_state , reward , done , _ = env.step(action)
            next_action1 = agent1.choose_action(next_state[0,:],epsilon)
            next_action2 = agent2.choose_action(next_state[1,:],epsilon)
            next_action3 = agent3.choose_action(next_state[2,:],epsilon)
            next_action = [next_action1,next_action2,next_action3]
            reward_all = sum(reward)/3
            steps += 1

            trajectory.append(Transition(state, action , reward_all , next_state))

            state = next_state
            action = next_action

        reward_his.append(reward_all)
        if reward_all!=0:
            trajectory = [jj._replace(reward=reward_all) for jj in trajectory]
            if reward_all>0:
                positive_memory += trajectory
                if len(positive_memory)>10*flags.memory_size:
                    positive_memory=positive_memory[50:]
            if reward_all<0:
                negative_memory +=trajectory
                if len(negative_memory)>flags.memory_size:
                    negative_memory=negative_memory[50:]
        else:
            zero_memory += trajectory
            if len(zero_memory)>flags.memory_size:
                zero_memory=zero_memory[50:]

        memory = positive_memory+negative_memory+zero_memory

        if len(memory) > flags.batch_size:
            batch_transition = random.sample(memory , flags.batch_size)
            batch_state, batch_action, batch_reward, batch_next_state= map(np.array , zip(*batch_transition))
            batch_next_action = np.array([[agent1.choose_action(s[0,:],0),agent2.choose_action(s[1,:],0),agent3.choose_action(s[2,:],0)] for s in batch_next_state])

            loss1,_ = agent1.train(state = batch_state[:,0,:] ,
                            action = batch_action[:,0] ,
                            other_actions = np.delete(batch_action,0,axis=1),
                            reward = batch_reward ,
                            state_ = batch_next_state[:,0,:],
                            action_ = batch_next_action[:,0],
                            other_actions_ = np.delete(batch_next_action,0,axis=1)
                             )
            loss2,_ = agent2.train(state = batch_state[:,1,:] ,
                            action = batch_action[:,1] ,
                            other_actions = np.delete(batch_action,1,axis=1),
                            reward = batch_reward ,
                            state_ = batch_next_state[:,1,:],
                            action_ = batch_next_action[:,1],
                            other_actions_ = np.delete(batch_next_action,1,axis=1)
                             )
            loss3,_ = agent3.train(state = batch_state[:,2,:] ,
                            action = batch_action[:,2] ,
                            other_actions = np.delete(batch_action,2,axis=1),
                            reward = batch_reward ,
                            state_ = batch_next_state[:,2,:],
                            action_ = batch_next_action[:,2],
                            other_actions_ = np.delete(batch_next_action,2,axis=1)
                             )

            agent1.soft_update()
            agent2.soft_update()
            agent3.soft_update()

        if reward_all!=0:
            print("epoch=",ii,"/time=",time.time()-t_start,"/loss1=",loss1,"/loss2=",loss2,"/loss3=",loss3,"/reward=",reward_all)

    return reward_his


flags = FLAGS()
reward_his=train()
np.savetxt('data',(reward_his))
