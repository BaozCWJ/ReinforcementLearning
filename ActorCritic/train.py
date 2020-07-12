from data import *
from model import ActorCritic
import collections
import random
import time

def train():

    memory = []
    Transition = collections.namedtuple("Transition" , ["state", "action" , "reward" , "next_state","next_action"])

    model=ActorCritic(
        flags.n_actions,
        flags.n_features,
        flags.lr_C,
        flags.lr_A,
        flags.gamma,
        empty_goal_action
    )

    loss_his=[]
    entropy_his=[]
    reward_his = []

    for ii in range(flags.max_epoch):
        state = env.reset()
        init_state = state.copy()
        reward_all = 0
        done = False
        steps = 0
        loss = 0
        t_start=time.time()
        action = model.choose_action(state)

        while not done:
            next_state , reward , done , _ = env.step(action)
            next_action = model.choose_action(next_state)
            reward_all += reward
            steps += 1

            if len(memory) > flags.memory_size:
                memory.pop(0)
            memory.append(Transition(state, action , reward , next_state, next_action))

            state = next_state
            action = next_action


        if len(memory) > flags.batch_size:
            batch_transition = random.sample(memory , flags.batch_size)
            batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = map(np.array , zip(*batch_transition))
            loss,_ = model.train(state = batch_state ,
                            action = batch_action ,
                            reward = batch_reward ,
                            state_ = batch_next_state,
                            action_ = batch_next_action
                             )
            entropy = model.compute_entropy(init_state)

        if loss!=0:
            loss_his.append(loss)
            entropy_his.append(entropy)
            reward_his.append(reward_all)
            print("epoch=",ii,"/time=",time.time()-t_start,"/loss=",loss,"/entropy=",entropy,"/reward=",reward_all)

    return loss_his,entropy_his,reward_his


flags = FLAGS()
loss_his,entropy_his,reward_his=train()
np.savetxt('data',(loss_his,entropy_his,reward_his))
