import os
import numpy as np
import tensorflow as tf
import gfootball.env as football_env

env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper",
        representation = 'simple115',
        number_of_left_players_agent_controls = 3,
        stacked = False, logdir = './log',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False)


class FLAGS(object):
    def __init__(self):
        self.n_actions=19
        self.n_features=115
        self.n_agents=3
        self.lr_C=0.001
        self.lr_A=0.001
        self.gamma=0.99
        self.decay_freq = 100
        self.memory_size = 200
        self.batch_size = 64
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.max_epoch = 5000
