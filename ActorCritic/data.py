import os
import numpy as np
import tensorflow as tf
import gfootball.env as football_env

env = football_env.create_environment(env_name="academy_empty_goal",
        representation = 'simple115',
        number_of_left_players_agent_controls = 1,
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
        self.lr_C=0.01
        self.lr_A=0.001
        self.gamma=1.0
        self.replace_target_freq = 50
        self.memory_size = 500
        self.batch_size = 32
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.max_epoch = 500

empty_goal_action=[3,4,5,6,7,12,13,14,15,17,18]
