import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        epsilon_max = 0.9,
        action_list = []
    ):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=epsilon_max
        self.action_list = action_list

        self.build_model()
        t_params = tf.get_collection('target_model_params')
        e_params = tf.get_collection('eval_model_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # 用来接收 observation
        with tf.variable_scope('eval_model'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_model_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2


        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions],name='Q_target')
        with tf.variable_scope('loss'):  
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)



    def choose_action(self,state):
        state = state[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
            if len(self.action_list)>0:
                action = self.action_list[np.argmax(actions_value[:,self.action_list])]
            else:
                action = np.argmax(actions_value)
        else:
            if len(self.action_list)>0:
                action = np.random.choice(self.action_list)
            else:
                action = np.random.randint(0,self.n_actions)
        return action


    def train(self,state,action,reward,state_):

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],feed_dict={self.s_: state_,self.s: state})
        q_target = q_eval.copy()
        batch_index = np.arange(len(q_target),dtype = np.int32)
        q_target[batch_index,action] = reward + self.gamma * np.max(q_next, axis=1)

        _,loss = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: state,
                                                self.q_target: q_target})
        return loss

    def decay_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon * 0.5

    def replace_target(self):
        self.sess.run(self.replace_target_op)
