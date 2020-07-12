import tensorflow as tf
import numpy as np


class ActorCritic:
    def __init__(
        self,
        n_actions,
        n_features,
        critic_learning_rate=0.01,
        actor_learning_rate = 0.001,
        reward_decay=0.9,
        action_list = []
    ):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr_C=critic_learning_rate
        self.lr_A = actor_learning_rate
        self.gamma=reward_decay
        self.action_list = action_list

        self.build_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):

        #Critic model
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')

        with tf.variable_scope('critic_model'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['critic_model_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q = tf.matmul(l1, w2) + b2


        with tf.variable_scope('actor_net'):
            c_names = ['actor_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.a_prob = tf.nn.softmax(tf.matmul(l1, w2) + b2)

        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions],name='Q_target')

        with tf.variable_scope('Critic_loss'):
            self.c_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q))
        with tf.variable_scope('Critic_train'):
            self.c_train_op = tf.train.AdamOptimizer(self.lr_C).minimize(self.c_loss)


        #Actor model
        with tf.variable_scope('Actor_loss'):
            log_prob = tf.log(self.a_prob)
            self.td_error = self.q_target - self.q
            self.a_loss = -tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
            self.entropy = -tf.reduce_sum(log_prob*self.a_prob)


        with tf.variable_scope('Actor_train'):
            self.a_train_op = tf.train.AdamOptimizer(self.lr_A).minimize(self.a_loss)



    def train(self,state,action,reward,state_,action_):

        q = self.sess.run(self.q,feed_dict={self.s: state})
        q_next = self.sess.run(self.q,feed_dict={self.s: state_})

        q_target = q.copy()

        batch_index = np.arange(len(q_target),dtype = np.int32)

        q_target[batch_index,action] = reward + self.gamma * q_next[batch_index,action_]

        _,c_loss = self.sess.run([self.c_train_op, self.c_loss],
                                     feed_dict={self.s: state,
                                                self.q_target: q_target})

        _,a_loss = self.sess.run([self.a_train_op, self.a_loss],
                                     feed_dict={self.s:state,self.q_target: q_target})
        return c_loss,a_loss

    def compute_entropy(self,state):
        state = state[np.newaxis, :]
        shanon_e= self.sess.run(self.entropy,{self.s:state})
        return shanon_e

    def choose_action(self,state):
        state = state[np.newaxis, :]
        probs = self.sess.run(self.a_prob, {self.s: state})
        if len(self.action_list)>1:
            sub_p = probs[0,self.action_list]
            sub_p = sub_p/sum(sub_p)
            action = np.random.choice(self.action_list, p=sub_p.ravel())
        else:
            action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action
