import tensorflow as tf
import numpy as np


class MADDPG:
    def __init__(
        self,
        name,
        n_actions,
        n_features,
        n_agents,
        critic_learning_rate=0.01,
        actor_learning_rate = 0.001,
        reward_decay=0.9,
        action_list = []
    ):
        self.n_actions=n_actions
        self.n_features=n_features
        self.n_agents = n_agents
        self.lr_C=critic_learning_rate
        self.lr_A = actor_learning_rate
        self.gamma=reward_decay
        self.action_list = action_list
        self.name = name



        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, 1],name='Q_target')
        self.action = tf.compat.v1.placeholder(tf.int32, [None, 1],name='actions')
        self.other_actions = tf.compat.v1.placeholder(tf.int32, [None, self.n_agents-1],name='actions')

        self.Actor_network()
        self.q_eval = self.Critic_network(self.action,self.other_actions,False)
        self.q = self.Critic_network(self.action,self.other_actions,True)


        #Critic loss
        with tf.variable_scope('Critic_loss'):
            self.c_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('Critic_train'):
            self.c_train_op = tf.train.AdamOptimizer(self.lr_C).minimize(self.c_loss)


        #Actor loss
        with tf.variable_scope('Actor_loss'):
            self.a_loss = -tf.reduce_mean(self.Critic_network(self.actor_eval,self.other_actions,False))
        with tf.variable_scope('Actor_train'):
            self.a_train_op = tf.train.AdamOptimizer(self.lr_A).minimize(self.a_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.target_init = self.collect_parameter(0)
        self.soft_update(True)
        self.target_update = self.collect_parameter(0.99)


    def Critic_network(self,action,other_actions,target):

        #Critic net
        if target:
            network_name = self.name+'_critic_target_net'
        else:
            network_name = self.name+'_critic_eval_net'

        with tf.variable_scope(network_name):

            n_l1,n_l2, w_initializer, b_initializer = 64,32,tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                l1_a = tf.concat([l1, tf.cast(action,dtype=tf.float32), tf.cast(other_actions,dtype=tf.float32)], axis=-1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1+self.n_agents, n_l2], initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer)
                l2 = tf.nn.relu(tf.matmul(l1_a, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, 1], initializer=w_initializer)
                b3 = tf.get_variable('b3', [1, 1], initializer=b_initializer)
                q = tf.matmul(l2, w3) + b3
        return q



    def Actor_network(self):

        #Actor eval net
        with tf.variable_scope(self.name+'_actor_eval_net'):
            n_l1,n_l2, w_initializer, b_initializer = 64,32,tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1],initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l1],initializer=b_initializer)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2],initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, n_l2],initializer=b_initializer)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions],initializer=w_initializer)
                b3 = tf.get_variable('b3', [1, self.n_actions],initializer=b_initializer)
                self.actor_eval = tf.expand_dims(tf.argmax(tf.matmul(l2, w3) + b3, axis=1),1)

        #Actor target net
        with tf.variable_scope(self.name+'_actor_target_net'):
            n_l1,n_l2, w_initializer, b_initializer = 64,32,tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1],initializer=w_initializer)
                b1 = tf.get_variable('b1', [1, n_l1],initializer=b_initializer)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2],initializer=w_initializer)
                b2 = tf.get_variable('b2', [1, n_l2],initializer=b_initializer)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions],initializer=w_initializer)
                b3 = tf.get_variable('b3', [1, self.n_actions],initializer=b_initializer)
                self.actor = tf.expand_dims(tf.argmax(tf.matmul(l2, w3) + b3, axis=1),1)


    def train(self,state,action,other_actions,reward,state_,action_,other_actions_):
        action = action[:,np.newaxis]
        action_ = action_[:,np.newaxis]

        _,a_loss = self.sess.run([self.a_train_op, self.a_loss],
                                     feed_dict={self.s:state,self.other_actions:other_actions})

        q_next = self.sess.run(self.q,feed_dict={self.s: state_,self.action:action_,self.other_actions:other_actions_})

        q_target = np.expand_dims(reward,1)+ self.gamma * q_next

        _,c_loss = self.sess.run([self.c_train_op, self.c_loss],
                                     feed_dict={self.s: state,self.action:action,
                                                self.other_actions:other_actions,
                                                self.q_target: q_target})

        return c_loss,a_loss

    def choose_action(self,state,epsilon):
        state = state[np.newaxis, :]
        if np.random.uniform() > epsilon:
            return self.sess.run(self.actor, feed_dict={self.s: state})[0][0]
        else:
            return np.random.randint(0,self.n_actions)



    def collect_parameter(self,tau):
        online_name = self.name+'_actor_eval_net'
        target_name = self.name+'_actor_target_net'
        online_var = [i for i in tf.trainable_variables() if online_name in i.name]
        target_var = [i for i in tf.trainable_variables() if target_name in i.name]
        actor_target_update=[tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

        online_name = self.name+'_critic_eval_net'
        target_name = self.name+'_critic_target_net'
        online_var = [i for i in tf.trainable_variables() if online_name in i.name]
        target_var = [i for i in tf.trainable_variables() if target_name in i.name]
        critic_target_update=[tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]
        return [actor_target_update,critic_target_update]

    def soft_update(self,init=False):
        if init:
            self.sess.run(self.target_init)
        else:
            self.sess.run(self.target_update)
