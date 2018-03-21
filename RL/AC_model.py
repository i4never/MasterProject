import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, sess, action_dim, feature_dim, learning_rate):
        self.sess = sess

        self.state_dim = feature_dim
        self.action_dim = action_dim
        self.lr = learning_rate

        self.obs = tf.placeholder(tf.float32, [1, feature_dim], "state")
        self.action = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope("Actor"):
            layer1 = tf.layers.dense(
                inputs=self.obs,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer1"
            )

            layer2 = tf.layers.dense(
                inputs=layer1,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer2"
            )

            self.acts_prob = tf.layers.dense(
                inputs=layer2,
                units=self.action_dim,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="acts_prob"
            )

        with tf.variable_scope("exp_value"):
            log_prob = tf.log(self.acts_prob[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)

    def learn(self, state, action, td):
        state = state[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 {self.obs: state, self.action: action, self.td_error: td})
        return exp_v

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.obs: state})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic:
    def __init__(self, sess, feature_dim, learning_rate, reward_decay):
        self.sess = sess

        self.state_dim = feature_dim

        self.lr = learning_rate
        self.gamma = reward_decay

        self.obs = tf.placeholder(tf.float32, [None, feature_dim], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.reward = tf.placeholder(tf.float32, None, "reward")

        with tf.variable_scope("Critic"):
            layer1 = tf.layers.dense(
                inputs=self.obs,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.1, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer1"
            )

            self.value = tf.layers.dense(
                inputs=layer1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name="Value"
            )

            with tf.variable_scope("TD_error"):
                self.td_error = self.reward + self.gamma * self.v_ - self.value
                self.loss = tf.square(self.td_error)

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, state, reward, state_):
        state, state_ = state[np.newaxis, :], state_[np.newaxis, :]

        v_ = self.sess.run(self.value, {self.obs: state_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.obs: state, self.v_: v_, self.reward: reward})
        return td_error


#######################
# gym for test
#######################
import gym

env = gym.make("CartPole-v0")
env.seed(1)
env = env.unwrapped

sess = tf.Session()
actor = Actor(sess, env.action_space.n, env.observation_space.shape[0], 0.001)
critic = Critic(sess, env.observation_space.shape[0], 0.001, 0.99)

sess.run(tf.global_variables_initializer())

for i_episode in range(1000):
    s = env.reset()
    t = 0
    track_r = []

    while True:
        env.render()
        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(20)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t>=10000:
            ep_rs_sum = sum(track_r)
            print("episode:", i_episode, " reward", int(ep_rs_sum))
            break