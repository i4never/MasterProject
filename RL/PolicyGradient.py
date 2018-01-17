import tensorflow as tf
import numpy as np


class PolicyGradient:
    def __init__(self, action_dim, feature_dim, learning_rate, reward_decay, log_path=None):
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lr = learning_rate
        self.gamma = reward_decay

        # 每个eposide的信息
        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []

        self._build_net()

        self.sess = tf.Session()

        if log_path is not None:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter(log_path, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("Inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.feature_dim], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # layer1
        layer1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="layer1"
        )

        # layer2
        layer2 = tf.layers.dense(
            inputs=layer1,
            units=self.action_dim,  # 输出个数
            activation=None,  # 之后再加 Softmax
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer2'
        )

        self.all_act_prob = tf.nn.softmax(layer2, name="action_prob")

        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

    def _build_net_2(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.feature_dim], name="observations")  # 接收 observation
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")  # 接收我们在这个回合中选过的 actions
            self.tf_vt = tf.placeholder(tf.float32, [None, ],
                                        name="actions_value")  # 接收每个 state-action 所对应的 value (通过 reward 计算)

        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,  # 输出个数
            activation=tf.nn.tanh,  # 激励函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.action_dim,  # 输出个数
            activation=None,  # 之后再加 Softmax
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # 激励函数 softmax 出概率

        with tf.name_scope('loss'):
            # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
            #                                                               labels=self.tf_acts)  # 所选 action 的概率 -log 值
            # 下面的方式是一样的:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # no epsilon-greedy
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_observations.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        # discounted reward

        discounted_ep_rs = np.zeros_like(self.ep_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        discounted_ep_rs_norm = discounted_ep_rs

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: self.ep_observations,
            self.tf_acts: self.ep_actions,
            self.tf_vt: discounted_ep_rs_norm
        })

        self.ep_observations.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        return


##############################
# Test script, opengym, cartpole
import gym

RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
DISPLAY_REWARD_THRESHOLD = 400  # 当 回合总 reward 大于 400 时显示模拟窗口

env = gym.make('CartPole-v0')  # CartPole 这个模拟
env = env.unwrapped  # 取消限制
env.seed(1)  # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

print(env.action_space)  # 显示可用 action
print(env.observation_space)  # 显示可用 state 的 observation
print(env.observation_space.high)  # 显示 observation 最高值
print(env.observation_space.low)  # 显示 observation 最低值

# 定义
RL = PolicyGradient(
    action_dim=env.action_space.n,
    feature_dim=env.observation_space.shape[0],
    learning_rate=0.01,
    reward_decay=0.99,
    # log_path="/Users/FC/Documents/MasterProject/log"
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)  # 存储这一回合的 transition

        if done:
            ep_rs_sum = sum(RL.ep_rewards)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # 判断是否显示模拟
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # 学习, 输出 vt, 我们下节课讲这个 vt 的作用

            break

        observation = observation_
