import sys
import numpy as np
import tensorflow as tf
import yaml
from Finance.Env import Loader, Env
from Finance.ReplayBuffer import ReplayBuffer
from RL.PolicyGradient import PolicyGradient as Agent

sys.path.append("./")

# Load env config
model_config = yaml.load(open("config/model.yaml"))

# Load data
loader = Loader("data/")
data, info = loader.load("000300")
# data_DDRLFFSR is start from 2014.1.1, the 407760th point in data
data_DDRLFFSR = data[407760:407760 + 20000]

# Use first 15000 point to train model
train = data_DDRLFFSR[:15000].close.values
test = data_DDRLFFSR[15000:].close.values

env_train = Env(train, action_dim=model_config['action_dim'], observation_dim=model_config['state_dim'])
env_test = Env(test, action_dim=model_config['action_dim'], observation_dim=model_config['state_dim'])
env = env_train

agent = Agent(
    action_dim=env.action_dim,
    feature_dim=env.observation_dim,
    learning_rate=0.001,
    reward_decay=0.99,
    log_path="./log"
)

# Training
EPISODE = 2000
TRADING_RANGE = 20000  # minutes

# epsilon-greedy
EXPLORE = 10000.
epsilon = 1.

for i_episode in range(EPISODE):

    observation = env.reset()

    while True:

        action = agent.choose_action(observation)

        observation_, reward, done = env.step(action)

        agent.store_transition(observation, action, reward)  # 存储这一回合的 transition

        if done:
            ep_rs_sum = sum(agent.ep_rewards)

            print("episode:", i_episode, "  reward:", ep_rs_sum)

            vt = agent.learn()  # 学习

            break

        observation = observation_

    # run test for every 10 episode
    if i_episode % 10 == 0:
        observation = env_test.reset()
        done = False
        ep_rs_sum = 0
        while not done:
            action = agent.choose_action(observation)
            _, reward, done = env_test.step(action)
            ep_rs_sum += reward
        print("test after episode:", i_episode, " reward:", ep_rs_sum)
