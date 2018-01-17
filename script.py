import sys

import numpy as np
import tensorflow as tf
import yaml

sys.path.append("./")
from Finance.Env import Loader, Env
from Finance.ReplayBuffer import ReplayBuffer

# Load data
loader = Loader("data/")
data, info = loader.load("000300")
# data_DDRLFFSR is start from 2014.1.1, the 407760th point in data
data_DDRLFFSR = data[407760:407760 + 20000]

# Use first 15000 point to train AC-model
train = data_DDRLFFSR[:15000].close.values
test = data_DDRLFFSR[15000:].close.values

env = Env(data[407760:].close.values)


# Build Actor-Critic Model
from RL.PolicyNetwork import ActorNetwork, CriticNetwork, Agent

# Load Network Config
model_config = yaml.load(open("config/model.yaml"))
an_config = yaml.load(open("config/an_config.yaml"))
an_config.update(model_config)

cn_config = yaml.load(open("config/cn_config.yaml"))
cn_config.update(model_config)

# Optimize GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Build Network
actor = ActorNetwork(sess, an_config)
critic = CriticNetwork(sess, cn_config)


agent = Agent(actor, critic)

agent.actor.describe()
agent.critic.describe()


# Training
EPISODE = 2000
BUFFER_SIZE = 4096
BATCH_SIZE = 128
TRADING_RANGE = 20000  # minutes

# epsilon-greedy
EXPLORE = 10000.
epsilon = 1.

state_size = model_config["state_size"]
action_size = model_config["action_size"]

env = Env(data[407760:].close.values)
buffer = ReplayBuffer(BUFFER_SIZE)

for i in range(EPISODE):
    print("Eposide: " + str(i) + "/" + str(EPISODE))

    # Reset env, total reward, hold
    hold = -1
    total_reward = 0
    state = env.reset()
    state = np.append(state, hold)
    for j in range(TRADING_RANGE):
        if j % 1000 == 0:
            print("Step: " + str(int(j / 1000)) + "/" + str(int(TRADING_RANGE / 1000)) + "k")
        epsilon -= 1. / EXPLORE
        action = agent.generateAction(state, epsilon)
        new_state, reward, _ = env.step(action, hold)
        new_state = np.append(new_state, action * hold)

        buffer.add(state, action, reward, new_state, False)

        # Do the batch update
        batch = buffer.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])

        # TODO: The sample code use new_state instead of states to calculate
        # Q value
        Q_from_target = agent.critic.target_model.predict([new_states, agent.actor.target_model.predict(new_states)])

        loss = 0
        loss += agent.critic.model.train_on_batch([states, actions], Q_from_target)
        a_for_grad = agent.actor.model.predict(states)
        grads = agent.critic.gradients(states, a_for_grad)
        agent.actor.train(states, grads)
        agent.actor.target_train()
        agent.critic.target_train()

        total_reward += reward
        state = new_state

    print("Eposide: " + str(i) + "done. Total reward: " + str(total_reward))

