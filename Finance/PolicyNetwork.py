import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# This need to be verified according to experiment
an_config = {
    "state_size": 10,
    "action_size": 3,
    "batch_size": 100,
    "learning_rate": 0.0001,
    "tau": 0.01,

    "hidden_layer": 2,
    "h0": 128,
    "h1": 128
}

cn_config = {
    "state_size": 10,
    "action_size": 3,
    "batch_size": 100,
    "learning_rate": 0.0001,
    "tau": 0.01,

    "hidden_layer": 3,
    "h0": 128,
    "h1": 128,
    "h2": 128
}


class ActorNetwork(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess  # This is used to optimize GPU
        self.state_size = config["state_size"]
        self.action_size = config["action_size"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_network()
        self.target_model, self.target_weights, self.target_state = self.create_network()
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        print("Build ActorNetwork")
        State = Input(shape=[self.config["state_size"]], name="State_Input")
        h0 = Dense(self.config["h0"], activation='relu', name="Hidden_Layer0")(State)
        h1 = Dense(self.config["h1"], activation='relu', name="Hidden_Layer1")(h0)
        Action = Dense(self.config["action_size"], activation="softmax", name="Action_Output")(h1)
        model = Model(input=State, output=Action)
        return model, model.trainable_weights, State

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def describe(self):
        print("ActorNetwork:")
        self.model.summary()
        plot_model(self.model, "actor_network.png", show_shapes=True, show_layer_names=True)


class CriticNetwork(object):
    def __init__(self, sess, config):
        self.config = config
        self.sess = sess  # This is used to optimize GPU
        self.state_size = config["state_size"]
        self.action_size = config["action_size"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_network()
        self.target_model, self.target_action, self.target_state = self.create_network()
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def create_network(self):
        print("Build CriticNetwork")
        State = Input(shape=[self.config["state_size"]], name="State_Input")
        Action = Input(shape=[self.config["action_size"]], name="Action_Input")

        hs1 = Dense(self.config["h0"], activation='relu', name="State_Hidden_Layer")(State)
        ha1 = Dense(self.config["h0"], activation='linear', name="Action_Hidden_Layer")(Action)

        h1 = Dense(self.config["h1"], activation='linear', name="Hidden_Layer0")(hs1)
        h2 = merge([h1, ha1], mode='sum', name="Hidden_Layer1")
        h3 = Dense(self.config['h2'], activation='relu', name="Hidden_Layer2")(h2)
        Q = Dense(self.config["action_size"], activation="linear", name="Q_Value")(h3)

        model = Model(input=[State, Action], output=Q)
        adam = Adam(lr=self.config["learning_rate"])
        model.compile(loss='mse', optimizer=adam)
        return model, Action, State

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def describe(self):
        print("CriticNetwork:")
        self.model.summary()
        plot_model(self.model, "critic_network.png", show_shapes=True, show_layer_names=True)
