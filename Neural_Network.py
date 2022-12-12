import tensorflow as tf

from keras import layers, Input
from keras.layers import LSTMCell
import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def Model(images, ht, ct):
    input = tf.compat.v1.to_float(images) / 255.0
    # conv1=(input)
    conv1 = layers.Conv2D(16, 8, 4, activation="relu", name="conv1")(input)
    conv2 = layers.Conv2D(32, 4, 2, activation="relu", name="conv2")(conv1)
    flat = layers.Flatten()(conv2)
    fully_connected = layers.Dense(256, activation="relu", name='fully')(flat)
    lstm = LSTMCell(256)
    lstm_out, (ht, ct) = lstm(fully_connected, states=[ht, ct])
    return lstm_out, ht, ct


class Network():
    def __init__(self, output_num, scope, trainer, reg=0.01):
        with tf.compat.v1.variable_scope(scope):
            self.ht = tf.zeros((1, 256))
            self.ct = tf.zeros((1, 256))
            self.num_of_output = output_num
            self.states = Input(shape=[84, 84, 4], dtype=tf.uint8, name="state")
            self.advantages = Input(shape=[], dtype=tf.float32, name="advantages")
            self.actions = Input(shape=[], dtype=tf.int32, name="actions")
            self.targets = Input(shape=[], dtype=tf.float32, name="target")
            with tf.compat.v1.variable_scope("network"):
                
                lstm, self.ht, self.ct = Model(self.states, self.ht, self.ct)
            with tf.compat.v1.variable_scope("policy_network"):
                # self.trainer1 = trainer
                self.output = layers.Dense(self.num_of_output, activation=None, name="policy")(lstm)
                self.probilitys = tf.nn.softmax(self.output)
                self.probilitys = tf.clip_by_value(self.probilitys, 1e-20, 1)
                cdist = tf.compat.v1.distributions.Categorical(logits=self.output)
                self.sample_action = cdist.sample()

                # if scope != 'Global':

                self.entropy = -tf.reduce_sum(self.probilitys * tf.math.log(self.probilitys),
                                              axis=1)
                batch_size = tf.shape(self.states)[0]
                gather_indices = tf.range(batch_size) * tf.shape(self.probilitys)[1] + self.actions
                self.selected_action_probs = tf.gather(tf.reshape(self.probilitys, [-1]), gather_indices)
                self.loss = tf.math.log(self.selected_action_probs) * self.advantages + self.entropy * reg
                self.loss = -tf.reduce_sum(self.loss, name="policy_loss")
                self.trainer1 = tf.compat.v1.train.RMSPropOptimizer(0.0001, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.trainer1.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

            with tf.compat.v1.variable_scope("value_network"):
                # self.trainer2 = trainer
                self.vhat = layers.Dense(1, activation=None, name="value")(lstm)
                self.vhat = tf.compat.v1.squeeze(self.vhat, squeeze_dims=[1])
                # if scope != 'Global':

                self.vloss = tf.compat.v1.squared_difference(self.vhat, self.targets)
                self.vloss = tf.reduce_sum(self.vloss, name="value_loss")
                self.trainer2 = tf.compat.v1.train.RMSPropOptimizer(0.0001, 0.99, 0.0, 1e-6)
                self.v_grads_and_vars = self.trainer2.compute_gradients(self.vloss)
                self.v_grads_and_vars = [[grad, var] for grad, var in self.v_grads_and_vars if grad is not None]


def Create_Network(num_of_output, scope, trainer):
    return Network(num_of_output, scope=scope, trainer=trainer)


