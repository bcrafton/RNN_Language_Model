
import numpy as np
import tensorflow as tf

from Layer import Layer

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate
        assert(False)
        # pretty sure need to dropout the output to the LSTM cell itself.
        # which means we need to do dropout inside the cell.

    ###################################################################

    def get_weights(self):
        assert (False)

    def num_params(self):
        assert (False)

    ###################################################################
    # are we sure this dropout implementation is legit ???
    # saving dropout mask like this is weird...

    def forward(self, X):
        self.dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(X)) > self.rate, tf.float32)
        A = X * self.dropout_mask
        return A, None

    def backward(self, AI, AO, DO, cache):
        DI = DO * self.dropout_mask
        return DI, []
        
    ###################################################################


