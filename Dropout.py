
import numpy as np
import tensorflow as tf

from Layer import Layer

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate
        # assert(False)
        # pretty sure need to dropout the output to the LSTM cell itself.
        # which means we need to do dropout inside the cell.

    ###################################################################

    def get_weights(self):
        assert (False)

    def params(self):
        return []

    def num_params(self):
        assert (False)

    ###################################################################
    # are we sure this dropout implementation is legit ???
    # saving dropout mask like this is weird...

    def forward(self, X):
        dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(X)) > self.rate, tf.float32)
        A = X * dropout_mask
        cache = {'dropout': dropout_mask}
        return A, cache

    def backward(self, AI, AO, DO, cache):
        dropout_mask = cache['dropout']
        DI = DO * dropout_mask
        return DI, []
        
    ###################################################################


