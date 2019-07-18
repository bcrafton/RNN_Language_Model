
import numpy as np
import tensorflow as tf

from Layer import Layer

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = 1.0 - rate
        # assert(False)
        # pretty sure need to dropout the output to the LSTM cell itself.
        # which means we need to do dropout inside the cell.
        self.dist = tf.distributions.Bernoulli(probs=self.rate)

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
        dropout_mask = tf.cast(self.dist.sample(sample_shape=tf.shape(X)), dtype=tf.float32)
        # dropout_mask = tf.Print(dropout_mask, [tf.count_nonzero(dropout_mask)], message='', summarize=1000)
        A = X * dropout_mask
        cache = {'dropout': dropout_mask}
        return A, cache

    def backward(self, AI, AO, DO, cache):
        dropout_mask = cache['dropout']
        DI = DO * dropout_mask
        return DI, []
        
    def dfa(self, AI, AO, DO, cache):
        return self.backward(AI, AO, DO, cache)
        
    ###################################################################


