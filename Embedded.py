
import tensorflow as tf

import numpy as np

from Layer import Layer 
from init_matrix import init_matrix

class Embedded(Layer):

    def __init__(self, input_shape, output_size, init='glorot_normal', name=None):
        self.input_shape = input_shape
        self.batch_size, self.time_size, self.input_size = self.input_shape
        self.output_size = output_size
        self.init = init
        self.name = name

        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        self.weights = tf.Variable(weights, dtype=tf.float32)

        zeros = np.zeros(shape=(self.input_size, self.output_size))
        self.zeros = tf.Variable(zeros, dtype=tf.float32)

    ###################################################################

    def get_weights(self):
        assert(self.name is not None)
        return [(self.name, self.weights)]

    def params(self):
        # return []
        return [self.weights]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        return weights_size

    ###################################################################

    def forward(self, X):
        # shape in = (32, 64) ... should be.
        
        shape = tf.shape(X)
        X = tf.reshape(X, [self.batch_size * self.time_size])
        A = tf.nn.embedding_lookup(self.weights, X)
        A = tf.reshape(A, [self.batch_size, self.time_size, self.output_size])

        return A, None
            
    def backward(self, AI, AO, DO, cache):
        # one of the 4 set branches:
        # https://github.com/bcrafton/dfa/blob/set_conv_sign/SparseFC.py

        # gonna need one of these two.
        # https://www.tensorflow.org/api_docs/python/tf/scatter_update
        # https://www.tensorflow.org/api_docs/python/tf/scatter_nd
        # think these are same thing ... they dont do +=

        AI = tf.reshape(AI, [self.batch_size * self.time_size])
        DO = tf.reshape(DO, [self.batch_size * self.time_size, self.output_size])

        DW = tf.scatter_update(self.zeros, AI, DO)
        # DW = tf.zeros_like(self.weights)
        # DW = tf.Print(DW, [tf.shape(AI), tf.shape(DO)], message='', summarize=1000)

        return None, []
        # return None, [(DW, self.weights)]

    ###################################################################
        
        
        
