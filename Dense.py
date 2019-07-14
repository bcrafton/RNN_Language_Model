
import numpy as np
import tensorflow as tf

from Layer import Layer 
from Activation import Activation
from Activation import Linear

from init_matrix import init_matrix

class Dense(Layer):

    def __init__(self, input_shape, size, activation=None, init='glorot_normal', name=None):
        self.input_shape = input_shape
        self.batch_size, self.time_size, self.input_size = self.input_shape
        self.output_size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.name = name

        bias = init_matrix(size=(self.output_size, 1), init=self.init)
        bias = np.reshape(bias, [self.output_size])
        weights = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        
        self.bias = tf.Variable(bias, dtype=tf.float32)
        self.weights = tf.Variable(weights, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    ###################################################################

    def forward(self, X):
        # X = tf.Print(X, [tf.shape(X)], message='Dense in: ', summarize=1000)
        X = tf.reshape(X, [self.batch_size * self.time_size, self.input_size])
        
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        
        A = tf.reshape(A, [self.batch_size, self.time_size, self.output_size])
        # A = tf.Print(A, [tf.shape(A)], message='Dense out: ', summarize=1000)

        return A, None
            
    def backward(self, AI, AO, DO, cache):
        AI = tf.reshape(AI, [self.batch_size * self.time_size, self.input_size])
        AO = tf.reshape(AO, [self.batch_size * self.time_size, self.output_size])
        DO = tf.reshape(DO, [self.batch_size * self.time_size, self.output_size])
        DO = DO * self.activation.gradient(AO)

        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DI = tf.reshape(DI, [self.batch_size, self.time_size, self.input_size])
        
        return DI, [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################

        
        
