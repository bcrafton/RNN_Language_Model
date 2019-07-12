
import numpy as np
import tensorflow as tf

from Layer import Layer 
from Activation import Activation
from Activation import Linear

from init_matrix import init_matrix

class Dense(Layer):

    def __init__(self, input_shape, size, activation=None, init=None, lr=0.001):
    
        self.input_shape = input_shape
        self.size = size
        self.init = init
        self.activation = Linear() if activation == None else activation
        self.lr = lr

        bias = np.zeros(shape=size)
        weights = init_matrix(size=(self.input_shape, self.size), init=self.init)
        
        self.bias = tf.Variable(bias, dtype=tf.float32)
        self.weights = tf.Variable(weights, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        assert (False)

    def num_params(self):
        assert (False)

    ###################################################################

    def forward(self, X):

        X = tf.Print(X, [tf.shape(X)], message='Dense: ', summarize=1000)

        shape = tf.shape(X)
        X = tf.reshape(X, [-1, self.input_shape])
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        A = tf.reshape(A, [shape[0], shape[1], self.size])

        # A = tf.Print(A, [shape, tf.shape(A)], message='', summarize=1000)

        return A, None
            
    def backward(self, AI, AO, DO, cache):
        shape_in = tf.shape(AI)
        shape_out = tf.shape(AO)

        AO = tf.reshape(AO, [-1, self.size])
        AI = tf.reshape(AI, [-1, self.input_shape])
        DO = tf.reshape(DO, [-1, self.size])

        DO = DO * self.activation.gradient(AO)
        DI = tf.matmul(DO, tf.transpose(self.weights))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)

        DI = tf.reshape(DI, [shape_in[0], shape_in[1], self.input_shape])

        return DI, [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################

        
        
