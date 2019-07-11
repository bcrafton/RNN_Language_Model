
import tensorflow as tf

import numpy as np

from Layer import Layer 
from init_matrix import init_matrix

class Embedded(Layer):

    def __init__(self, input_size, output_size, name=None, load=None, train=True):
    
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.load = load
        self.train_flag = train
        
        assert(not self.load)
        assert(self.train_flag)

        weights = init_matrix(size=(self.input_size, self.output_size), init='glorot_normal')
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
        # shape i = (64, 49)
        # shape o = (64, 49) # this output after a softmax tho.
        
        A = tf.nn.embedding_lookup(self.weights, X)
        return A
            
    def backward(self, AI, AO, DO):
        return None
        
    def gv(self, AI, AO, DO):
        # one of the 4 set branches:
        # https://github.com/bcrafton/dfa/blob/set_conv_sign/SparseFC.py
        
        # gonna need one of these two.
        # https://www.tensorflow.org/api_docs/python/tf/scatter_update
        # https://www.tensorflow.org/api_docs/python/tf/scatter_nd
        # think these are same thing ... they dont do +=
    
        DW = tf.scatter_update(tf.zeros_like(w), AI, DO)
        return [(DW, self.weights)]

    ###################################################################
        
        
        
