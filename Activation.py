
import numpy as np
import tensorflow as tf

class Activation(object):
    def forward(self, x):
        assert(False)

    def gradient(self, x):
        assert(False)
        
class Sigmoid(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return tf.sigmoid(x)

    def gradient(self, x):
        return tf.multiply(x, tf.subtract(1.0, x))
        
class Relu(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return tf.nn.relu(x)

    def gradient(self, x):
        # pretty sure this gradient works for A and Z
        return tf.cast(x > 0.0, dtype=tf.float32)

class Tanh(Activation):
# https://theclevermachine.wordpress.com/tag/tanh-function/ 

    def __init__(self):
        pass

    def forward(self, x):
        return tf.tanh(x)

    def gradient(self, x):
        # this is gradient wtf A, not Z
        return 1 - tf.pow(x, 2)
        
class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return x 

    def gradient(self, x):
        return tf.ones(shape=tf.shape(x))
       
        
        
        
        
