
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Model:
    def __init__(self, batch_size, time_size, layers):
        self.batch_size = batch_size
        self.time_size = time_size
        self.num_layers = len(layers)
        self.layers = layers
        
    def get_weights(self):
        weights = {}
        for ii in range(self.num_layers):
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value
            
        return weights
        
    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum
        
    def train(self, X, Y):
        A = [None] * self.num_layers
        C = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], C[ii] = l.forward(X)
            else:
                A[ii], C[ii] = l.forward(A[ii-1])

        N = tf.shape(A[self.num_layers-1])[0]
        N = tf.cast(N, dtype=tf.float32)

        pred = A[self.num_layers-1]

        '''
        E = (tf.nn.softmax(pred) - Y) / N
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=pred)
        '''
        zeros = tf.cast(tf.sign(X), dtype=tf.float32)
        zeros = tf.reshape(zeros, [self.batch_size, self.time_size, 1])
        E = ((tf.nn.softmax(pred) - Y) / N) * zeros
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Y, axis=2), logits=pred) * tf.reshape(zeros, [self.batch_size, self.time_size]))

        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii], G = l.backward(A[ii-1], A[ii], E, C[ii])
                grads_and_vars.extend(G)
            elif (ii == 0):
                D[ii], G = l.backward(X, A[ii], D[ii+1], C[ii])
                grads_and_vars.extend(G)
            else:
                D[ii], G = l.backward(A[ii-1], A[ii], D[ii+1], C[ii])
                grads_and_vars.extend(G)
                
        return grads_and_vars, loss
              
    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], _ = l.forward(X)
            else:
                A[ii], _ = l.forward(A[ii-1])
                
        return A[self.num_layers-1]
        
        
        
        
        
        
        
        