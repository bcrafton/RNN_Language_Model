
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Model:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        
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
        E = (tf.nn.softmax(A[self.num_layers-1]) - Y) / N

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
                
        return grads_and_vars
              
    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], _ = l.forward(X)
            else:
                A[ii], _ = l.forward(A[ii-1])
                
        return A[self.num_layers-1]
        
        
        
        
        
        
        
        
