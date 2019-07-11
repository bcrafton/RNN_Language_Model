
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    ###################################################################

    def get_weights(self):
        assert(False)
        
    def num_params(self):
        assert(False)

    ###################################################################

    def forward(self, X):
        assert(False)    
        
    def backward(self, AI, AO, DO):    
        assert(False)

    def train(self, AI, AO, DO):    
        assert(False)
                
    ###################################################################   
