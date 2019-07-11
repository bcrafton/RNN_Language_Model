
import tensorflow as tf
import numpy as np

from Layer import Layer 

from init_matrix import init_matrix

########################################

def tanh(x):
    return tf.tanh(x)

def dtanh(x):
    # wrt A, not Z
    return 1 - tf.pow(x, 2)

def sigmoid(x):
    return tf.sigmoid(x)

def dsigmoid(x):
    # wrt A, not Z
    return x * (1 - x)
    
########################################

class LSTM(Layer):

    def __init__(self, input_shape, size, init='glorot_normal', return_sequences=True):
        self.input_shape = input_shape
        self.time_size, self.batch_size, self.input_size = self.input_shape
        self.output_size = size
        self.init = init
        self.return_sequences = return_sequences
        
        # print (self.time_size, self.batch_size, self.input_size, self.output_size)
        
        Wa_x = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        Wi_x = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        Wf_x = init_matrix(size=(self.input_size, self.output_size), init=self.init)
        Wo_x = init_matrix(size=(self.input_size, self.output_size), init=self.init)

        Wa_h = init_matrix(size=(self.output_size, self.output_size), init=self.init)
        Wi_h = init_matrix(size=(self.output_size, self.output_size), init=self.init)
        Wf_h = init_matrix(size=(self.output_size, self.output_size), init=self.init)
        Wo_h = init_matrix(size=(self.output_size, self.output_size), init=self.init)
        
        ba = np.zeros(shape=self.output_size)
        bi = np.zeros(shape=self.output_size)
        bf = np.zeros(shape=self.output_size)
        bo = np.zeros(shape=self.output_size)
        
        self.Wa_x = tf.Variable(Wa_x, dtype=tf.float32)
        self.Wi_x = tf.Variable(Wi_x, dtype=tf.float32)
        self.Wf_x = tf.Variable(Wf_x, dtype=tf.float32)
        self.Wo_x = tf.Variable(Wo_x, dtype=tf.float32)

        self.Wa_h = tf.Variable(Wa_h, dtype=tf.float32)
        self.Wi_h = tf.Variable(Wi_h, dtype=tf.float32)
        self.Wf_h = tf.Variable(Wf_h, dtype=tf.float32)
        self.Wo_h = tf.Variable(Wo_h, dtype=tf.float32)
        
        self.ba = tf.Variable(ba, dtype=tf.float32)
        self.bi = tf.Variable(bi, dtype=tf.float32)
        self.bf = tf.Variable(bf, dtype=tf.float32)
        self.bo = tf.Variable(bo, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        assert (False)

    def num_params(self):
        assert (False)

    ###################################################################

    def forward(self, X):
        # put some tensorflow assertion here to check the shape
        '''
        if (np.shape(X) != (self.time_size, self.batch_size, self.input_size)):
            print (np.shape(X))
            assert(np.shape(X) == (self.time_size, self.batch_size, self.input_size))
        '''
        
        la = [None] * self.time_size
        li = [None] * self.time_size
        lf = [None] * self.time_size
        lo = [None] * self.time_size
        ls = [None] * self.time_size
        lh = [None] * self.time_size
        
        for t in range(self.time_size):
            x = X[t]
            
            if t == 0:
                a = tanh(   tf.matmul(x, self.Wa_x) + self.ba) 
                i = sigmoid(tf.matmul(x, self.Wi_x) + self.bi) 
                f = sigmoid(tf.matmul(x, self.Wf_x) + self.bf)
                o = sigmoid(tf.matmul(x, self.Wo_x) + self.bo) 
            else:
                a = tanh(   tf.matmul(x, self.Wa_x) + tf.matmul(lh[t-1], self.Wa_h) + self.ba)
                i = sigmoid(tf.matmul(x, self.Wi_x) + tf.matmul(lh[t-1], self.Wi_h) + self.bi)
                f = sigmoid(tf.matmul(x, self.Wf_x) + tf.matmul(lh[t-1], self.Wf_h) + self.bf)
                o = sigmoid(tf.matmul(x, self.Wo_x) + tf.matmul(lh[t-1], self.Wo_h) + self.bo)

            la[t] = a
            li[t] = i
            lf[t] = f
            lo[t] = o
            
            if t == 0:
                s = a * i               
            else:
                s = a * i + ls[t-1] * f
                
            h = tanh(s) * o

            ls[t] = s
            lh[t] = h

        # [T, B, O]
        outputs = tf.stack(lh, axis=0)

        cache = {}
        cache['a'] = la
        cache['i'] = li
        cache['f'] = lf
        cache['o'] = lo
        cache['s'] = ls
        cache['h'] = lh
        
        if self.return_sequences:
            return outputs, cache
        else:
            return outputs[-1], cache


    # combining backward and train together
    def backward(self, AI, AO, DO, cache):
        if not self.return_sequences:
            zeros = tf.zeros(shape=(self.time_size - 1, self.batch_size, self.output_size))
            DO = tf.reshape(DO, (1, self.batch_size, self.output_size))
            DO = tf.concat((zeros, DO), axis=0)
    
        a = cache['a'] 
        i = cache['i'] 
        f = cache['f'] 
        o = cache['o'] 
        s = cache['s'] 
        h = cache['h'] 
        
        lds = [None] * self.time_size
        ldx = [None] * self.time_size
        
        dWa_x = tf.zeros_like(self.Wa_x)
        dWi_x = tf.zeros_like(self.Wi_x)
        dWf_x = tf.zeros_like(self.Wf_x)
        dWo_x = tf.zeros_like(self.Wo_x)

        dWa_h = tf.zeros_like(self.Wa_h)
        dWi_h = tf.zeros_like(self.Wi_h)
        dWf_h = tf.zeros_like(self.Wf_h)
        dWo_h = tf.zeros_like(self.Wo_h)

        dba = tf.zeros_like(self.ba)
        dbi = tf.zeros_like(self.bi)
        dbf = tf.zeros_like(self.bf)
        dbo = tf.zeros_like(self.bo)

        for t in range(self.time_size-1, -1, -1):
            if t == 0:
                dh = DO[t] + dout
                ds = dh * o[t] * dtanh(tanh(s[t])) + lds[t+1] * f[t+1]
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = tf.zeros_like(da)
                do = dh * tanh(s[t]) * dsigmoid(o[t]) 
            elif t == self.time_size-1:
                dh = DO[t]
                ds = dh * o[t] * dtanh(tanh(s[t]))
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = ds * s[t-1] * dsigmoid(f[t]) 
                do = dh * tanh(s[t]) * dsigmoid(o[t]) 
            else:
                dh = DO[t] + dout
                ds = dh * o[t] * dtanh(tanh(s[t])) + lds[t+1] * f[t+1]
                da = ds * i[t] * dtanh(a[t])
                di = ds * a[t] * dsigmoid(i[t]) 
                df = ds * s[t-1] * dsigmoid(f[t]) 
                do = dh * tanh(s[t]) * dsigmoid(o[t]) 

            dout_a = tf.matmul(da, tf.transpose(self.Wa_h))
            dout_i = tf.matmul(di, tf.transpose(self.Wi_h))
            dout_f = tf.matmul(df, tf.transpose(self.Wf_h))
            dout_o = tf.matmul(do, tf.transpose(self.Wo_h))
            dout = dout_a + dout_i + dout_f + dout_o
            
            dx_a = tf.matmul(da, tf.transpose(self.Wa_x))
            dx_i = tf.matmul(di, tf.transpose(self.Wi_x))
            dx_f = tf.matmul(df, tf.transpose(self.Wf_x))
            dx_o = tf.matmul(do, tf.transpose(self.Wo_x))
            dx = dx_a + dx_i + dx_f + dx_o
            
            dWa_x += tf.matmul(tf.transpose(AI[t]), da)
            dWi_x += tf.matmul(tf.transpose(AI[t]), di)
            dWf_x += tf.matmul(tf.transpose(AI[t]), df)
            dWo_x += tf.matmul(tf.transpose(AI[t]), do)
            
            dba += tf.reduce_sum(da, axis=0)
            dbi += tf.reduce_sum(di, axis=0)
            dbf += tf.reduce_sum(df, axis=0) 
            dbo += tf.reduce_sum(do, axis=0)
            
            if t > 0:
                dWa_h += tf.matmul(tf.transpose(h[t-1]), da)
                dWi_h += tf.matmul(tf.transpose(h[t-1]), di)
                dWf_h += tf.matmul(tf.transpose(h[t-1]), df)
                dWo_h += tf.matmul(tf.transpose(h[t-1]), do)
                
            lds[t] = ds
            ldx[t] = dx

        dw = [
        (dWa_x, self.Wa_x), (dWi_x, self.Wi_x), (dWf_x, self.Wf_x), (dWo_x, self.Wo_x),
        (dWa_h, self.Wa_h), (dWi_h, self.Wi_h), (dWf_h, self.Wf_h), (dWo_h, self.Wo_h),
        (dba, self.ba), (dbi, self.bi), (dbf, self.bf), (dbo, self.bo)
        ]

        return ldx, dw

    ###################################################################

        
