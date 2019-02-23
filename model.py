#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

import math
import random
import time
import os
from mdn2 import *

import tensorflow as tf


# In[2]:


class Model():
    
    def __init__(self, args):
        self.args = args
        self.initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
        self.learning_rate = args['learning_rate']
        self.tsteps = args['tsteps']
        self.num_mixtures = args['num_mixtures']
        self.rnn_size = args['rnn_size']
        self.batch_size = args['batch_size']
        self.biases = args['biases']
        self.grad_clip = args['grad_clip']
        self.keep_prob = args['keep_prob'] 
        self.train = args['train']
        
        # Build an LSTM cell, each cell has rnn_size number of units
        with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            cell_func = tf.contrib.rnn.LSTMCell
            self.cell0 = cell_func(args['rnn_size'], state_is_tuple=True, initializer=self.initializer)
            self.cell1 = cell_func(args['rnn_size'], state_is_tuple=True, initializer=self.initializer)
            self.cell2 = cell_func(args['rnn_size'], state_is_tuple=True, initializer=self.initializer)
            if (self.train and self.keep_prob < 1): 
                self.cell0 = tf.contrib.rnn.DropoutWrapper(self.cell0, output_keep_prob = self.keep_prob)
                self.cell1 = tf.contrib.rnn.DropoutWrapper(self.cell1, output_keep_prob = self.keep_prob)
                self.cell2 = tf.contrib.rnn.DropoutWrapper(self.cell2, output_keep_prob = self.keep_prob)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        # Placeholders for input and output data, each entry has tsteps points at a time
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, args['tsteps'], 3])
        self.output = tf.placeholder(dtype=tf.float32, shape=[None, args['tsteps'], 3])
        
        # Setting the states of memory cells in each LSTM cell.
        # batch_size is the number of training examples in a batch. Each training example is a set of tsteps number of
        # (x,y, <end_of_stroke>) tuples, i.e. a sequence of strokes till t time steps.
        self.istate_cell0 = self.cell0.zero_state(batch_size=args['batch_size'], dtype=tf.float32)
        self.istate_cell1 = self.cell1.zero_state(batch_size=args['batch_size'], dtype=tf.float32)
        self.istate_cell2 = self.cell2.zero_state(batch_size=args['batch_size'], dtype=tf.float32)
        
        # Input to model is a set of batch_size number of training samples. Step below splits by tsteps, giving one element in a tstep worth in each batch
        input_to_model = [tf.squeeze(input_, [1]) for input_ in tf.split(self.input, self.tsteps, 1)]
        
        def build_computational_graph(self, inputs, cell, initial_cell_state, scope):
            output, cell_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_cell_state, cell, loop_function=None, scope=scope)
            return [output, cell_final_state]
        
        outs_layer0, self.cell0_final_state = build_computational_graph(self, input_to_model, self.cell0, self.istate_cell0, 'cell0')
        outs_layer1, self.cell1_final_state = build_computational_graph(self, outs_layer0, self.cell1, self.istate_cell1, 'cell1')
        outs_layer2, self.cell2_final_state = build_computational_graph(self, outs_layer1, self.cell2, self.istate_cell2, 'cell2')
        
        # The output of final layer goes into MDN
        # for each output we predict 6 parameters + 1 eos for "num_mixtures" mixture density components
        total_mdn_params = 6 * self.num_mixtures + 1
        
        # according to eq(17) in https://arxiv.org/pdf/1308.0850.pdf
        with tf.variable_scope('mdn_dense'):
            # initializing W and b matrix for eq(17)
            W_to_mdn = tf.get_variable("output_w", [self.rnn_size, total_mdn_params], initializer=self.initializer)
            b_to_mdn = tf.get_variable("output_b", [total_mdn_params], initializer=self.initializer)
            
        outs_layer2 = tf.reshape(tf.concat(outs_layer2, 1), [-1, args['rnn_size']])
        output_layer = tf.nn.xw_plus_b(outs_layer2, W_to_mdn, b_to_mdn) #eq(17) in https://arxiv.org/pdf/1308.0850.pdf
        flat_output = tf.reshape(self.output,[-1, 3])
        [output_x, output_y, eos_data] = tf.split(flat_output, 3, 1)
        
        # MDN
        [self.eos, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho] = get_mdn_coef(self, output_layer)
        loss = get_loss(self.pi, output_x, output_y, eos_data, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.eos)
        self.cost = loss / (self.batch_size * self.tsteps) # J = 1/m*sum(Loss) , m = number of training examples
        
        # Using RMSProp optimizer for optimizing the cost, defining it's parameters
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.decay = tf.Variable(0.0, trainable=False)
        self.momentum = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
        
        # Gradient clipping
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), self.grad_clip)
        self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_variables))

        self.sess = tf.InteractiveSession()
        # saver for saving the model's variables
        self.saver = tf.train.Saver(tf.global_variables())
        self.sess.run(tf.global_variables_initializer())
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




