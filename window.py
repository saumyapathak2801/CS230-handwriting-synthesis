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


# Build a gaussian character window
def build_gaussian_window(alpha, beta, kappa, c):
    # Character sequence time steps
    sequence_steps = c.get_shape()[1].value
    phi = get_phi(alpha, beta, kappa, sequence_steps)
    window = tf.matmul(phi, c)
    window = tf.squeeze(window, [1])
    return window, phi

# Returns the phi (the weight for each character window mixture)
def get_phi(alpha, beta, kappa, char_steps):
    u = np.linspace(0, char_steps - 1, char_steps)
    kappa_term = tf.square(kappa - u)
    exp_term = tf.exp(-1*beta*kappa_term)
    phi = tf.reduce_sum(tf.multiply(alpha, exp_term), 1, keep_dims = True)
    return phi

def get_window_coefficients(out_cell0, kmixtures, prev_kappa, initializer, reuse):
    hidden = out_cell0.get_shape()[1] # 
    abk_out = 3*kmixtures
    print(hidden, abk_out)
    with tf.variable_scope('window',reuse=reuse):
        window_w = tf.get_variable("window_w", [hidden, abk_out], initializer=initializer)  # [?, abk_out]
        window_b = tf.get_variable("window_b", [abk_out], initializer=initializer) # [abk_out]
    abk_hat = tf.nn.xw_plus_b(out_cell0, window_w, window_b) # [abk_out, 1]
    # Compute values of alpha beta kappa.
    abk = tf.exp(tf.reshape(abk_hat, [-1, 3*kmixtures, 1]))
    alpha, beta, kappa = tf.split(abk, 3, 1)
    kappa = kappa + prev_kappa
    return alpha, beta, kappa # [?, kmixtures, 1]
    


# In[ ]:




