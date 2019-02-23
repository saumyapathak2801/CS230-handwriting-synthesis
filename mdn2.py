#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np


# In[3]:


def get_mdn_coef(model, Z):
    # eq 18...23 of http://arxiv.org/abs/1308.0850
    eos_hat = Z[:, 0:1] #end of sentence tokens
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(Z[:, 1:], 6, 1)
    model.pi_hat, model.sigma1_hat, model.sigma2_hat = pi_hat, sigma1_hat, sigma2_hat 

    eos = tf.sigmoid(-1*eos_hat) 
    pi = tf.nn.softmax(pi_hat) # softmax z_pi:
    mu1 = mu1_hat; mu2 = mu2_hat 
    sigma1 = tf.exp(sigma1_hat); sigma2 = tf.exp(sigma2_hat) # exp for sigmas
    rho = tf.tanh(rho_hat) 
    return [eos, pi, mu1, mu2, sigma1, sigma2, rho]

def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
    # define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
    x_mu1 = tf.subtract(x1, mu1)
    x_mu2 = tf.subtract(x2, mu2)
    Z = tf.square(tf.div(x_mu1, s1)) + tf.square(tf.div(x_mu2, s2)) - 2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
    rho_square_term = 1-tf.square(rho)
    power_e = tf.exp(tf.div(-Z,2*rho_square_term))
    regularize_term = 2*np.pi*tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
    gaussian = tf.div(power_e, regularize_term)
    return gaussian

def get_loss(pi, x1_data, x2_data, eos_data, mu1, mu2, sigma1, sigma2, rho, eos):
    # define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
    gaussian = gaussian2d(x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho)
    term1 = tf.multiply(gaussian, pi)
    term1 = tf.reduce_sum(term1, 1, keep_dims=True) #do inner summation
    term1 = -tf.log(tf.maximum(term1, 1e-20)) # some errors are zero -> numerical errors.

    term2 = tf.multiply(eos, eos_data) + tf.multiply(1-eos, 1-eos_data) # Bernoulli -> eos probability
    term2 = -tf.log(term2) #negative log error gives loss

    return tf.reduce_sum(term1 + term2) # outer summation


# In[ ]:




