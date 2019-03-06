#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import os as os

from dataloader import *

def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def sample(input_text, model, args):
    # initialize some parameters
    args['tsteps'] = 1000 # produce tsteps worth of samples
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
    kappa = np.zeros((1, args['window_mixtures'], 1))
    
    # Convert input string to one hot.
    input_oh = convert_to_one_hot(input_text, model.char_steps, args['alphabet'])

    prev_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)     # start with a pen stroke at (0,0)
    strokes, pis, char_to_plot, windows, kappas, phis = [], [], [], [], [], [] 

    finished = False ; i = 0
    while not finished:
        feed = {model.input: prev_x, model.char_seq: input_oh, model.kappa_start: kappa,                 model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2,                 model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
        fetch = [model.pi_hat, model.mu1, model.mu2, model.sigma1_hat, model.sigma2_hat, model.rho, model.eos,                  model.window, model.phi, model.kappa, model.alpha,                  model.cell0_final_state.c, model.cell1_final_state.c, model.cell2_final_state.c,                 model.cell0_final_state.h, model.cell1_final_state.h, model.cell2_final_state.h]
        [pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho, eos,                  window, phi, kappa, alpha,                  c0, c1, c2, h0, h1, h2] = model.sess.run(fetch, feed)
        
        # bias
        sigma1 = np.exp(sigma1_hat - args['biases']) ; sigma2 = np.exp(sigma2_hat - args['biases']) # eqn 61 and 62
        pi_hat *= 1 + args['biases'] 
        pi = np.zeros_like(pi_hat) 
        pi[0] = np.exp(pi_hat[0]) / np.sum(np.exp(pi_hat[0]), axis=0) # softmax
        
        # choose a component from the MDN
        idx = np.random.choice(pi.shape[1], p=pi[0])
        a = eos[0][0]
        eos = 1 if 0.02 < eos[0][0] else 0 # use an arbitrary boundary for detecting eos
        x1, x2 = sample_gaussian2d(mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx])
        
        # Append MDN information at every timestep
        windows.append(window)
        phis.append(phi[0])
        kappas.append(kappa[0].T)
        
        
        pis.append(pi[0])
        print([mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx], a])
        strokes.append([mu1[0][idx], mu2[0][idx], sigma1[0][idx], sigma2[0][idx], rho[0][idx], eos])
        char_to_plot.append([x1, x2, eos])
        
        # sample tsteps number of strokes
        finished = True if i > args['tsteps'] else False
        
        # new input is previous output
        prev_x[0][0] = np.array([x1, x2, eos], dtype=np.float32)
        i+=1

    windows = np.vstack(windows)
    phis = np.vstack(phis)
    kappas = np.vstack(kappas)
    strokes = np.vstack(strokes)
    char_to_plot = np.vstack(char_to_plot)

    # the network predicts the displacements between pen points, so do a running sum over the time dimension
    strokes[:,:2] = np.cumsum(strokes[:,:2], axis=0)
    char_to_plot[:,:2] = np.cumsum(char_to_plot[:,:2], axis=0)
    return strokes, char_to_plot, phis, windows, kappas

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

# plots the stroke data (handwriting!)
def line_plot_coef(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

def line_plot_char(strokes, char_to_plot, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(char_to_plot[start:stop,0], char_to_plot[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

