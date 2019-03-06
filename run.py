#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

import argparse
import time
import os

from model import *
from importlib import reload
from dataloader import DataProcess
from sample import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# In[2]:


def init_args():
    
        args = {}
        args['rnn_size'] = 100 
        args['tsteps'] = 300 
        args['batch_size'] = 32 
        args['num_batches'] = 500 
        args['num_mixtures'] = 20 # number of MDN mixtures
        args['window_mixtures'] = 10 # number of attention window mixtures
        args['learning_rate'] = 0.001
        args['epochs'] = 2500 
        args['alphabet'] = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        args['tsteps_per_char'] = 25
        
        args['biases'] = 1.0
        args['data_dir'] = "./data"
        args['logs_dir'] = './logs/'
        args['save_path'] = 'model9/model.ckpt' # path to save the model at
        args['load_path'] = 'model8/model.ckpt' # path to load the model from
        args['grad_clip'] = 10
        args['n_to_save'] = 500 #step difference at which to save the model
        args['scale_factor'] = 20
        args['gap'] = 500 #remove data with gap greater than this threshhold
        args['learning_rate_decay'] = 0.99 
        args['keep_prob'] = 0.85 # keep_prob for dropout
        args['train'] = False
        args['decay'] = 0.95
        args['momentum'] = 0.9
        return args

def load_pretrained_model(model, path):
        global_step = 0
        try:
            save_dir = '/'.join(path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            model.saver.restore(model.sess, load_path)
            #load_was_success = True
        except Exception as e:
            print(e)
            load_was_success = False
        else:
            model.saver = tf.train.Saver(tf.global_variables())
            global_step = int(load_path.split('-')[-1])
            load_was_success = True
        return load_was_success, global_step
    
def train_model():
    args = init_args()
    args['train'] = True
    data_loader = DataProcess(args)
    
    # num_batches is calculated in dataloader based on total data size and batch_size
    args['num_batches'] = data_loader.num_batches
    print("num_bacthes", args['num_batches'])
    
    model = Model(args)
    load_was_success, global_step = load_pretrained_model(model, args['save_path'])
    
    # list to hold the loss values in each itration of the mini batch
    plot_loss = []
    model.sess.run(tf.assign(model.decay, args['decay']))
    model.sess.run(tf.assign(model.momentum, args['momentum']))
    
    # Mini batch for given number of epochs
    for e in range(int(global_step/args['num_batches']), args['epochs']):
        print("Running epoch", e)
        
        # learning rate decay
        model.sess.run(tf.assign(model.learning_rate, args['learning_rate'] * (args['learning_rate_decay'] ** e)))
        
       
        # initializes data pointer to starting of batch in each epoch
        data_loader.init_batch_comp()
        c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
        h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
        kappa = np.zeros((args['batch_size'], args['window_mixtures'], 1))

        for b in range(global_step%args['num_batches'], args['num_batches']):

            i = e * args['num_batches'] + b
            if global_step is not 0 : i+=1 ; global_step = 0

            if i % args['n_to_save'] == 0 and (i > 0):
                # save the model we have right now
                model.saver.save(model.sess, args['save_path'], global_step = i) ;
            
            # get next batch of data to train on
            x, y, asciis, asciis_oh = data_loader.get_next_batch()
            
            feed = {model.input: x, model.output: y, model.char_seq: asciis_oh, model.kappa_start: kappa,                     model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2,                     model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
            [train_loss, _] = model.sess.run([model.cost, model.train_op], feed)

            plot_loss.append(train_loss)
        print("train_loss: " + str(i))
        print(train_loss)
    
    # plotting the loss graph
    plt.plot(plot_loss, linewidth=2.0)
    plt.savefig("./loss.png")

# Function to sample some handwriting, this does not try to sample any particular style YET
def sample_model():
    args = init_args()
    args['tsteps'] = 1
    args['batch_size'] = 1

    model = Model(args)
    
    # load a pretrained model
    load_was_success, global_step = load_pretrained_model(model, args['load_path'])
    if load_was_success:
            strokes, char_to_plot, phis, windows, kappas = sample(model, args)
            line_plot_coef(strokes, 'Line plot', figsize = (20,4), save_path="./coef_plot7.png")
            line_plot_char(strokes, char_to_plot, 'Line plot', figsize = (20,4), save_path="./line_char7.png")
            print("plotted")

    else:
        print("Model failed to load, can't sample")


            
            
            
            
            
            
            
            
            
    


# In[3]:


sample_model()


# In[ ]:




