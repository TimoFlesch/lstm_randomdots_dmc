"""
Code skeleton for tensorflow projects

Timo Flesch, 2017
update 2019: naming convention, r1.14 compatibility
"""

# external
import rdktools.rdk_params as params
from rdktools.rdk_experiment import set_trials, DMCTrial
import tensorflow as tf
import pygame
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# custom
from nntools.data import *
from agent.runAgent import runAgent
from nntools.io import save_log, load_data, loadMyModel

#debug
from agent.model import NNet


# -- define flags --
FLAGS = tf.app.flags.FLAGS

# directories
tf.app.flags.DEFINE_string('data_dir',  './data/',
                           """ (string) data directory           """)

tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints/',
                            """ (string) checkpoint directory    """)
# log_dir is already defined by abseil, as of r1.14 (fucking hell...)
tf.app.flags.DEFINE_string('logging_dir',          './log/',
                           """ (string) log/summary directory    """)


# dataset
tf.app.flags.DEFINE_integer('n_samples_train', 100000,
                           """ (int) number of training samples """)

tf.app.flags.DEFINE_integer('n_samples_test',  50000,
                           """ (int) number of test samples     """)


# model
tf.app.flags.DEFINE_string('model',                'mymodel',
                            """ (string)  chosen model          """)

tf.app.flags.DEFINE_bool('do_training',               1,
                            """ (boolean) train or not          """)

tf.app.flags.DEFINE_float('weight_init_mu',         0.0,
                            """ (float)   initial weight mean   """)

tf.app.flags.DEFINE_float('weight_init_std',        .1,
                            """ (float)   initial weight std    """)

tf.app.flags.DEFINE_string('nonlinearity',       'relu',
                            """ (string)  activation function   """)

tf.app.flags.DEFINE_integer('n_hidden',             64,
                            """ dimensionality of hidden layers """)



# training
tf.app.flags.DEFINE_float('learning_rate',     1e-4,
                            """ (float)   learning rate               """)

tf.app.flags.DEFINE_integer('n_training_episodes',   10,
                            """ (int)    number of training episodes  """)


tf.app.flags.DEFINE_integer('display_step',         1,
                            """(int) episodes until training log      """)

tf.app.flags.DEFINE_integer('batch_size',         128,
                            """ (int)     training batch size         """)

tf.app.flags.DEFINE_string('optimizer',       'RMSProp',
                            """ (string)   optimisation procedure     """)

tf.app.flags.DEFINE_integer('n_training_batches',
                            int(FLAGS.n_samples_train/FLAGS.batch_size),
                            """    number of training batches per ep  """)

tf.app.flags.DEFINE_integer('n_test_batches',
                            int(FLAGS.n_samples_test/FLAGS.batch_size),
                            """    number of test batches per ep  """)





def main(argv=None):
    """ here starts the magic """

    # debugging
    # print(tf.flags.FLAGS.__flags.items())
    # print(tf.app.flags.FLAGS.flag_values_dict())
    FLAGS = tf.app.flags.FLAGS

    sess = tf.compat.v1.Session()
    # with tf.device('/GPU:0'):
    nnet = NNet(lr=FLAGS.learning_rate,
               optimizer=FLAGS.optimizer,
               nonlinearity=FLAGS.nonlinearity,
               n_hidden=FLAGS.n_hidden)
               # # initialize all variables
    nnet.init_graph_vars(sess, log_dir=FLAGS.logging_dir)

    # instantiate task
    dmc = DMCTrial()

    # instantiate neural netowrk
    # nnet = Agent()
    trials = dmc.set_trialids()
    #
    # n_iter = 0
    # while n_iter <50000:
    #         # sample orientations and category label
    #         # TODO: have vect of all combinations and label (n*3)
    #         # angle_sample, angle_probe, cat_label = dmc.get_io()
    #         # run trial
    #         ii = np.random.randint(trials.shape[0])
    #         frames = dmc.run(trials[ii,0],trials[ii,1])
    #         frames = np.expand_dims(np.reshape(frames,(params.SEQ_LENGTH,params.WINDOW_WIDTH**2)),0)
    #         y_true = trials[ii,-1][np.newaxis, np.newaxis]
    #         nnet.train(frames,y_true)
    #         if n_iter%1000==0:
    #                 ckpt_dir = FLAGS.ckpt_dir + 'iter' + str(n_iter) + '/'
    #                 if not(tf.gfile.Exists(ckpt_dir)):
    #                     tf.gfile.MakeDirs(ckpt_dir)
    #                 nnet.save_ckpt(ckpt_dir)
    #         n_iter +=1
    #
    # nnet.save_ckpt(FLAGS.ckpt_dir)


    # new idea: fixed training data (try to overfit)
    data_train = [[dmc.run(trials[ii,0],trials[ii,1])] for ii in range(trials.shape[0])]
    labels_train = trials[:,-1]
    n_iter = 0
    while n_iter < 50000:
        ii_shuff = np.random.permutation(np.arange(trials.shape[0]))
        for ii in range(len(data_train)):
            frames = np.expand_dims(np.reshape(data_train[ii_shuff[ii]],(params.SEQ_LENGTH,params.WINDOW_WIDTH**2)),0)
            y_true = labels_train[ii_shuff[ii]][np.newaxis, np.newaxis]
            nnet.train(frames,y_true)
            if n_iter%1000==0:
                print(n_iter)
                ckpt_dir = FLAGS.ckpt_dir + 'iter' + str(n_iter) + '/'
                if not(tf.gfile.Exists(ckpt_dir)):
                    tf.gfile.MakeDirs(ckpt_dir)
                nnet.save_ckpt(ckpt_dir)
            n_iter +=1

    # now evaluate:
    data_test = [[dmc.run(trials[ii,0],trials[ii,1])] for ii in range(trials.shape[0])]
    labels_test = trials[:,-1]

    ckpt_dir = FLAGS.ckpt_dir + 'iter' + str(48000) + '/'
    ops = loadMyModel(sess, ['nnet','input'], ckpt_dir)
#
    nnet.session = sess
    nnet.y_hat = ops[0]
    nnet.x = ops[1]
    y_hat = np.zeros(len(data_test))
    for jj in range(len(data_test)):
        frames = np.expand_dims(np.reshape(data_test[jj],(params.SEQ_LENGTH,params.WINDOW_WIDTH**2)),0)
        y_hat[jj] = nnet.inference(frames)
        # print(str(labels_test[jj]) + '  ' + str(y_hat[jj]))
    acc = np.mean((y_hat>0.5)==labels_test)
    print(acc)




if __name__ == '__main__':
    """ take care of flags on load """
    tf.compat.v1.app.run()
