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
from tensorflow.keras import layers
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

tf.app.flags.DEFINE_string('optimizer',       'Adam',
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

    model = tf.keras.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(256,activation='tanh'),input_shape=(params.SEQ_LENGTH,96**2)))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1,activation='sigmoid'))
    # model.build()
    model.summary()

if __name__ == '__main__':
    """ take care of flags on load """
    tf.compat.v1.app.run()
    main()
