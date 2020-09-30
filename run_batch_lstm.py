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
from nntools.io import *
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
tf.app.flags.DEFINE_integer('n_sets_train', 200,
                           """ (int) number of training samples """) # was 200

tf.app.flags.DEFINE_integer('n_sets_test',  20,
                           """ (int) number of test samples     """) # was 20
tf.app.flags.DEFINE_integer('imsize',             48,
                            """ image width """)
tf.app.flags.DEFINE_integer('bound_idx',             1,
                            """ which boundary """)

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

tf.app.flags.DEFINE_integer('n_hidden',             512,
                            """ dimensionality of hidden layers """)



# training
tf.app.flags.DEFINE_float('learning_rate',     1e-4,
                            """ (float)   learning rate               """)

tf.app.flags.DEFINE_integer('n_training_episodes',   15,
                            """ (int)    number of training episodes  """)


tf.app.flags.DEFINE_integer('display_step',         1,
                            """(int) episodes until training log      """)

tf.app.flags.DEFINE_integer('batch_size',         64,
                            """ (int)     training batch size         """)

tf.app.flags.DEFINE_integer('run_id',   1,
                            """ (int)    which run  """)


tf.app.flags.DEFINE_string('optimizer',       'RMSProp',
                            """ (string)   optimisation procedure     """)

tf.app.flags.DEFINE_integer('n_training_batches',
                            int(FLAGS.n_sets_train*64/FLAGS.batch_size),
                            """    number of training batches per ep  """)

tf.app.flags.DEFINE_integer('n_test_batches',
                            int(FLAGS.n_sets_test*64/FLAGS.batch_size),
                            """    number of test batches per ep  """)



def main(argv=None):
    """ here starts the magic """


    FLAGS = tf.app.flags.FLAGS
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.compat.v1.Session(config=config) as sess:
        # with tf.device('/GPU:0'):
        nnet = NNet(lr=FLAGS.learning_rate,
                   optimizer=FLAGS.optimizer,
                   nonlinearity=FLAGS.nonlinearity,
                   n_hidden=FLAGS.n_hidden,
                   dim_inputs=[None,45,FLAGS.imsize,FLAGS.imsize])
                   # # initialize all variables
        nnet.init_graph_vars(sess, log_dir=FLAGS.logging_dir)

        # instantiate task
        DOT_CATEGORIES = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0]])
        DOT_BOUNDDISTS = np.asarray([[1, 2, 2, 1, 1, 2, 2, 1], [1, 1, 2, 2, 1, 1, 2, 2], [2, 1, 1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2, 1, 1]])
        DOT_BOUNDANGLES = np.asarray([180, 45, 90, 135])
        DOT_BOUNDIDX = FLAGS.bound_idx
        DOT_CATLABELS = DOT_CATEGORIES[DOT_BOUNDIDX-1, :]
        DOT_BOUNDARY = DOT_BOUNDANGLES[DOT_BOUNDIDX-1]
        dmc = DMCTrial(catbound=DOT_BOUNDARY,catlabels=DOT_CATLABELS)
        trials = dmc.set_trialids()



        # create large training set
        data_train = np.empty((FLAGS.n_sets_train*trials.shape[0],45,FLAGS.imsize,FLAGS.imsize))
        labels_train = np.empty((FLAGS.n_sets_train*trials.shape[0],trials.shape[1]))
        idx = 0
        for ii in range(FLAGS.n_sets_train):
            # shuffle the trial vector
            np.random.shuffle(trials)
            # generate trials
            for jj in range(trials.shape[0]):
                # run a trial
                this_trial = dmc.run(trials[jj, 0], trials[jj, 1])
                data_train[idx,:,:,:] = this_trial
                labels_train[idx,:] = trials[jj,:]
                idx += 1

            if ii % 10 == 0:
                print('generated training batch ' + str(ii+1) +  '/' + str(FLAGS.n_sets_train))

        # create test set
        data_test = np.empty((FLAGS.n_sets_test*trials.shape[0],45,FLAGS.imsize,FLAGS.imsize))
        labels_test = np.empty((FLAGS.n_sets_test*trials.shape[0],trials.shape[1]))
        idx = 0
        for ii in range(FLAGS.n_sets_test):
            # shuffle the trial vector
            np.random.shuffle(trials)
            # generate trials
            for jj in range(trials.shape[0]):
                # run a trial
                this_trial = dmc.run(trials[jj, 0], trials[jj, 1])
                data_test[idx,:,:,:] = this_trial
                labels_test[idx,:] = trials[jj,:]
                idx += 1

            if ii % 10 == 0:
                ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(ep_time + ' generated test batch ' + str(ii+1) + '/' + str(FLAGS.n_sets_test))

        # train the LSTM
        print(' now training the network')
        n_runs = 1
        # set up datastructures for logging
        results = dict()
        results['acc_train'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_training_batches))
        results['loss_train'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_training_batches))
        results['acc_test'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_test_batches))
        results['loss_test'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_test_batches))
        results['lstm_test'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_test_batches, FLAGS.batch_size, 45, FLAGS.n_hidden))
        results['pred_test'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_test_batches, FLAGS.batch_size))
        results['trials_test'] = np.empty((FLAGS.n_training_episodes+1, FLAGS.n_test_batches, FLAGS.batch_size, trials.shape[1]))

        idx_batch_train = np.arange(0,data_train.shape[0],FLAGS.batch_size).astype('int')

        for bb in range(FLAGS.n_training_batches):
            x_train = np.reshape(data_train[idx_batch_train[bb]:idx_batch_train[bb]+FLAGS.batch_size,:,:,:],[FLAGS.batch_size,45,FLAGS.imsize**2])
            y_train = labels_train[idx_batch_train[bb]:idx_batch_train[bb]+FLAGS.batch_size,-1]
            y_train = y_train[:,np.newaxis]
            y_batch,loss_batch = sess.run([nnet.pred,nnet.loss],feed_dict={nnet.x:x_train,nnet.y_true:y_train});
            results['acc_train'][0,bb] = np.mean(y_batch==y_train)
            results['loss_train'][0,bb] = loss_batch


        idx_batch_test = np.floor(np.arange(0,data_test.shape[0],FLAGS.batch_size)).astype('int')
        for bb in range(FLAGS.n_test_batches):
            x_test = np.reshape(data_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,:,:,:],[FLAGS.batch_size,45,FLAGS.imsize**2])
            y_test = labels_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,-1]
            y_test = y_test[:,np.newaxis]
            y_batch, loss_batch, lstm_batch,yhat_batch = sess.run([nnet.pred,nnet.loss,nnet.lstm,nnet.y_hat],feed_dict={nnet.x:x_test,nnet.y_true:y_test});
            results['acc_test'][0,bb] = np.mean(y_batch==y_test)
            results['loss_test'][0,bb] = loss_batch
            results['lstm_test'][0,bb,:,:,:] = lstm_batch
            results['pred_test'][0,bb,:] = np.squeeze(yhat_batch)
            results['trials_test'][0,bb,:,:] = labels_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,:]

        ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print(np.mean(results['acc_train'][0,:],1))
        # print(np.mean(results['acc_test'][0,:],1))
        # print(np.mean(results['loss_train'][0,:],1))
        # print(np.mean(results['loss_test'][0,:],1))
        print('{} episode {},  train accuracy  {:.2f}, test accuracy {:.2f} --- train loss {:.5f},  test loss {:.5f} '.format(ep_time, 0,np.mean(results['acc_train'][0,:]),np.mean(results['acc_test'][0,:]),np.mean(results['loss_train'][0,:]),np.mean(results['loss_test'][0,:])))

        for ep in range(FLAGS.n_training_episodes):
            # train on random batches

            for bb in range(FLAGS.n_training_batches):
                ii = np.random.permutation(FLAGS.n_sets_train*trials.shape[0])
                ii = ii[0:FLAGS.batch_size]
                x_train = np.reshape(data_train[ii,:,:,:],[FLAGS.batch_size,45,FLAGS.imsize**2])
                y_train = labels_train[ii,-1]
                y_train = y_train[:,np.newaxis]
                nnet.train(x_train,y_train)
            # evaluate training performance on whole dataset
            for bb in range(FLAGS.n_training_batches):
                x_train = np.reshape(data_train[idx_batch_train[bb]:idx_batch_train[bb]+FLAGS.batch_size,:,:,:],[FLAGS.batch_size,45,FLAGS.imsize**2])
                y_train = labels_train[idx_batch_train[bb]:idx_batch_train[bb]+FLAGS.batch_size,-1]
                y_train = y_train[:,np.newaxis]
                y_batch, loss_batch = sess.run([nnet.pred,nnet.loss],feed_dict={nnet.x:x_train,nnet.y_true:y_train});
                results['acc_train'][ep+1,bb] = np.mean(y_batch==y_train)
                results['loss_train'][ep+1,bb] = loss_batch

            # evaluate test performance on whole dataset
            for bb in range(FLAGS.n_test_batches):
                x_test = np.reshape(data_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,:,:,:],[FLAGS.batch_size,45,FLAGS.imsize**2])
                y_test = labels_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,-1]
                y_test = y_test[:,np.newaxis]
                y_batch,loss_batch,lstm_batch, yhat_batch = sess.run([nnet.pred,nnet.loss, nnet.lstm, nnet.y_hat],feed_dict={nnet.x:x_test,nnet.y_true:y_test});
                results['acc_test'][ep+1,bb] = np.mean(y_batch==y_test)
                results['loss_test'][ep+1,bb] = loss_batch
                results['lstm_test'][ep+1,bb,:,:,:] = lstm_batch
                results['pred_test'][ep+1,bb,:] = np.squeeze(yhat_batch)
                results['trials_test'][ep+1,bb,:,:] = labels_test[idx_batch_test[bb]:idx_batch_test[bb]+FLAGS.batch_size,:]
            ep_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} episode {},  train accuracy  {:.2f}, test accuracy {:.2f} --- train loss {:.5f},  test loss {:.5f} '.format(ep_time, ep+1,np.mean(results['acc_train'][ep+1,:]),np.mean(results['acc_test'][ep+1,:]),np.mean(results['loss_train'][ep+1,:]),np.mean(results['loss_test'][ep+1,:])))
    fn = 'results_run_' + str(FLAGS.run_id) + '_bound_' + str(FLAGS.bound_idx)
    # save_data(results, fn, FLAGS.data_dir)
    save_log(results, fn + '.mat', FLAGS.data_dir)


if __name__ == '__main__':
    """ take care of flags on load """
    tf.compat.v1.app.run()
