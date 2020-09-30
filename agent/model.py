# implements an LSTM to solve the DMC task
#
# Timo Flesch, 2019,
# Human Information Processing Lab,
# Experimental Psychology Department,
# University of Oxford

import os
import numpy as np
import tensorflow as tf

from nntools.io import saveMyModel
from nntools.layers import *
from nntools.external import *

FLAGS = tf.app.flags.FLAGS


class NNet(object):
    """
    model class, implements an LSTM to be trained on the DMC task
    """

    def __init__(self,
        dim_inputs=[None,45,48,48], # batch size, sequence length, dim x, dim y
        dim_outputs=[None,1], # single integer as output (class probability)
        n_hidden=200,
        lr=0.001,
        optimizer='Adam',
        nonlinearity=None,
        is_trained=False):
        """ initializes the model with parameters """

        self.ITER = 0
        self.session = None
        self.learning_rate = lr

        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.n_hidden = n_hidden

        self.nonlinearity = getattr(tf.nn, nonlinearity)
        self.initializer = tf.initializers.variance_scaling(scale=2.0,
            mode='fan_in',
            distribution='truncated_normal')
        #tf.truncated_normal_initializer(FLAGS.weight_init_mu, FLAGS.weight_init_std)

        # dictionary for all parameters (weights + biases)
        self.params = {}

        self.init_done = False

        if not(is_trained):
            with tf.name_scope('placeholders'):
                # input placeholder expects flattened images
                self.x = tf.compat.v1.placeholder(tf.float32, [None,self.dim_inputs[1],
                    self.dim_inputs[2]*self.dim_inputs[3]],
                    name='x_flat')
                # output placeholder expects a single integer
                self.y_true = tf.compat.v1.placeholder(tf.float32, [None,
                    self.dim_outputs[1]], name='y_true')

            # the neural network and label placeholder
            with tf.name_scope('lstm'):
                self.nnet_builder()

            # optimizer
            with tf.name_scope('optimisation'):
                self.optimizer_builder(optimizer)

        else:
            self.init_done = True

    def nnet_builder(self):
        """ creates the actual neural network """


        with tf.name_scope('fc_enc'):
            # self.fc_enc, self.params['fc_enc_weights'], self.params['fc_enc_biases'] = layer_fc(self.x, 128,nonlinearity=tf.keras.activations.tanh,name='fc_enc')
            self.fc_enc = tf.layers.dense(self.x,units=48*48)


        # recurrent
        with tf.name_scope('recurrent_unit'):
            self.lstm = layer_lstm(self.fc_enc,n_hidden=self.n_hidden)



        # fc, only on most recent prediction
        with tf.name_scope('fc_dec'):
            self.y_hat, self.params['fc_dec_weights'], self.params['fc_dec_biases'] = layer_fc(self.lstm[:,-1,:], self.dim_outputs[1],nonlinearity=tf.keras.activations.sigmoid,name='fc_dec')

        return

    def optimizer_builder(self, optimizer):
        """
        creates optimiser
        """
         # loss function
        with tf.name_scope('loss-function'):
            bce = tf.keras.losses.BinaryCrossentropy()
            self.loss = bce(self.y_true,self.y_hat)
            self.pred = tf.cast(self.y_hat > .5,tf.float32)

            self.acc =tf.reduce_mean(tf.squeeze(tf.cast(self.y_true==self.pred,tf.int32)))
        # optimisation procedure
        with tf.name_scope('optimizer'):
            self.optimizer = getattr(tf.compat.v1.train,optimizer+'Optimizer')(learning_rate=self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)
        return

    def init_graph_vars(self, sess, log_dir='.'):
        """ initializes graph variables """
        # set session
        self.session = sess
        # initialize all variables
        self.init_op = tf.compat.v1.global_variables_initializer()
        self.session.run(self.init_op)

        # define saver object AFTER variable initialisation
        self.saver = tf.compat.v1.train.Saver()

        # define summaries
        self.summaryLoss = tf.compat.v1.summary.scalar("loss", tf.reduce_mean(self.loss))
        self.summaryTraining = tf.compat.v1.summary.scalar("loss_training", self.loss)
        self.summaryCorrect = tf.compat.v1.summary.scalar("acc",self.acc)
        # self.summaryPred = tf.compat.v1.summary.scalar("pred",tf.squeeze(self.pred[-1]))
        # self.summaryTrue = tf.compat.v1.summary.scalar("label",tf.squeeze(self.y_true[-1]))
        self.summaryTest = tf.compat.v1.summary.scalar("loss_test", self.loss)

        self.merged_summary = tf.compat.v1.summary.merge_all()

        self.writer = tf.compat.v1.summary.FileWriter(log_dir, self.session.graph)

        self.init_done = True

    def save_ckpt(self, modelDir):
        """ saves model checkpoint """
        # save the whole statistical model
        saved_ops = [('nnet', self.y_hat),('input', self.x)]

        saveMyModel(self.session,
                    self.saver,
                    saved_ops,
                    globalStep=self.ITER,
                    modelName=modelDir+FLAGS.model)

    def inference(self, x):
        """ forward pass of x through myModel """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        y_hat = self.session.run(self.y_hat, feed_dict={self.x: x})
        # print(np.asarray(y_hat).shape)
        return y_hat

    def train(self, x, y_true):
        """ training step """
        _,loss,summary_train,summary_loss,summary_acc, = self.session.run([self.train_step,
            self.loss, self.summaryTraining,self.summaryLoss,self.summaryCorrect],
            feed_dict={self.x: x,self.y_true: y_true})
        self.writer.add_summary(summary_train, self.ITER)
        self.writer.add_summary(summary_loss, self.ITER)
        self.writer.add_summary(summary_acc, self.ITER)
        # self.writer.add_summary(summary_pred, self.ITER)
        # self.writer.add_summary(summary_true, self.ITER)
        self.ITER += 1 # count one iteration up
        return loss

    def eval(self, x, y_true):
        """ evaluation step """
        y_hat,loss, summary_test = self.session.run([self.y_hat,
            self.loss, self.summaryTest],
            feed_dict={self.x: x, self.y_true: y_true})
        self.writer.add_summary(summary_test, self.ITER)
        return loss
