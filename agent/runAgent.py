"""
runner function for convolutional variational autoencoder
Timo Flesch, 2017
update 2019: naming convention, r1.14 compatibility
"""
# external
import numpy as np
import tensorflow as tf

# custom
from agent.model import NNet
from nntools.trainer import trainModel
# from nntools.evaluation import evalModel
from nntools.io import loadMyModel
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

def runAgent(x_train, x_test):
    # checkpoint run model folder
    ckpt_dir_run = FLAGS.ckpt_dir + 'model_' + FLAGS.model
    log_dir_run = FLAGS.logging_dir + 'model_' + FLAGS.model

    if not(tf.gfile.Exists(ckpt_dir_run)):
        tf.gfile.MakeDirs(ckpt_dir_run)

    if not(tf.gfile.Exists(log_dir_run)):
        tf.gfile.MakeDirs(log_dir_run)

    with tf.Session() as sess:

        if FLAGS.do_training:
            # create new model instance
            nnet = NNet(lr=FLAGS.learning_rate,
                           optimizer=FLAGS.optimizer,
                           nonlinearity=FLAGS.nonlinearity,
                           n_hidden=FLAGS.n_hidden,
                           n_latent=FLAGS.n_latent,
                           beta=FLAGS.beta,
                           )
            # initialize all variables
            nnet.init_graph_vars(sess, log_dir=log_dir_run)

            # tell researcher what's going on ...
            print("{} Now training my model, LR: {} , EPs: {}, BS: {}"
                .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    FLAGS.learning_rate, FLAGS.n_training_episodes,
                    FLAGS.batch_size))
            # .. and begin with training
            results = trainModel(sess, nnet, x_train, x_test,
                n_episodes=FLAGS.n_training_episodes,
                n_batches=FLAGS.n_training_batches,
                batch_size=FLAGS.batch_size,
                model_dir=ckpt_dir_run)
            evalModel(sess, nnet,x_train)

        else:
            nnet = myModel(is_trained=True)
            print("Now evaluating my model")
            ops = loadMyModel(sess, ['nnet','input'], ckpt_dir_run)
            print(ops)
            nnet.session = sess
            nnet.y_hat = ops[0]
            nnet.x = ops[1]

            results = evalModel(sess, nnet, x_train)

        return results
