import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy      as np
import tensorflow as tf
from datetime import datetime
# from nntools.visualisation import *

# np.random.seed(0)
# tf.compat.v1.set_random_seed(0)

FLAGS = tf.app.flags.FLAGS


def trainModel(sess,nnet,x_train,x_test,n_episodes=100,n_batches=10,batch_size=128,model_dir='./'):
    sess = nnet.session;

    # declare some log vars
    results = {}
    loss_total_train     = []
    loss_total_test      = []


    # for memory reasons, split test set into minibatches
    minibatches_xTest = [x_test[k:k+batch_size] for k in range(0,x_test.shape[0],batch_size)]
    # extract some sample images (4 times full grid)
    x_sample = x_test[0:100,:]

    # --- main training loop ---

    for ep in range(n_episodes):
        # save model
        nnet.save_ckpt(model_dir + '/')

        # shuffle training data
        x_train = shuffleData(x_train)
        nnet.reconstruct(x_train[0,:])
        # minibatches_xTrain = [x_train[k:k+batch_size] for k in range(0,x_train.shape[0],batch_size)]
        # iterate through several batches and train!
        loss_train = 0
        # for mb_train in minibatches_xTrain:
        nnet.inference(x_test[0,:])
        for ii in range(FLAGS.n_training_batches):
            # perform training step
            # tmp, _ = mnist.train.next_batch(batch_size)
            tmp = getRandomSubSet(x_train,batch_size)
            loss = nnet.train(tmp,tmp)
            loss_train += loss / FLAGS.n_samples_train * FLAGS.batch_size


        loss_total_train.append(loss_train)


        # interim evaluation
        if ((ep%FLAGS.display_step)==0):
            # save model
            nnet.save_ckpt(model_dir + '/')
            # iterate through batches and average test loss
            loss_test = 0

            for jj in range(FLAGS.n_test_batches):
                tmp = getRandomSubSet(x_test,batch_size)
                loss = nnet.eval(tmp,tmp)
                loss_test += loss / FLAGS.n_samples_test * FLAGS.batch_size
            loss_total_test.append(loss_test)


            # log results to stdout
            epTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Episode {} Training Loss {:.5f} Test Loss {:.5f} \n'.format(epTime, int(ep)+1, loss_train, loss_test))

    # write results to result dictionary and return
    results = { 'loss_train'     : loss_total_train,
                'loss_test'      : loss_total_test,
                 }
    return results

def getRandomSubSet(x,batch_size=128):
    """ small helper function to retrieve a sample batch """

    sampleStartIDX = np.random.randint(0,len(x)-batch_size)
    return x[sampleStartIDX:(sampleStartIDX+batch_size),:]


def shuffleData(x):
    """ helper function, shuffles data """
    ii_shuff = np.random.permutation(x.shape[0])
    # shuffle data
    x = x[ii_shuff,:]
    return x
