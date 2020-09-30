"""
helper functions for data handling
Timo Flesch, 2017

"""
import numpy as np

def genData(trialIDs,nreps=10):
    pass

def genBatch(data,batch_size=250):
    pass

def normalizeData(data_train,data_test):
    """
    helper function to mean-center images along each colour channel
    """
    tmp_train = data_train.reshape(data_train.shape[0],128*128,3).astype('float32') / 255.
    tmp_test = data_test.reshape(data_test.shape[0],128*128,3).astype('float32')    / 255.

    for chan in range(3):
        tmp2 = np.mean(tmp_train[:,:,chan])
        tmp_train[:,:,chan] -= tmp2
        tmp3 = np.std(tmp_train[:,:,chan])
        tmp_train[:,:,chan] /= tmp3
        tmp_test[:,:,chan] -= tmp2
        tmp_test[:,:,chan] /= tmp3
    print('min and max of normalized images')
    print(np.min(tmp_train[0,:]))
    print(np.max(tmp_train[0,:]))
    return tmp_train.reshape(data_train.shape[0],128*128*3),tmp_test.reshape(data_test.shape[0],128*128*3)



def shuffleData(x,y=None):
    """ helper function, shuffles data """
    ii_shuff = np.random.permutation(x.shape[0])
    # shuffle data
    x = x[ii_shuff,:]
    if y is not None:
        y = y[ii_shuff,:]
    return x, y
