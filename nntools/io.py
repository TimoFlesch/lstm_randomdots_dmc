# file io
# Timo Flesch, 2017
import tensorflow as tf
import scipy.io as sio
import pickle

def save_data(data, fileName, dataDir):
	with open(dataDir+fileName,'wb') as f:
		pickle.dump(data,f)


def load_data(fileName, dataDir):
	"""
	loads trees data from hard disk
	"""
	with open(dataDir+fileName,'rb') as f:
		data = pickle.load(f)
		return data

def save_log(log_dict, modelName, logDir):
	"""
    saves log on harddisk
    """
	fileName = logDir + modelName
	sio.savemat(fileName, log_dict)



def loadMyModel(sess, op_ids, ckpt_dir):
	ckpt = tf.compat.v1.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
		saver.restore(sess, ckpt.model_checkpoint_path)

	saved_ops = []
	for ii, op in enumerate(op_ids):
		saved_ops.append(tf.compat.v1.get_collection(op)[0])
	print("succesfully retrieved model checkpoint")
	return saved_ops


def saveMyModel(sess,saver,ops,globalStep=1,modelName='./model_ckpt'):
	""" saves selected model ops.
		ops are tuples ('name',op)
    """
	for op in ops:
		tf.add_to_collection(op[0],op[1])

	saver.save(sess,modelName,global_step=globalStep)
	print("succesfully saved model checkpoint")
	return True
