"""
wrapper function for frequently-used tensorflow layers
Timo Flesch, 2017
"""
import tensorflow as tf

def layer_dropout(x,
                keep_prob=1.0,
                name='dropout'):
    """
    dropout layer
    """
    with tf.compat.v1.variable_scope(name):
        y = tf.nn.dropout(x, keep_prob)
        return y


def layer_flatten(x,
                name = 'flatten'):
    """
    flattens the output of a conv/maxpool layer
    """
    with tf.compat.v1.variable_scope(name):
        shape_in = x.get_shape().as_list()
        y = tf.reshape(x,[-1, shape_in[1]*shape_in[2]*shape_in[3]])

        return y


def layer_fc(x,
			dim_y,
			bias_const=0.0,
			name='linear',
			initializer=tf.initializers.glorot_normal(),
            nonlinearity=None):
	"""
	simple fully connected layer with optional nonlinearity
	"""

	with tf.compat.v1.variable_scope(name):
		weights = tf.compat.v1.get_variable('weights',[x.get_shape().as_list()[-1],dim_y],tf.float32, initializer=initializer)
		biases  = tf.compat.v1.get_variable('biases',[dim_y],initializer=tf.constant_initializer(bias_const))
		y = tf.nn.bias_add(tf.matmul(x,weights),biases)
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases


def layer_lstm(x, name='lstm_layer',n_hidden=32):
    """
    wrapper for an lstm layer
    """
    with tf.compat.v1.variable_scope(name):
        cell = tf.keras.layers.LSTMCell(n_hidden)
        # layer = tf.keras.layers.RNN(cell)
        # y = layer(x)
        y,_ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        return y

def layer_conv2d(x,
				n_filters   =       32,
				filter_size =    (6,6),
				stride      =    (1,1),
				padding     =  'VALID',
				name        = 'conv2d',
				bias_const  =      0.0,
				initializer = tf.initializers.glorot_normal(),
				nonlinearity =   None):
	"""
	wrapper for convolution
	"""
	with tf.compat.v1.variable_scope(name):

		kernel = [filter_size[0],filter_size[1],x.get_shape()[-1],n_filters]
		stride = [1,stride[0],stride[1],1]
		shape = x.get_shape().as_list()

		weights = tf.compat.v1.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.compat.v1.get_variable('biases',[n_filters],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d(x,weights,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')

		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases, shape


def layer_max_pool_2x2(x,
                    filter_size = (2,2),
                    stride = (2,2),
                    padding='SAME',
                    name='maxpool_2x2'):

    """
    wrapper for max pool operation
    """
    with tf.compat.v1.variable_scope(name):
        kernel = [1,filter_size[0],filter_size[1],1]
        stride = [1,stride[0],stride[1],1]
        shape = x.get_shape().as_list()

        y = tf.nn.max_pool(x, kernel, stride, padding)
        return y,shape


def layer_transpose_conv2d(x,
				n_filters   =       32,
				filter_size =    (6,6),
				stride      =    (1,1),
				shape       = [1,2,2,2],
				padding     =  'VALID',
				name        = 'transconv2d',
				bias_const  =      0.0,
				initializer = tf.initializers.glorot_normal(),
				nonlinearity =   None):
	"""
	wrapper for transposed convolution
	"""

	with tf.compat.v1.variable_scope(name):
		kernel = [filter_size[0],filter_size[1],shape[3],n_filters]
		stride = [1,stride[0],stride[1],1]
		out_dims = tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]])
		weights = tf.compat.v1.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.compat.v1.get_variable('biases',[shape[3]],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d_transpose(x,weights,out_dims,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		shape = x.get_shape().as_list()
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases,shape
