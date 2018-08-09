import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l2_regularizer as l2_reg


def conv2d(input, kernel_size, fliters_in, fliters_out, n_layer, stride=1, pad='SAME', wd=5e-7, init=None):
    wshape = [kernel_size, kernel_size, fliters_in, fliters_out]
    wconv_name = 'weights' + str(n_layer)
    conv_name = 'conv' + str(n_layer)
    if init:
        wconv = tf.get_variable(name=wconv_name, shape=wshape, dtype=tf.float32, initializer=init)
    else:
        wconv = tf.Variable(tf.random_normal(shape=wshape, mean=0, stddev=1e-3), dtype=tf.float32, name=wconv_name)
    wconv = wconv - tf.reduce_mean(wconv)
    tf.add_to_collection('losses', l2_reg(wd)(wconv))
    biases = tf.Variable(tf.zeros([fliters_out]))
    conv = tf.nn.conv2d(input, wconv, strides=[1, stride, stride, 1], padding=pad, name=conv_name)
    conv = tf.nn.bias_add(conv, biases)
    return conv


def relu(input, n_layer):
    relu_name = 'relu' + str(n_layer)
    output = tf.nn.relu(input, name=relu_name)
    return output


def L4(input):
    with tf.variable_scope('L4'):
        with tf.variable_scope('layer1'):
            conv1 = conv2d(input, kernel_size=11, fliters_in=1, fliters_out=48, n_layer=1, init=xavier_initializer())
            relu1 = relu(conv1, n_layer=1)
        with tf.variable_scope('layer2'):
            conv2 = conv2d(relu1, kernel_size=3, fliters_in=48, fliters_out=64, n_layer=2, init=xavier_initializer())
            relu2 = relu(conv2, n_layer=1)
        with tf.variable_scope('layer3'):
            conv3 = conv2d(relu2, kernel_size=3, fliters_in=64, fliters_out=64, n_layer=3, init=xavier_initializer())
            relu3 = relu(conv3, n_layer=1)
        with tf.variable_scope('layer4'):
            conv4 = conv2d(relu3, kernel_size=5, fliters_in=64, fliters_out=1, n_layer=4, init=None)
    output = conv4 + input
    return output


def L8(input):
    with tf.variable_scope('L8'):
        with tf.variable_scope('layer1'):
            conv1 = conv2d(input, kernel_size=11, fliters_in=1, fliters_out=32, n_layer=1, init=xavier_initializer())
            relu1 = relu(conv1, n_layer=1)
        with tf.variable_scope('layer2'):
            conv2 = conv2d(relu1, kernel_size=3, fliters_in=32, fliters_out=64, n_layer=2, init=xavier_initializer())
            relu2 = relu(conv2, n_layer=1)
        with tf.variable_scope('layer3'):
            conv3 = conv2d(relu2, kernel_size=3, fliters_in=64, fliters_out=64, n_layer=3, init=xavier_initializer())
            relu3 = relu(conv3, n_layer=1)
        with tf.variable_scope('layer4'):
            conv4 = conv2d(relu3, kernel_size=3, fliters_in=64, fliters_out=64, n_layer=1, init=xavier_initializer())
            relu4 = relu(conv4, n_layer=1)
            relu4 = tf.concat([relu1, relu4], axis=-1)
        with tf.variable_scope('layer5'):
            conv5 = conv2d(relu4, kernel_size=1, fliters_in=96, fliters_out=64, n_layer=2, init=xavier_initializer())
            relu5 = relu(conv5, n_layer=1)
        with tf.variable_scope('layer6'):
            conv6 = conv2d(relu5, kernel_size=5, fliters_in=64, fliters_out=64, n_layer=3, init=xavier_initializer())
            relu6 = relu(conv6, n_layer=1)
            relu6 = tf.concat([relu1, relu6], axis=-1)
        with tf.variable_scope('layer7'):
            conv7 = conv2d(relu6, kernel_size=1, fliters_in=96, fliters_out=128, n_layer=1, init=xavier_initializer())
            relu7 = relu(conv7, n_layer=1)
        with tf.variable_scope('layer8'):
            conv8 = conv2d(relu7, kernel_size=5, fliters_in=128, fliters_out=1, n_layer=2, init=None)
    output = conv8 + input
    return output
