import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
# import CNN_util as CNN
# from functools import reduce

def conv2d(input,k,filters_in,filters_out,n_layer,s=1,pad = 'SAME',init=None):
    wshape = [k, k, filters_in, filters_out]
    if init:
        wconv = tf.get_variable('weights_'+n_layer, shape = wshape, dtype = tf.float32, initializer=init)
    else:
        wconv = tf.Variable(tf.random_normal(wshape,mean=0.0,stddev = 0.01),dtype = tf.float32,name = 'weights_'+n_layer)
    wconv = wconv-tf.reduce_mean(wconv) # !!!Note: explicitly make weight have zero mean 
    conv = tf.nn.conv2d(input,wconv,strides=[1,s,s,1],padding=pad,name='conv_'+n_layer)
    return conv
    
def bn(input,is_training,decay=0.999,epsilon=1e-3):
    return slim.layers.batch_norm(input,decay=decay,scale = True, epsilon=epsilon, is_training = is_training,updates_collections = None)
    
def relu(input,n_layer):
    relu_name = 'relu_' + n_layer
    output = tf.nn.relu(input,name=relu_name)
    return output
    
def L4(J):# basic residual with  last conv-layer
    with tf.variable_scope('L4'):
        with tf.variable_scope('layer_1'):
            conv1=conv2d(J,11,1,48,'1',init=xavier_initializer())
            relu1=relu(conv1,'1')
        with tf.variable_scope('layer_2'):
            conv2=conv2d(relu1,3,48,64,'2',init=xavier_initializer())
            relu2=relu(conv2,'2')
        with tf.variable_scope('layer_3'):
            conv3=conv2d(relu2,3,64,64,'3',init=xavier_initializer())
            relu3=relu(conv3,'3')
        with tf.variable_scope('layer_4'):
            conv4=conv2d(relu3,5,64,1,'4')   
            relu4=relu(conv4,'4') 
        out=J+relu4   # residual learning
    onemtx=tf.ones(tf.shape(out))
    zeromtx=tf.zeros(tf.shape(out))
    out=tf.where(tf.where(out>1,onemtx,out)<0,zeromtx,out)
    # out = tf.sigmoid(out)
    return relu4, out
    
def L8(J):
    conv1=conv2d(J,11,1,32,'1')
    relu1=relu(conv1,'1')
    conv2=conv2d(relu1,3,32,64,'2')
    relu2=relu(conv2,'2')
    conv3=conv2d(relu2,3,64,64,'3')
    relu3=relu(conv3,'3')
    conv4=conv2d(relu3,3,64,64,'4')
    relu4=relu(conv4,'4')
    relu4=tf.concat([relu4,relu1],-1)    # skip architecture  
    conv5=conv2d(relu4,3,96,64,'5')
    relu5=relu(conv5,'5')
    conv6=conv2d(relu5,3,64,64,'6')
    relu6=relu(conv6,'6')
    relu6=tf.concat([relu6,relu1],-1)    # skip architecture 
    conv7=conv2d(relu6,3,96,128,'7')
    relu7=relu(conv7,'7')
    conv8=conv2d(relu7,5,128,1,'8')  
    out=J+conv8     # residual learning
    onemtx=tf.ones(tf.shape(out))
    zeromtx=tf.zeros(tf.shape(out))
    out=tf.where(tf.where(out>1,onemtx,out)<0,zeromtx,out)
    return  conv8,out
    
def CNN4Tst(J):
    w1 = tf.get_variable('weights_conv1', shape = [3,3,1,64], dtype = tf.float32, initializer=xavier_initializer())
    conv1 = tf.nn.atrous_conv2d(J, w1, rate=1, padding='SAME', name='conv_1')
    relu1 = relu(conv1)
    w2 = tf.get_variable('weights_conv2', shape = [3,3,1,64], dtype = tf.float32, initializer=xavier_initializer())
    conv2 = tf.nn.atrous_conv2d(relu1, w2, rate=2, padding='SAME', name='conv_2')
    relu2 = relu(conv2)
    w3 = tf.get_variable('weights_conv3', shape = [3,3,1,1], dtype = tf.float32, initializer=xavier_initializer())
    conv3 = tf.nn.atrous_conv2d(relu2, w2, rate=4, padding='SAME', name='conv_3')
    out=J+conv3
    return conv3,out
    
def jpg_af_cnn(J):
    R,O=L4(J)
    return R,O