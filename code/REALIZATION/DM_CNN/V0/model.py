import tensorflow as tf
import cv2
import numpy as np
import os
from scipy.misc import imread





def get_batch(path_train_x, path_train_y, BATCH, PATCH, dct=False):  # train_x:original  train_y:jpeg
    list_fname = os.listdir(path_train_x)
    perm = np.arange(len(list_fname)).astype(np.int32)
    np.random.shuffle(perm)
    x_batch = np.zeros([BATCH, PATCH, PATCH, 1])
    y_batch = np.zeros([BATCH, PATCH, PATCH, 1])
    for i in range(BATCH):
        file = list_fname[perm[i]]
        img_x = imread(path_train_x + file, mode='L')
        img_y = imread(path_train_y + file[:-3] + 'jpg', mode='L')
        img_x = img_x.reshape([img_x.shape[0], img_x.shape[1], 1])
        img_y = img_y.reshape([img_y.shape[0], img_y.shape[1], 1])
        pos_w = np.random.randint(img_x.shape[0] - PATCH, size=1)
        pos_h = np.random.randint(img_x.shape[1] - PATCH, size=1)
        x_batch[i, ...] = img_x[pos_w[0]:pos_w[0] + PATCH, pos_h[0]:pos_h[0] + PATCH, :]
        y_batch[i, ...] = img_y[pos_w[0]:pos_w[0] + PATCH, pos_h[0]:pos_h[0] + PATCH, :]
    if not dct:
        return x_batch, y_batch
    if dct:
        x_batch = DCT_Layer(x_batch)
        y_batch = DCT_Layer(y_batch)
        return x_batch, y_batch


def DCT_Layer(BATCH):
    for i, batch in enumerate(BATCH):
        batch = cv2.dct(batch)
        BATCH[i] = batch.reshape(batch.shape[0], batch.shape[1], 1)
    dct = BATCH
    return dct


def IDCT_Layer(BATCH):
    for i, batch in enumerate(BATCH):
        batch = cv2.idct(batch)
        BATCH[i] = batch.reshape(batch.shape[0], batch.shape[1], 1)
    idct = BATCH
    return idct


def DRU_Layer(ae_out, dct_in, q_table):
    shape = dct_in.shape
    q_table = q_table.reshape([1, 8, 8, 1])
    q_table = np.tile(q_table, [shape[0], int(shape[1] / 8), int(shape[2] / 8), shape[3]])
    min_ = dct_in - (q_table / 2)
    max_ = dct_in + (q_table / 2)
    less = np.where(ae_out < min_)
    more = np.where(ae_out > max_)
    ae_out[less] = min_[less]
    ae_out[more] = max_[more]
    dru = ae_out
    return dru


def dilated_conv2d(input, kernel_size, filters_in, filters_out, rate, n_layer, padding='SAME'):
    # tf.nn.atrous_conv2d(value=1,filters=1,rate=1,padding='SAME',name=)
    # value=[batch, height, width, channels]
    # filters=[filter_height, filter_width, channels, out_channels]
    wconv_name = 'weights' + str(n_layer)
    conv_name = 'conv' + str(n_layer)
    wshape = [kernel_size, kernel_size, filters_in, filters_out]
    wconv = tf.Variable(tf.random_normal(shape=wshape, mean=0., stddev=0.005), dtype=tf.float32, name=wconv_name)
    return tf.nn.atrous_conv2d(value=input, filters=wconv, rate=rate, padding=padding, name=conv_name)


def deconv(input, kernel_size, filters_in, filters_out, output_shape, n_layer, rate, padding='SAME'):
    # value,filter,  disable=redefined-builtin output_shape,strides,padding = "SAME",data_format = "NHWC",name = None
    wshape = [kernel_size, kernel_size, filters_out, filters_in]
    dewconv_name = 'deweights' + str(n_layer)
    deconv_name = 'deconv' + str(n_layer)
    wconv = tf.Variable(tf.random_normal(shape=wshape, mean=0., stddev=0.005), dtype=tf.float32, name=dewconv_name)
    outputs_shape = [output_shape[0], output_shape[1], output_shape[2], filters_out]
    return tf.nn.atrous_conv2d_transpose(value=input, filters=wconv, output_shape=outputs_shape, rate=rate,
                                         padding=padding, name=deconv_name)


def relu(input, n_layer, alpha=0.1):
    relu_name = 'prelu' + str(n_layer)
    return tf.maximum(alpha * input, input, name=relu_name)


def derelu(input, n_layer, alpha=0.1):
    relu_name = 'deprelu' + str(n_layer)
    return tf.maximum(alpha * input, input, name=relu_name)


def DCT_Branch(Input):
    # input:DCT output:DCT before DRU
    dct = Input
    with tf.variable_scope('DCT-Auto-Encoder'):
        with tf.variable_scope('conv'):
            conv1 = dilated_conv2d(input=dct, kernel_size=3, filters_in=1, filters_out=16, rate=1, n_layer=1)
            relu1 = relu(conv1, n_layer=1)
            conv2 = dilated_conv2d(input=relu1, kernel_size=3, filters_in=16, filters_out=32, rate=1, n_layer=2)
            relu2 = relu(conv2, n_layer=2)
            conv3 = dilated_conv2d(input=relu2, kernel_size=3, filters_in=32, filters_out=64, rate=2, n_layer=3)
            relu3 = relu(conv3, n_layer=3)
            conv4 = dilated_conv2d(input=relu3, kernel_size=3, filters_in=64, filters_out=128, rate=1, n_layer=4)
            relu4 = relu(conv4, n_layer=4)
            conv5 = dilated_conv2d(input=relu4, kernel_size=3, filters_in=128, filters_out=256, rate=4, n_layer=5)
            relu5 = relu(conv5, n_layer=5)
            conv6 = dilated_conv2d(input=relu5, kernel_size=3, filters_in=256, filters_out=512, rate=1, n_layer=6)
            relu6 = relu(conv6, n_layer=6)
            conv7 = dilated_conv2d(input=relu6, kernel_size=3, filters_in=512, filters_out=1024, rate=8, n_layer=7)
            relu7 = relu(conv7, n_layer=7)
            conv8 = dilated_conv2d(input=relu7, kernel_size=3, filters_in=1024, filters_out=2048, rate=1, n_layer=8)
            relu8 = relu(conv8, n_layer=8)
        with tf.variable_scope('dconv'):
            decon7 = deconv(input=relu8, kernel_size=3, filters_in=2048, filters_out=1024,
                            output_shape=tf.shape(conv7), n_layer=7, rate=1)
            derelu7 = derelu(input=decon7, n_layer=7)
            decon6 = deconv(input=derelu7, kernel_size=3, filters_in=1024, filters_out=512,
                            output_shape=tf.shape(conv6), n_layer=6, rate=8)
            derelu6 = derelu(input=decon6, n_layer=6)
            decon5 = deconv(input=derelu6, kernel_size=3, filters_in=512, filters_out=256,
                            output_shape=tf.shape(conv5), n_layer=5, rate=1)
            derelu5 = derelu(input=decon5, n_layer=5)
            decon4 = deconv(input=derelu5, kernel_size=3, filters_in=256, filters_out=128,
                            output_shape=tf.shape(conv4), n_layer=4, rate=4)
            derelu4 = derelu(input=decon4, n_layer=4)
            decon3 = deconv(input=derelu4, kernel_size=3, filters_in=128, filters_out=64,
                            output_shape=tf.shape(conv3), n_layer=3, rate=1)
            derelu3 = derelu(input=decon3, n_layer=3)
            decon2 = deconv(input=derelu3, kernel_size=3, filters_in=64, filters_out=32,
                            output_shape=tf.shape(conv2), n_layer=2, rate=2)
            derelu2 = derelu(input=decon2, n_layer=2)
            decon1 = deconv(input=derelu2, kernel_size=3, filters_in=32, filters_out=16,
                            output_shape=tf.shape(conv1), n_layer=1, rate=1)
            derelu1 = derelu(input=decon1, n_layer=1)
            recon = deconv(input=derelu1, kernel_size=3, filters_in=16, filters_out=1,
                           output_shape=tf.shape(dct), n_layer=0, rate=1)
    with tf.variable_scope('Concat'):
        concat = dct + recon
    with tf.variable_scope('Output'):
        output = concat
    return output


def Pixel_Branch(Input_Pixel, Input_IDCT):
    with tf.variable_scope('Input'):
        input1 = Input_Pixel
        input2 = Input_IDCT
    with tf.variable_scope('Pixel-Auto-Encoder'):
        input = input1 + input2
        with tf.variable_scope('conv'):
            conv1 = dilated_conv2d(input=input, kernel_size=3, filters_in=1, filters_out=16, rate=1, n_layer=1)
            relu1 = relu(conv1, n_layer=1)
            conv2 = dilated_conv2d(input=relu1, kernel_size=3, filters_in=16, filters_out=32, rate=2, n_layer=2)
            relu2 = relu(conv2, n_layer=2)
            conv3 = dilated_conv2d(input=relu2, kernel_size=3, filters_in=32, filters_out=64, rate=4, n_layer=3)
            relu3 = relu(conv3, n_layer=3)
            conv4 = dilated_conv2d(input=relu3, kernel_size=3, filters_in=64, filters_out=128, rate=1, n_layer=4)
            relu4 = relu(conv4, n_layer=4)
            conv5 = dilated_conv2d(input=relu4, kernel_size=3, filters_in=128, filters_out=256, rate=8, n_layer=5)
            relu5 = relu(conv5, n_layer=5)
        with tf.variable_scope('dconv'):
            decon4 = deconv(input=relu5, kernel_size=3, filters_in=256, filters_out=128,
                            output_shape=tf.shape(conv4), n_layer=4, rate=8)
            derelu4 = derelu(input=decon4, n_layer=4)
            decon3 = deconv(input=derelu4, kernel_size=3, filters_in=128, filters_out=64,
                            output_shape=tf.shape(conv3), n_layer=3, rate=1)
            derelu3 = derelu(input=decon3, n_layer=3)
            decon2 = deconv(input=derelu3, kernel_size=3, filters_in=64, filters_out=32,
                            output_shape=tf.shape(conv2), n_layer=2, rate=4)
            derelu2 = derelu(input=decon2, n_layer=2)
            decon1 = deconv(input=derelu2, kernel_size=3, filters_in=32, filters_out=16,
                            output_shape=tf.shape(conv1), n_layer=1, rate=2)
            derelu1 = derelu(input=decon1, n_layer=1)

            recon = deconv(input=derelu1, kernel_size=1, filters_in=16, filters_out=1,
                           output_shape=tf.shape(input), n_layer=0, rate=1)
    with tf.variable_scope('Concat'):
        concat = recon + (0.5 * Input_IDCT) + (0.5 * Input_Pixel)
    with tf.variable_scope('Output'):
        O = concat
        O1 = derelu1
        O2 = derelu2
    return O, O1, O2


