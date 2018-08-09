import tensorflow as tf
import os
import cv2
import numpy as np
import auxi
import time
from model import DCT_Branch, Pixel_Branch, get_batch, DCT_Layer, IDCT_Layer, DRU_Layer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

info = '/'
QF = 20
BATCH_SIZE = 4
PATCHS = [56, 112, 224]
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 3
GLOBAL_STEPS = 30000
DISPLAY_STEP = 10
DECAY_STEP = 30
VALIDATION_STEP = 100
VALIDATION_SET = 10

dct_model_dir = os.getcwd() + info + '/network' + '/dct_model/' + str(QF) + '/'
pixel_model_dir = os.getcwd() + info + '/network' + '/pixel_model/' + str(QF) + '/'
if not os.path.exists(dct_model_dir):
    os.makedirs(dct_model_dir)
if not os.path.exists(pixel_model_dir):
    os.makedirs(pixel_model_dir)
landa = 0.9
sita = 0.618
train_O = '/data1/zhanghx/DMCNN/data/train_O/'
train_J = '/data1/zhanghx/DMCNN/data/train_J/20/'
val_O = '/data1/zhanghx/DMCNN/data/val_O/'
val_J = '/data1/zhanghx/DMCNN/data/val_J/20/'
test_O = '/data1/zhanghx/DMCNN/data/test_O/'
test_J = '/data1/zhanghx/DMCNN/data/test_J/20/'

table1 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

with tf.name_scope('input') as scope:
    # x = tf.placeholder(tf.float32, [None, None, None, 1])  # compressed image
    # y = tf.placeholder(tf.float32, [None, None, None, 1])  # original image
    x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])  # compressed image
    y = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])  # original image
    dct_x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])
    dct_y = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])
    idct_x = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])
    idct_y = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1])
    lr = tf.placeholder(tf.float32, [1])
with tf.name_scope('DCT_BRANCH') as scope:
    dru_x = DCT_Branch(dct_x)
    dru_y = DCT_Branch(dct_y)
with tf.name_scope('Pixel_BRANCH') as scope:
    gen_image_x, gen_1_x, gen_2_x = Pixel_Branch(Input_Pixel=x, Input_IDCT=idct_x)
    gen_image_y, gen_1_y, gen_2_y = Pixel_Branch(Input_Pixel=y, Input_IDCT=idct_y)

with tf.name_scope('loss') as scope:
    dct_loss = tf.norm(idct_x - y) * landa
    mse_loss0 = tf.norm(gen_image_x - y) * 1.
    mse_loss1 = tf.norm(gen_1_x - gen_1_y) * sita
    mse_loss2 = tf.norm(gen_2_x - gen_2_y) * sita * sita
    mse_loss = mse_loss0+mse_loss1+mse_loss2

    # losses = dct_loss
    # losses = mse_loss0
    #losses = dct_loss + mse_loss0
    # losses = mse_loss
     losses = mse_loss + dct_loss

    tf.add_to_collection('losses', losses)
    loss = tf.add_n(tf.get_collection('losses'))

global_ = tf.Variable(0, trainable=False)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss, global_step=global_)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    step = 0
    val_dct_loss_list = [10000]
    val_pixel_loss_list = [10000]
    while step < (GLOBAL_STEPS + 1):
        i = int(step / (GLOBAL_STEPS / 3))
        batch_O, batch_J = get_batch(train_O, train_J, BATCH_SIZE, PATCHS[0])
        dct_batch_O, dct_batch_J = get_batch(train_O, train_J, BATCH_SIZE, PATCHS[0], dct=True)
        dru_batch_O, dru_batch_J = sess.run([dru_y, dru_x], feed_dict={dct_y: dct_batch_O, dct_x: dct_batch_J})
        dru_batch_O = DRU_Layer(ae_out=dru_batch_O, dct_in=dct_batch_O, q_table=table1)
        dru_batch_J = DRU_Layer(ae_out=dru_batch_J, dct_in=dct_batch_J, q_table=table1)
        idct_batch_O = IDCT_Layer(dru_batch_O)
        idct_batch_J = IDCT_Layer(dru_batch_J)
        _, dct_cast, pixel_cast, x_gens = sess.run([train_op, dct_loss, mse_loss, gen_image_x],
                                                   feed_dict={y: batch_O, x: batch_J, dct_y: dct_batch_O,
                                                              dct_x: dct_batch_J, idct_y: idct_batch_O,
                                                              idct_x: idct_batch_J})
        if step % DISPLAY_STEP == 0:
            PSNR1, SSIM1 = auxi.eval_compare(batch_O, batch_J, batch_size=BATCH_SIZE)
            PSNR2, SSIM2 = auxi.eval_compare(batch_O, x_gens, batch_size=BATCH_SIZE)
            print('The', step, 'steps in training:')
            print('dct_loss:', dct_cast, 'pixel_loss:', pixel_cast)
            print('PSNR1 ', PSNR1, ' SSIM1 ', SSIM1)
            print('PSNR2 ', PSNR2, ' SSIM2 ', SSIM2)
        # validation
        if step >= VALIDATION_STEP:
            if step % VALIDATION_STEP == 0:
                print('Testing the validation set.....')
                batch_O, batch_J = get_batch(train_O, train_J, BATCH_SIZE, PATCHS[0])
                dct_batch_O, dct_batch_J = get_batch(train_O, train_J, BATCH_SIZE, PATCHS[0], dct=True)
                dru_batch_O, dru_batch_J = sess.run([dru_y, dru_x], feed_dict={dct_y: dct_batch_O, dct_x: dct_batch_J})
                dru_batch_O = DRU_Layer(ae_out=dru_batch_O, dct_in=dct_batch_O, q_table=table1)
                dru_batch_J = DRU_Layer(ae_out=dru_batch_J, dct_in=dct_batch_J, q_table=table1)
                idct_batch_O = IDCT_Layer(dru_batch_O)
                idct_batch_J = IDCT_Layer(dru_batch_J)
                _, dct_cast, pixel_cast, x_gens = sess.run([train_op, dct_loss, mse_loss, gen_image_x],
                                                           feed_dict={y: batch_O, x: batch_J, dct_y: dct_batch_O,
                                                                      dct_x: dct_batch_J, idct_y: idct_batch_O,
                                                                      idct_x: idct_batch_J})
                PSNR1, SSIM1 = auxi.eval_compare(batch_O, batch_J, batch_size=BATCH_SIZE)
                PSNR2, SSIM2 = auxi.eval_compare(batch_O, x_gens, batch_size=BATCH_SIZE)
                print('The', step, 'steps in validtion:')
                print('dct_loss:', dct_cast, 'pixel_loss:', pixel_cast)
                print('PSNR1 ', PSNR1, ' SSIM1 ', SSIM1)
                print('PSNR2 ', PSNR2, ' SSIM2 ', SSIM2)
                if dct_cast < min(val_dct_loss_list):
                    savepath = dct_model_dir + str(step) + '_iters' + ' lr' + str(LEARNING_RATE) + '.ckpt'
                    saver = tf.train.Saver()
                    saver.save(sess, savepath)
                    print('The dct_model of lr' + str(LEARNING_RATE) + str(
                        step) + '_iters' + 'save to path' + dct_model_dir)
                else:
                    LEARNING_RATE = LEARNING_RATE / LEARNING_RATE_DECAY
                val_dct_loss_list.append(dct_cast)
                if pixel_cast < min(val_pixel_loss_list):
                    savepath = pixel_model_dir + str(step) + '_iters' + ' lr' + str(LEARNING_RATE) + '.ckpt'
                    saver = tf.train.Saver()
                    saver.save(sess, savepath)
                    print('The pixel_model of lr' + str(LEARNING_RATE) + str(
                        step) + '_iters' + 'save to path' + pixel_model_dir)
                else:
                    LEARNING_RATE = LEARNING_RATE / LEARNING_RATE_DECAY
                val_pixel_loss_list.append(pixel_cast)
        step += 1
    print(time.asctime() + '   Train Finished.')
