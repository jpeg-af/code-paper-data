import os
import numpy as np
import tensorflow as tf
from model import L4, L8
from scipy.misc import imread
import auxi

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
LEARNIN_RATE = 1e-5
LEARNIN_RATE_DECARY = 0.5
GLOBAL_STEP = 250000
DECAY_STEP = 50000
DISPLAY_STEP = 10
BATCH_SIZE = 64
PATCH_WIDTH = 64
PATCH_HEIGHT = 64

QF=20
train_O='/data1/zhanghx/crk/paper/data/trainO/'
train_J='/data1/zhanghx/crk/paper/data/trainJ/'+str(QF)+'/'
test_O='/data1/zhanghx/crk/paper/data/testO/'
test_J='/data1/zhanghx/crk/paper/data/testJ/'+str(QF)+'/'

model_dir = os.getcwd() + '/networkL4/'+str(QF)+ 'model/'
log_dir = os.getcwd() + '/networkL4/' +str(QF)+ 'logs/'

output_dir = ['./result/' + '/' + '/output/O/',
              './result/' + '/' + '/output/J/',
              './result/' + '/' + '/output/G/']

mkdirs_func = lambda p: os.makedirs(p) if not os.path.exists(p) else 0
for d in [model_dir, log_dir] + output_dir: mkdirs_func(d)


def get_batch(path_train_x,path_train_y):  #train_x:original  train_y:jpeg
    list_fname = os.listdir(path_train_x)
    perm = np.arange(len(list_fname)).astype(np.int32)
    np.random.shuffle(perm)
    x_batch=np.zeros([BATCH_SIZE,PATCH_WIDTH,PATCH_HEIGHT,1])
    y_batch=np.zeros([BATCH_SIZE,PATCH_WIDTH,PATCH_HEIGHT,1])
    for i in range(BATCH_SIZE):
        file = list_fname[perm[i]]
        img_x = imread(path_train_x + file, mode='L')
        img_y = imread(path_train_y + file[:-3]+'jpg', mode='L')
        img_x = img_x.reshape([img_x.shape[0],img_x.shape[1],1])
        img_y = img_y.reshape([img_y.shape[0],img_y.shape[1],1])
        pos_w = np.random.randint(img_x.shape[0]-PATCH_WIDTH,size=1)
        pos_h = np.random.randint(img_x.shape[1]-PATCH_HEIGHT,size=1)
        x_batch[i,...] = img_x[pos_w[0]:pos_w[0]+PATCH_WIDTH,pos_h[0]:pos_h[0]+PATCH_HEIGHT,:]
        y_batch[i,...] = img_y[pos_w[0]:pos_w[0]+PATCH_WIDTH,pos_h[0]:pos_h[0]+PATCH_HEIGHT,:]
    return x_batch/256., y_batch/256.


with tf.name_scope('input') as scope:
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # J
    x = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # O

x_gen = L4(y)

with tf.name_scope('loss') as scope:
    residual_loss = tf.norm(x - x_gen)
    tf.add_to_collection('losses', residual_loss)
    loss = tf.add_n(tf.get_collection('losses'))

vars = tf.trainable_variables()
var1_3 = tf.trainable_variables()[0:6]
var4 = tf.trainable_variables()[6:]
print(len(vars))
print(var1_3)
print(var4)

global_ = tf.Variable(0, trainable=False)
decayed_learning_rate = tf.train.exponential_decay(learning_rate=LEARNIN_RATE, global_step=global_,
                                                   decay_steps=DECAY_STEP, decay_rate=LEARNIN_RATE_DECARY,
                                                   staircase=True)
train_op1 = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate).minimize(loss, global_step=global_,var_list=var1_3)
train_op2 = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate * 0.1).minimize(loss, global_step=global_,var_list=var4)
train_op = tf.group(train_op1, train_op2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=0)


    for step in range(GLOBAL_STEP + 1):
        x_batch, y_batch = get_batch(train_O, train_J)
        _,x_gens,loss_val=sess.run([train_op,x_gen,loss], feed_dict={x: x_batch, y: y_batch})
        if step % DISPLAY_STEP == 0:
            PSNR1, SSIM1 = auxi.eval_compare(x_batch, y_batch, batch_size=BATCH_SIZE)
            PSNR2, SSIM2 = auxi.eval_compare(x_batch, x_gens, batch_size=BATCH_SIZE)
            print('The', step, 'steps in training ', 'loss: ', loss_val)
            print('PSNR1 ', PSNR1, ' SSIM1 ', SSIM1)
            print('PSNR2 ', PSNR2, ' SSIM2 ', SSIM2)
    print('Train done')
    save_path = model_dir + '.ckpt'
    saver.save(sess, save_path)
    print('Model saved done')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model loaded done')

    PSNR1_test, SSIM1_test,PSNR2_test, SSIM2_test=(0,0,0,0)
    list_fname = os.listdir(test_O)
    for i, file in enumerate(list_fname):
        img_x = imread(test_O + file, mode='L')
        img_y = imread(test_J + file[:-3] + 'jpg', mode='L')
        img_x = img_x.reshape([1, img_x.shape[0], img_x.shape[1], 1]) / 255.
        img_y = img_y.reshape([1, img_y.shape[0], img_y.shape[1], 1]) / 255.
        x_gens = sess.run(x_gen, feed_dict={x: img_x, y: img_y})

        PSNR1, SSIM1 = auxi.eval_compare(img_x, img_y, batch_size=1)
        PSNR2, SSIM2 = auxi.eval_compare(img_x, x_gens, batch_size=1)

        print('PSNR1 ', PSNR1, ' SSIM1 ', SSIM1)
        print('PSNR2 ', PSNR2, ' SSIM2 ', SSIM2)

        PSNR1_test=PSNR1+PSNR1_test
        SSIM1_test=SSIM1+SSIM1_test
        PSNR2_test=PSNR2+PSNR2_test
        SSIM2_test=SSIM2+SSIM2_test
    print('Text done')

    txt_file = open("all_resultsL4", "a")
    txt_file.write("L4 : \n")
    txt_file.write('QF:'+str(QF)+ '\n')
    txt_file.write('PSNR1:' + str(PSNR1_test/ len(list_fname)) + '\n')
    txt_file.write('SSIM1:' + str(SSIM1_test/ len(list_fname)) + '\n')
    txt_file.write('PSNR2:' + str(PSNR2_test/ len(list_fname)) + '\n')
    txt_file.write('SSIM2:' + str(SSIM2_test/ len(list_fname)) + '\n')
    txt_file.close()