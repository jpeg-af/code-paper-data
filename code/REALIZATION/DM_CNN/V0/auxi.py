from skimage.measure import compare_psnr, compare_ssim
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave

# import pdb

dct_index = [1, 9, 2, 3, 10, 17, 25, 18, \
             11, 4, 5, 12, 19, 26, 33, 41, \
             34, 27, 20, 13, 6, 7, 14, 21, \
             28, 35, 42, 49, 57, 50, 43, 36, \
             29, 22, 15, 8, 16, 23, 30, 37, \
             44, 51, 58, 59, 52, 45, 38, 31, \
             24, 32, 39, 46, 53, 60, 61, 54, \
             47, 40, 48, 55, 62, 63, 56, 64]
train_O = 'E:/anti-forensics/week3/forensic_method/testO/'
train_J = 'E:/anti-forensics/week3/forensic_method/testJ/20/'

def eval_compare(batch_X, batch_Y, batch_size=50):
    batch_X = img_fmt(batch_X)
    batch_Y = img_fmt(batch_Y)
    PSNR, SSIM = (0, 0)
    for i in range(batch_size):
        PSNR += compare_psnr(batch_X[i, :, :, 0], batch_Y[i, :, :, 0])
        SSIM += compare_ssim(batch_X[i, :, :, 0], batch_Y[i, :, :, 0])
    meanPSNR = PSNR / batch_size
    meanSSIM = SSIM / batch_size
    return meanPSNR, meanSSIM


def img_fmt(batch_X):
    return np.asarray(batch_X * 255., dtype=np.uint8)


def bdctmtx(n=8):
    ind = np.arange(8)
    c, r = np.meshgrid(ind, ind)
    c0, r0 = np.meshgrid(c, c)
    c1, r1 = np.meshgrid(r, r)
    x = np.sqrt(2. / n) * np.cos(np.pi * (2. * c + 1.) * r / (2. * n))
    x[0, :] = x[0, :] / np.sqrt(2.)
    x_plain = x.reshape([-1])
    m = x_plain[r0 + c0 * n] * x_plain[r1 + c1 * n]
    return m


def im2vec(im, bsize, padsize=0):
    bsize = [0 + bsize, 0 + bsize]
    if padsize < 0:
        print('Error: Pad size must not be negative')
        return False
    padsize = [0 + padsize, 0 + padsize]
    imshape = im.shape
    y = bsize[0] + padsize[0]
    x = bsize[1] + padsize[1]
    rows = np.floor((imshape[0] + padsize[0]) / y).astype(np.int32)
    cols = np.floor((imshape[1] + padsize[1]) / x).astype(np.int32)

    t = np.zeros([x * cols, y * rows])
    imy = y * rows - padsize[0]
    imx = x * cols - padsize[0]
    t[0:imx, 0:imy] = im[0:imx, 0:imy]
    t = t.reshape([cols, x, rows, y])
    t = t.transpose([0, 2, 1, 3])
    t = t.reshape([rows * cols, x, y])
    v = t[0:rows * cols, 0:bsize[1], 0:bsize[0]]
    v = v.reshape([rows * cols, y * x])
    return v, rows, cols


def vec2im(v, padsize=0, bsize=None, rows=None, cols=None):
    n, m = v.shape
    if padsize < 0:
        print('Error: Pad size must not be negative')
        return False
    padsize = [0 + padsize, 0 + padsize]
    if bsize is None:
        bsize = np.floor(np.sqrt(m))
    bsize = [0 + bsize, 0 + bsize]
    if rows is None:
        rows = np.floor(np.sqrt(n))
    if rows is None:
        cols = np.ceil(n / rows).astype(np.int32)
    y = bsize[0] + padsize[0]
    x = bsize[1] + padsize[1]
    t = np.zeros([rows * cols, x, y])
    t[0:n, 0:bsize[1], 0:bsize[1]] = v.reshape(n, bsize[1], bsize[0])
    t = t.reshape([cols, rows, x, y])
    t = t.transpose([0, 2, 1, 3])
    t = t.reshape([x * cols, y * rows])
    im = t[0:x * cols - padsize[1], 0:y * rows - padsize[0]]
    return im


def bdct(a, n=8):
    # pdb.set_trace()
    dctm = bdctmtx(n)
    v, r, c = im2vec(a.T, n)
    b = vec2im(v.dot(dctm), bsize=n, rows=r, cols=c)
    return b.T


def dct_hist(im, n=8, bandnum=64, binRange=None):
    if binRange is None:
        binRange = np.arange(-50, 52)
    hist_data = np.zeros([1, (len(binRange) - 1) * bandnum])
    y, x = im.shape
    yblk = y // n
    xblk = x // n
    im_dct = bdct(im, n=n)
    im_dct = np.round(im_dct)
    v_dct, _, _ = im2vec(im_dct.T, n)
    dct_subbands = v_dct.reshape([xblk, yblk, n * n])
    for i in range(bandnum):
        # print(i)
        dct_subband = dct_subbands[:, :, dct_index[i] - 1].reshape([xblk * yblk, 1])
        hist_data[0, i * (len(binRange) - 1):(i + 1) * (len(binRange) - 1)], _ = np.histogram(dct_subband, binRange)
    return hist_data


def batch_dct_hist(batch_X, n=8, bandnum=64, binRange=None):
    xshape = batch_X.shape
    batch_X = img_fmt(batch_X)
    if binRange is None:
        binRange = np.arange(-50, 52)
    batch_hist = np.zeros([0, bandnum * (len(binRange) - 1)])
    for i in range(xshape[0]):
        im_2d = batch_X[i].reshape(xshape[1:3])
        single_hist = dct_hist(im_2d, n, bandnum, binRange)
        batch_hist = np.concatenate((batch_hist, single_hist))
    return batch_hist


'''
def save_batch(imgs,path):
    for i in range(imgs.shape[0]):
        imsave(path+'%s.png'%i,img_fmt(imgs[i,:,:,0]))
'''


def save_gray_imgs(imgs, path=None, names=None):
    num = imgs.shape[0]
    # print(num)
    if not path:
        path = os.getcwd()
    if not names:
        names = list(map(lambda x: str(x), range(num)))
    for i in range(num):
        imsave(path + '%s.png' % names[i], img_fmt(imgs[i, :, :, 0]))


def show_duraionTime(duT):
    print('train step: %.4f sec' % duT[0])
    print('get batch: %.4f sec' % duT[1])
    print('CNN forward: %.4f sec' % duT[2])
    print('dct hist: %.4f sec' % duT[3])
    print('SBCNN preproc: %.4f sec' % duT[4])
    print('optimize: %.4f sec' % duT[5])
