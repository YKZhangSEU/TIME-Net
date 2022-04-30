# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
import cv2

def LossL2(tensor1, tensor2):
    return tf.reduce_mean(tf.square(tensor1 - tensor2))

def LossL1(tensor1, tensor2):
    return tf.reduce_mean(tf.abs(tensor1 - tensor2))

def img_grad(inputs, name='img_grad'):
    with tf.variable_scope(name):
        inputs = inputs / 1000 * 0.0192

        kernel_hor_init = np.reshape(np.array([[-1, 0, 1],
                                               [-2, 0, 2],
                                               [-1, 0, 1], ], dtype=np.float32),
                                     [3, 3, 1, 1])

        kernel_ver_init = np.reshape(np.array([[-1, -2, -1],
                                               [0, 0, 0],
                                               [1, 2, 1], ], dtype=np.float32),
                                     [3, 3, 1, 1])

        kernel_hor = tf.concat([kernel_hor_init] * inputs.shape[-1], 2)
        kernel_ver = tf.concat([kernel_ver_init] * inputs.shape[-1], 2)

        grad_h = tf.nn.depthwise_conv2d(inputs, kernel_hor, [1, 1, 1, 1], 'SAME')
        grad_v = tf.nn.depthwise_conv2d(inputs, kernel_ver, [1, 1, 1, 1], 'SAME')

        grad = 1 / tf.rsqrt(tf.square(grad_h) + tf.square(grad_v))

        # grad = tf.clip_by_value(grad, 100, 1300, name + '_clip')

        # mask = tf.ones_like(grad)
        # mask = tf.where(grad < 100, 0 * mask, mask)
        # # mask = tf.where(grad > 500, 0 * mask, mask)
        # grad = grad * mask
        #
        # grad = tf.clip_by_value(grad, 0, 250, name + '_clip')

        return grad

def LossGrad(tensor1, tensor2):
    return LossL1(img_grad(tensor1), img_grad(tensor2))

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    max_temp = tf.reduce_max(img2)
    img1 = img1 / max_temp
    img2 = img2 / max_temp
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
             (mssim[level - 1] ** weight[level - 1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def GaussianSmooth(input, kernelsize=11, sigma=3.5):
    # inputs: batch, width, height, channel
    f = np.multiply(cv2.getGaussianKernel(kernelsize, sigma), np.transpose(cv2.getGaussianKernel(kernelsize, sigma)))
    #    kernel = tf.reshape(tf.float32(f), [kernelsize, kernelsize, 1, 1], 'kernel')

    #    f = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
    #                  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
    #                  [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
    #                  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
    #                  [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]], dtype=np.float32)
    kernel = tf.reshape(np.float32(f), [kernelsize, kernelsize, 1, 1], 'kernel')

    low = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], name='f1', padding='SAME')
    high = input - low
    return high