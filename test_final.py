# -*- coding: utf-8 -*-
import read_tfr_img_dect
from models_final import *
import os
import py_tools
import glob
import numpy as np
from scipy import io
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from functions import *

import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def cat_dedata(input):
    output = tf.expand_dims(tf.concat([input[:, :, :, 0], input[:, :, :, 1]], axis=1), dim=3)
    return output

BatchSize = 1
Pixel = 512
Channel = 2

model_name = 'timeneta105'

img_input_holder = tf.placeholder(tf.float32, [BatchSize, Pixel, Pixel, Channel])
de_results = TimeNet(img_input_holder, chl=8, num=32, reuse=False)
out_results = cat_dedata(de_results)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess, './logs/' + model_name + '/checkpoints/' + model_name + '-98')

testdir = './logs/' + model_name + '/test/'
if os.path.exists(testdir) == False:
    os.makedirs(testdir)

# test_num, test_data = read_raw_data_all('./DataTIME/DECT/limited_ct_105/C/', 2*Pixel, Pixel, 0, -15)
test_num, test_data = read_raw_data_all('./DataTIME/DECT/limited_ct_105/D/', 2*Pixel, Pixel, 0, -15)

test_data = test_data + 1024
out_data = np.zeros([test_num, 2*Pixel, Pixel], dtype=np.float32)

for test_index in range(test_num):

    net_out = sess.run(out_results, feed_dict={
        img_input_holder: np.reshape(np.transpose(np.reshape(test_data[test_index, :, :], [2, Pixel, Pixel]), [1, 2, 0]), [1, Pixel, Pixel, Channel])})
    net_out = np.reshape(net_out, [2 * Pixel, Pixel])

    out_data[test_index, :, :] = net_out

out_data[out_data < 0] = 0
output = out_data - 1024
output = np.around(output)
output.astype(np.float32).tofile(testdir + 'D' + '_' + model_name + '_epoch_98_test.raw')