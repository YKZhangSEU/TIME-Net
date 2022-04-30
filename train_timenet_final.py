# -*- coding: utf-8 -*-
from py_tools import *
import read_tfr_img_dect
from models_final import *
import os
import glob
import numpy as np
from scipy import io
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from functions import *
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cat_dedata(input):
    output = tf.expand_dims(tf.concat([input[:, :, :, 0], input[:, :, :, 1]], axis=1), dim=3)
    return output

TrainDataSize = 8000
BatchSize = 4
Pixel = 512
Channel = 2
NumEpoch = 100
TrainIter = TrainDataSize // BatchSize

img_input_holder = tf.placeholder(tf.float32, [BatchSize, Pixel, Pixel, Channel])
img_label_holder = tf.placeholder(tf.float32, [BatchSize, Pixel, Pixel, Channel])
learn_rate_holder = tf.placeholder(tf.float32)

de_results = TimeNet(img_input_holder, chl=8, num=32, reuse=False)

loss_l1 = (LossL1(de_results[:, :, :, 0], img_label_holder[:, :, :, 0]) +
           LossL1(de_results[:, :, :, 1], img_label_holder[:, :, :, 1])) / 2
ssim_metric = (tf_ssim(tf.expand_dims(de_results[:, :, :, 0], -1), tf.expand_dims(img_label_holder[:, :, :, 0], -1)) +
               tf_ssim(tf.expand_dims(de_results[:, :, :, 1], -1), tf.expand_dims(img_label_holder[:, :, :, 1], -1))) / 2
loss_ssim = 100 * (1 - ssim_metric)
loss_diff = LossL1(de_results[:, :, :, 0] - de_results[:, :, :, 1], img_label_holder[:, :, :, 0] - img_label_holder[:, :, :, 1])

loss = loss_l1 + loss_ssim + loss_diff

with tf.name_scope("Train"):

    img_label_scalar = tf.summary.image('img/label_img', tf.maximum(cat_dedata(img_label_holder), 0), 1)
    img_result_scalar = tf.summary.image('img/result_img', tf.maximum(cat_dedata(de_results), 0), 1)
    img_residual_scalar = tf.summary.image('img/residual_img', cat_dedata(img_label_holder)-cat_dedata(de_results), 1)

    merge_train = tf.summary.merge([img_label_scalar, img_result_scalar, img_residual_scalar])

with tf.name_scope("Test"):

    train_l1_holder = tf.placeholder(tf.float32)
    train_ssim_holder = tf.placeholder(tf.float32)
    train_l1_scalar = tf.summary.scalar('loss/train_l1', train_l1_holder)
    train_ssim_scalar = tf.summary.scalar('loss/train_ssim', train_ssim_holder)

    valid_l1_holder = tf.placeholder(tf.float32)
    valid_ssim_holder = tf.placeholder(tf.float32)
    valid_l1_scalar = tf.summary.scalar('loss/valid_l1', valid_l1_holder)
    valid_ssim_scalar = tf.summary.scalar('loss/valid_ssim', valid_ssim_holder)


    valid_img_input_holder = tf.placeholder(tf.float32, [1, Pixel*2, Pixel, 1])
    valid_img_output_holder = tf.placeholder(tf.float32, [1, Pixel*2, Pixel, 1])
    valid_img_label_holder = tf.placeholder(tf.float32, [1, Pixel*2, Pixel, 1])
    valid_img_input_scalar =  tf.summary.image('img/input', tf.maximum(valid_img_input_holder, 0), 1)
    valid_img_output_scalar =  tf.summary.image('img/output', tf.maximum(valid_img_output_holder, 0), 1)
    valid_img_label_scalar =  tf.summary.image('img/label', tf.maximum(valid_img_label_holder, 0), 1)
    valid_img_error_scalar =  tf.summary.image('img/error', valid_img_output_holder - valid_img_label_holder, 1)

    merge_test = tf.summary.merge([train_l1_scalar, train_ssim_scalar, valid_l1_scalar, valid_ssim_scalar, valid_img_input_scalar, valid_img_output_scalar, valid_img_label_scalar, valid_img_error_scalar])

optimizer = tf.train.AdamOptimizer(learn_rate_holder)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=NumEpoch)


method = './logs/timenetv2a105/'
check_dir = method + 'checkpoints/'

print("Initialing Network:success \nTraining....")
'''训练'''

writer = tf.summary.FileWriter(method + 'logdir/', sess.graph)

filenames = ['./TimeTFR/train_dect_timenet_a105_p8000.tfrecords']
img_input_batch, img_label_batch = read_tfr_img_dect.input_pipeline_dect(filenames, batch_size=BatchSize, num_epochs=NumEpoch,
                                              num_features_patch=[Pixel, Pixel, Channel],
                                              num_features_img=[Pixel, Pixel, Channel])

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

else:  # 判断模型是否存在
    chkpt_fname = tf.train.latest_checkpoint(check_dir)
    saver.restore(sess, chkpt_fname)  # 存在就从模型中恢复变量
    sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

valid_input_holder = tf.placeholder(tf.float32, [1, Pixel, Pixel, 2])
valid_label_holder = tf.placeholder(tf.float32, [1, Pixel, Pixel, 2])
valid_output = TimeNet(valid_input_holder, chl=8, num=32, reuse=True)

valid_loss_l1 = (LossL1(valid_output[:, :, :, 0], valid_label_holder[:, :, :, 0]) + LossL1(valid_output[:, :, :, 1], valid_label_holder[:, :, :, 1])) / 2
valid_ssim = (tf_ssim(tf.expand_dims(valid_output[:, :, :, 0], -1), tf.expand_dims(valid_label_holder[:, :, :, 0], -1)) +
              tf_ssim(tf.expand_dims(valid_output[:, :, :, 1], -1), tf.expand_dims(valid_label_holder[:, :, :, 1], -1))) / 2

try:
    for epoch in range(0, NumEpoch):

        if epoch < 100:
            curr_lr = 1e-4 * (1.0 - epoch * 0.01)
        else:
            curr_lr = 1e-6

        train_avg_l1 = 0
        train_avg_ssim = 0

        for iter in range(TrainIter):
            samp_img_input, samp_img_label = sess.run([img_input_batch, img_label_batch])
            lossl1_, ssim_, _, trainmerge_ = \
                sess.run([loss_l1, ssim_metric, train, merge_train],
                         feed_dict={img_input_holder: samp_img_input,
                                    img_label_holder: samp_img_label,
                                    learn_rate_holder: curr_lr})

            train_avg_l1 += lossl1_
            train_avg_ssim += ssim_

            if ((iter+1) % TrainIter == 0):
                print('Epoch: {0}, Iter: {1}, lossl1: {2}'.format(epoch+1, iter+1, lossl1_))
                writer.add_summary(trainmerge_, epoch * TrainIter + iter + 1)

        train_avg_l1 = train_avg_l1 / TrainIter
        train_avg_ssim = train_avg_ssim / TrainIter

        saver.save(sess, os.path.join(check_dir, method[7:-1]), global_step=epoch+1)



        valid_avg_l1 = 0
        valid_avg_ssim = 0

        valid_list = ['A', 'B']
        valid_num = 0

        for patient in valid_list:

            test_num, test_data = read_raw_data_all('./DataTIME/DECT/limited_ct_105/' + patient + '/', 2*Pixel, Pixel, 0, -15)
            test_data = test_data + 1024
            test_num = np.size(test_data, 0)

            _, test_label = read_raw_data_all('./DataTIME/DECT/full_ct/' + patient + '/', 2*Pixel, Pixel, 0, -12)
            test_label = test_label + 1024

            valid_num += test_num

            for index_test in range(test_num):

                net_out, mae_, ssim_ = sess.run([valid_output, valid_loss_l1, valid_ssim], feed_dict={valid_input_holder: np.reshape(np.transpose(np.reshape(test_data[index_test, :, :], [2, Pixel, Pixel]), [1, 2, 0]), [1, Pixel, Pixel, 2]),
                                                                                                      valid_label_holder: np.reshape(np.transpose(np.reshape(test_label[index_test, :, :], [2, Pixel, Pixel]), [1, 2, 0]), [1, Pixel, Pixel, 2])})
                
                net_out = np.reshape(np.concatenate([net_out[:, :, :, 0], net_out[:, :, :, 1]], axis=1), [1, Pixel*2, Pixel, 1])

                valid_avg_ssim += ssim_
                valid_avg_l1 += mae_

        valid_avg_l1 = valid_avg_l1 / valid_num
        valid_avg_ssim = valid_avg_ssim / valid_num

        testmerge_ = sess.run(merge_test, feed_dict={train_l1_holder: train_avg_l1, train_ssim_holder: train_avg_ssim,
                                                     valid_l1_holder: valid_avg_l1, valid_ssim_holder: valid_avg_ssim,
                                                     valid_img_input_holder: np.reshape(test_data[-1, :, :], [1, 2*Pixel, Pixel, 1]),
                                                     valid_img_label_holder: np.reshape(test_label[-1, :, :], [1, 2*Pixel, Pixel, 1]),
                                                     valid_img_output_holder: np.reshape(net_out, [1, 2*Pixel, Pixel, 1]),
                                                     })

        writer.add_summary(testmerge_, epoch + 1)

except tf.errors.OutOfRangeError:

    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    coord.join(threads)
writer.close()