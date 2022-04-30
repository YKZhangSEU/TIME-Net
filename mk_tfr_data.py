import os
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
from py_tools import *

GPUID = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPUID

root_dir = './DataTIME/DECT/full_ct/'
case_list = os.listdir(root_dir)
case_list.remove('A')
case_list.remove('B')
case_list.remove('C')
case_list.remove('D')

writer_file = tf.python_io.TFRecordWriter('./TimeTFR/train_dect_timenet_a90_p8000.tfrecords')

pixels = 512

total_num = 8000
count = 0
#
for case in case_list:

    case_name = case

    hdct_path = './DataTIME/DECT/full_ct/' + case_name + '/'
    slices, hdct_data = read_raw_data_all(hdct_path, 2*pixels, pixels, 0, -12)
    hdct_data = hdct_data + 1024

    ldct_path = './DataTIME/DECT/limited_ct_90/' + case_name + '/'
    _, ldct_data = read_raw_data_all(ldct_path, 2*pixels, pixels, 0, -15)
    ldct_data = ldct_data + 1024

    for index in range(slices):

        tf.reset_default_graph()

        hdct_img = np.transpose(np.reshape(hdct_data[index, :, :], [2, pixels, pixels]), [1, 2, 0])

        ldct_img = np.transpose(np.reshape(ldct_data[index, :, :], [2, pixels, pixels]), [1, 2, 0])

        # # plt.imshow(hdct_img[:, :, 1], cmap=plt.cm.gray)
        # # plt.show()

        hdct = hdct_img.tobytes()
        ldct = ldct_img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'hdct': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hdct])),
            'ldct': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ldct])),
        }))

        count += 1

        writer_file.write(example.SerializeToString())

        if count >= total_num:
            break
    
    print('Patient {0} finished, total number is {1}.'.format(case, count))

writer_file.close()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)