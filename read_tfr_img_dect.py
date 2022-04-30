import tensorflow.compat.v1 as tf

def read_and_decode_img(filename_queue, shape_patch, shape_img):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,

        features={
            'ldct': tf.FixedLenFeature([], tf.string),
            'hdct': tf.FixedLenFeature([], tf.string),
        })

    LDImg = tf.decode_raw(features['ldct'], tf.float32, little_endian=True)
    HDImg = tf.decode_raw(features['hdct'], tf.float32, little_endian=True)

    LDImg = tf.reshape(LDImg, [shape_patch[0], shape_patch[1], shape_patch[2]])
    HDImg = tf.reshape(HDImg, [shape_img[0], shape_img[1], shape_img[2]])

    return LDImg, HDImg


def input_pipeline_dect(filenames, batch_size, num_epochs=None, num_features_patch=None, num_features_img=None):
    '''num_features := width * height for 2D image'''
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    img_imput, img_label = read_and_decode_img(filename_queue, shape_patch=num_features_patch, shape_img=num_features_img)
    min_after_dequeue = 64
    capacity = min_after_dequeue + 10 * batch_size
    img_input_batch, img_label_batch = tf.train.shuffle_batch(
        [img_imput, img_label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=64, allow_smaller_final_batch=True)

    return img_input_batch, img_label_batch