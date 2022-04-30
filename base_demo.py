import tensorflow.compat.v1 as tf

def conv2d(input, filters, kernel_size, name, strides=(1, 1), paddings='same', dilation_rate=(1, 1)):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=paddings, dilation_rate=(1, 1),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.01),
                            name=name)

def conv2d_transpose(input, filters, kernel_size, name, paddings='same', strides=(2, 2)):
    return tf.layers.conv2d_transpose(input, filters, kernel_size, strides=strides, padding=paddings,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      bias_initializer=tf.constant_initializer(0.01), name=name)

def padding2D(input, size=(1, 1)):
    return tf.pad(input, ((0, 0), size, size, (0, 0)), "SYMMETRIC")


def lrelu(inputs):
    # return tf.maximum(inputs, 0.2 * inputs)
    return tf.nn.relu(inputs)

def GeLu(inputs):
    return 0.5 * inputs * (1 + tf.nn.tanh(inputs * 0.7978845608 * (1 + 0.044715 * inputs * inputs)))

def batch_norm(input_,scope='BN',bn_train=True):

    return tf.layers.batch_norm(input_,scale=True,epsilon=1e-8,
                                        is_training=bn_train,scope=scope)

def LayerNorm(input, name="instance_norm_3d"):
    with tf.variable_scope(name):
        depth = input.get_shape()[-1]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def res_block(input, num, kernel, name):
    with tf.variable_scope(name):
        c1 = conv2d(input, num, kernel, 'c1', (1, 1))
        l1 = lrelu(c1)
        c2 = conv2d(l1, num, kernel, 'c2', (1, 1))
        l2 = lrelu(c2)
        return input + l2


def res_block_v2(input1, input2, num, kernel, name):
    with tf.variable_scope(name):
        c1 = conv2d(input1, num, kernel, 'c1', (1, 1))
        l1 = lrelu(c1)
        c2 = conv2d(l1, num, kernel, 'c2', (1, 1))
        l2 = lrelu(c2)
        return input2 + l2


def conv_block(input, num, kernel, name, stride=(1, 1)):
    with tf.variable_scope(name):
        c1 = conv2d(input, num, kernel, 'c1', stride)
        l1 = lrelu(c1)
        return l1

def deconv_block(input, num, kernel, name):
    with tf.variable_scope(name):
        c1 = conv2d_transpose(input, num, kernel, 'c1')
        return c1
        
def batch_matmul(tensor1, tensor2):

    tensor1_temp = tf.reshape(tensor1, [-1, tensor1.shape[-1]])
    out = tf.reshape(tf.matmul(tensor1_temp, tensor2), [-1, tensor1.shape[1], tensor2.shape[-1]])
    return out

def get_split(InputTemp, num, split_dim, concat_dim):

    OutTemp = tf.split(InputTemp, num, axis=split_dim)
    splitTensor = OutTemp[0]
    for i in range(len(OutTemp) - 1):
        splitTensor = tf.concat([splitTensor, OutTemp[i+1]], 1)
    return splitTensor
        
def extract_patches(InputTemp, RowNum, ColNum):
    batch, rows, cols, channels = InputTemp.shape
    InputTempExpandDim = tf.reshape(InputTemp, [batch, 1, rows, cols, channels])
    
    TempWeight = get_split(InputTempExpandDim, num=RowNum, split_dim=2, concat_dim=0)
    TempHeight = get_split(TempWeight, num=ColNum, split_dim=3, concat_dim=0)

    return TempHeight

def rebin_patches(InputTemp, NumTemp):

    TempWeight = get_split(InputTemp, num=NumTemp, split_dim=0, concat_dim=3)
    TempHeight = get_split(TempWeight, num=NumTemp, split_dim=0, concat_dim=2)

    return TempHeight