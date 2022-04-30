# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from transformer_final import *
from functions import *
from base_demo import *

def TimeNet(input, chl=8, num=32, kernel=(3, 3), name='TimeNet', reuse=None):

    with tf.variable_scope(name, reuse=reuse):    
        
        c1 = conv_block(input, num, kernel, 'c1', (1, 1))
        res1 = res_block(c1, num, kernel, 'res1')

        d1 = conv2d(res1, 2 * num, kernel, 'd1', (2, 2))
        res2 = res_block(d1, 2 * num, kernel, 'res2')

        d2 = conv2d(res2, 4 * num, kernel, 'd2', (2, 2))
        res3 = res_block(d2, 4 * num, kernel, 'res3')

        d3 = conv2d(res3, 8 * num, kernel, 'd3', (2, 2))
        res4 = res_block(d3, 8 * num, kernel, 'res4')

        edgec1 = conv_block(tf.expand_dims(input[:, :, :, 0] + input[:, :, :, 1], 3) * 0.5, chl, kernel, 'edgec1', (1, 1))
        edgeres1 = res_block(edgec1, chl, kernel, 'edgeres1')

        edged1 = conv2d(edgeres1, 2 * chl, kernel, 'edged1', (2, 2))
        edgeres2 = res_block(edged1, 2 * chl, kernel, 'edgeres2')

        edged2 = conv2d(edgeres2, 4 * chl, kernel, 'edged2', (2, 2))
        edgeres3 = res_block(edged2, 4 * chl, kernel, 'edgeres3')

        edged3 = conv2d(edgeres3, 8 * chl, kernel, 'edged3', (2, 2))
        edgeres4 = res_block(edged3, 8 * chl, kernel, 'edgeres4')

        trans_maps = TransformerDemo(input, name='trans_maps')

        fusion = tf.concat([edgeres4, res4, trans_maps], axis=3)
        fusion_conv = conv2d(fusion, 8 * num, kernel, 'fusion_conv', (1, 1))

        leu3 = deconv_block(fusion_conv, 4 * num, kernel, 'leu3')
        lecat3 = tf.concat([res3, leu3, edgeres3], axis=3)

        leres5 = res_block_v2(lecat3, leu3, 4 * num, kernel, 'leres5')
        leu2 = deconv_block(leres5, 2 * num, kernel, 'leu2')
        lecat2 = tf.concat([res2, leu2, edgeres2], axis=3)

        leres6 = res_block_v2(lecat2, leu2, 2 * num, kernel, 'leres6')
        leu1 = deconv_block(leres6, 1 * num, kernel, 'leu1')
        lecat1 = tf.concat([res1, leu1, edgeres1], axis=3)

        leres7 = res_block_v2(lecat1, leu1, num, kernel, 'leres7')
        lec8 = conv2d(leres7, 1, kernel, 'lec8', (1, 1))

        heu3 = deconv_block(fusion_conv, 4 * num, kernel, 'heu3')
        hecat3 = tf.concat([res3, heu3, edgeres3], axis=3)

        heres5 = res_block_v2(hecat3, heu3, 4 * num, kernel, 'heres5')
        heu2 = deconv_block(heres5, 2 * num, kernel, 'heu2')
        hecat2 = tf.concat([res2, heu2, edgeres2], axis=3)

        heres6 = res_block_v2(hecat2, heu2, 2 * num, kernel, 'heres6')
        heu1 = deconv_block(heres6, 1 * num, kernel, 'heu1')
        hecat1 = tf.concat([res1, heu1, edgeres1], axis=3)

        heres7 = res_block_v2(hecat1, heu1, num, kernel, 'heres7')
        hec8 = conv2d(heres7, 1, kernel, 'hec8', (1, 1))

        c8 = tf.concat([lec8, hec8], axis=3)

        return lrelu(input + c8)