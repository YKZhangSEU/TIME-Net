# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from base_demo import *

def EmbededPatches(inputs, patch_size, dim=1024, emb_dropout=0.1, name='EmbededPatches'):
    with tf.variable_scope(name):
        batch, rows, cols, channels = inputs.shape
        row_num = rows // patch_size
        col_num = cols // patch_size
        splitTensor = extract_patches(inputs, row_num, col_num)
        _, num, _, _, _ = splitTensor.shape
        split_patches = tf.reshape(splitTensor, [batch, num, -1]) # [batch, num, patch_dim]
        _, _, patch_dim = split_patches.shape     
        
        weights = tf.get_variable("weights", [patch_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0))
        embeded_patches = batch_matmul(split_patches, weights) + bias # [batch, num, dim]

        pos_embedding = tf.get_variable("pos_embedding", [1, num, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        embeded_patches_pos = embeded_patches + pos_embedding
        
        embeded_patches_pos_drop = tf.nn.dropout(embeded_patches_pos, 1-emb_dropout)
        
        return embeded_patches_pos_drop # V2  [batch, num, dim] # V1 [batch, num+1, dim]

def Attention(inputs, dim, heads=8, dim_head=64, dropout=0., name ='attn'):
    with tf.variable_scope(name):
        b, n, _, h = *inputs.shape, heads
        inner_dim = dim_head * heads
        
        weights1 = tf.get_variable(name + 'weight1', [dim, inner_dim * 3], tf.float32, tf.random_normal_initializer(stddev=0.01))
        qkv = batch_matmul(inputs, weights1)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = tf.transpose(tf.reshape(q, [b, n, h, -1]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [b, n, h, -1]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [b, n, h, -1]), [0, 2, 1, 3]) # [b, h, n, d]

        scale = dim_head ** -0.5
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * scale
        
        attn = tf.nn.softmax(dots, axis=-1)
        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out_reshape = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [b, n, -1])
        
        if not (heads == 1 and dim_head == dim):
            weights2 = tf.get_variable(name + 'weight2', [inner_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
            bias2 = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0))
            out_final = batch_matmul(out_reshape, weights2) + bias2
            out_final_drop = tf.nn.dropout(out_final, 1-dropout)
            return out_final_drop

        else:
            return out_reshape    

def FeedForWard(inputs, dim, hidden_dim, dropout, name='feed'):
    with tf.variable_scope(name):
        weights1 = tf.get_variable("weights1", [dim, hidden_dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        bias1 = tf.get_variable("bias1", [hidden_dim], initializer=tf.constant_initializer(0.0)) 
        linear1 = batch_matmul(inputs, weights1) + bias1 
        gelu1 = GeLu(linear1)
        drop1 = tf.nn.dropout(gelu1, 1-dropout)
        
        weights2 = tf.get_variable("weights2", [hidden_dim, dim], tf.float32, tf.random_normal_initializer(stddev=0.01))
        bias2 = tf.get_variable("bias2", [dim], initializer=tf.constant_initializer(0.0)) 
        linear2 = batch_matmul(drop1, weights2) + bias2
        drop2 = tf.nn.dropout(linear2, 1-dropout)
        return drop2

def TransformerEncoder(inputs, dim, depth, heads, dim_head, mlp_dim, dropout, name='TransformerEncoder'):
    with tf.variable_scope(name):
        feed = inputs
        for d in range(depth):
            attn = Attention(LayerNorm(feed, 'LN_Attn'+str(d)), dim, heads, dim_head, dropout, 'attn'+str(d)) + feed
            feed = FeedForWard(LayerNorm(attn, 'LN_Feed'+str(d)), dim, mlp_dim, dropout, 'feed'+str(d)) + attn
            #attn = Attention(feed, dim, heads, dim_head, dropout, 'attn'+str(d)) + feed
            #feed = FeedForWard(attn, dim, mlp_dim, dropout, 'feed'+str(d)) + attn
        return feed

def TransformerDemo(inputs, name='transformer'):
    with tf.variable_scope(name):
        batch, rows, cols, channels = inputs.shape
        patch_size = 8  # patch size
        dim = 128   # latent dimension 256
        emb_dropout = 0.2 * 1.0
        depth = 3  # num of transformer
        heads = 8  # num of head
        dim_head = 64  # head dimension
        mlp_dim = 512  # mlp dimension 512
        dropout = 0.2 * 1.0  # dropout for MLP
        embeded_patches = EmbededPatches(inputs, patch_size, dim, emb_dropout, name='EmbededPatches')
        transformer = TransformerEncoder(embeded_patches, dim, depth, heads, dim_head, mlp_dim, dropout, name='TransformerEncoder')  # V2 [batch, num, dim]
        decoder = tf.reshape(transformer, [batch, rows // patch_size, cols // patch_size, dim])

        return decoder