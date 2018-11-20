# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import learn
import numpy as np
import os

growth_rate = 8
filter = 16
nb_block = 3
dropout_rate = 0.5


def conv_layer(input,filter,kernel,stride=1,layer_name="conv"):

    kernel_initializer= tf.contrib.layers.variance_scaling_initializer()
    bias_initialzer= tf.constant_initializer(value= 0.)

    activation_func= tf.nn.relu

    network = tf.layers.conv2d(inputs=input,
                               filters=filter,
                               kernel_size=kernel,
                               strides=stride,
                               padding='SAME',
                               data_format= 'channels_last',
                               activation= activation_func,
                               use_bias=True,
                               kernel_initializer= kernel_initializer,
                               bias_initializer= bias_initialzer,
                               name= layer_name)
    return network

def Batch_Normalization(x,training,scope):
    return tf.layers.batch_normalization(x,axis=3,training=training, name= scope)

def Drop_out(x,rate,training):
    return tf.layers.dropout(inputs=x,rate=rate,training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x,pool_size=[2,2],stride=2,padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Max_Pooling(x,pool_size=[3,3],stride=2,padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Concatenation(layers):
    return tf.concat(layers,axis=3)



def bottleneck_layer(x,scope,training):
    with tf.name_scope(scope):
        x = Batch_Normalization(x,training=training,scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x,filter=4*growth_rate,kernel=[1,1],layer_name=scope+'_conv1')
        x = Drop_out(x,rate=dropout_rate,training=training)

        x =Batch_Normalization(x,training=training,scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x,filter=growth_rate,kernel=[3,3],layer_name=scope+'_conv2')
        x = Drop_out(x,rate=dropout_rate,training=training)
        return x

def transition_layer(x,filters,scope,training):
    with tf.name_scope(scope):
        x = Batch_Normalization(x,training=training,scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x,filter = filters,kernel=[1,1],layer_name=scope+'_conv1')
        x = Drop_out(x,rate=dropout_rate,training=training)
        x = Average_pooling(x,pool_size=[2,2],stride=[2,1])
        return x

def dense_block(input_x,nb_layers,layer_name,training):
    with tf.name_scope(layer_name):
        layers_concat =list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x,scope=layer_name+'_bottleN_'+str(0),training=training)

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x,scope=layer_name + '_bottleN_' + str(i+1),training=training)
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x

def dense_net(input_x, widths, training):

    with tf.variable_scope('densenet'):
        # training = (mode == learn.ModeKeys.TRAIN)
        # input_x:[ 32 ,width , 3 ]
        x = conv_layer(input_x,filter=filter,kernel=[3,3],stride=1,layer_name='conv0')
        # x = Max_Pooling(x,pool_size=[3,3],stride=2)
        # x: [32,width,16]
        x = dense_block(input_x = x,nb_layers=4,layer_name='dense_1',training=training)
        # x: [32,width,16+4*8=48]
        x = transition_layer(x, 32,scope='trans_1',training=training)#transition_layer(x,filters,scope,training)
        # x: [16,width-1,32]
        x = dense_block(input_x = x,nb_layers=6,layer_name='dense_2',training=training)
        # x: [16,width,32+6*8=80]
        x = transition_layer(x, 64,scope='trans_2',training=training)
        # x: [8,width-1,64]
        x = Max_Pooling(x,[2,2],2)
        # x:[4,width/2,64]
        x = dense_block(input_x =x ,nb_layers=8,layer_name='dense_3',training=training)
        # x: [4,width,64+8*8=112]
        x = transition_layer(x, 96,scope='trans_3',training=training)
        # x: [2,width-1,96]

        x = Batch_Normalization(x,training=training,scope='linear_batch')
        x = Relu(x)
        # x = Global_Average_Pooling(x)  # cifar-10中用于分类
        x = Max_Pooling(x,[2,2],[2,1])
        # x: [1, width - 1,96]

        features = tf.squeeze(x,axis=1,name='features')
        # calculate resulting sequence length

        sequence_length= _get_sequence_length(widths)

    return features,sequence_length


def _get_sequence_length(widths):
    one = tf.constant(1, dtype=tf.int32, name='one')
    two = tf.constant(2, dtype=tf.int32, name='two')

    after_conv0=widths
    after_dense_1=after_conv0
    after_trans_1=tf.subtract(after_dense_1,one)
    after_dense_2=after_trans_1
    after_trans_2=tf.subtract(after_dense_2,one)
    after_first_maxpool=tf.floor_div(after_trans_2, two )#向下取整
    after_dense_3=after_first_maxpool
    after_trans_3=tf.subtract(after_dense_3,one)
    after_second_maxpool=tf.subtract(after_trans_3,one)
    sequence_length = tf.reshape(after_second_maxpool,[-1], name='seq_len')
    return sequence_length

if __name__ == '__main__':
    pass