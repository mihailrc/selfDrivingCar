from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

def conv2d(scope, input, filter_length, nb_filter, strides=1, padding='SAME'):
    # accepts tensor wiht shape [number_of_images, width, length, channels]
    channels = input.get_shape()[3]
    with tf.variable_scope(scope):
        w_shape = [filter_length, filter_length, channels, nb_filter]

        W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable("biases", nb_filter, initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding=padding)

        return tf.nn.bias_add(x, b)


def maxpool2d(x, ksize=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1],
                          padding=padding, name=x.op.name + "-pool")


def dropout_layer(input, fraction):
    return tf.nn.dropout(input, fraction, name=input.op.name + '-dropout')


def flatten(input):
    scopeName = input.op.name + '-flatten'
    with tf.variable_scope(scopeName) as scope:
        # turns tensor with shape [batch_size, a, b, c, ...] into tensor with shape [-1, a*b*c*...]
        flattenedSize = np.prod(input.get_shape().as_list()[1:])
        return tf.reshape(input, [-1, flattenedSize])


def dense(scope, input, size):
    # fully connected layer
    input_size = input.get_shape()[1]
    with tf.variable_scope(scope):
        w_shape = [input_size, size]
        W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable("biases", size, initializer=tf.constant_initializer(0.0))
        return tf.add(tf.matmul(input, W), b)


def relu(input):
    # relu activation
    return tf.nn.relu(input, name=input.op.name + 'relu')


def activation_summary(tensor):
    tensor_name = tensor.op.name
    tf.histogram_summary(tensor_name + '/activations', tensor)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))
