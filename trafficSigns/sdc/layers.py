from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

def conv2d(name, input, filter_length, nb_filter, strides=1, padding='SAME'):
    # accepts tensor wiht shape [number_of_images, width, length, channels]
    channels = input.get_shape()[3]._value
    with tf.variable_scope(name):
        w_shape = [filter_length, filter_length, channels, nb_filter]

        W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable("biases", nb_filter, initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding=padding)

        x = tf.nn.bias_add(x, b)
        number_of_params = (filter_length*filter_length*channels + 1)*nb_filter
        print_info(name, x.get_shape(), number_of_params)
        return x


def maxpool2d(name, x, ksize=2, padding='VALID'):
    out= tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1],
                          padding=padding, name=name)
    print_info(name, x.get_shape(), 0)
    return out


def dropout_layer(name, input, fraction):
    out = tf.nn.dropout(input, fraction, name=name)
    print_info(name, out.get_shape(), 0)
    return out


def flatten(name, input):
    with tf.variable_scope(name):
        # turns tensor with shape [batch_size, a, b, c, ...] into tensor with shape [-1, a*b*c*...]
        flattenedSize = np.prod(input.get_shape().as_list()[1:])
        out =  tf.reshape(input, [-1, flattenedSize])
        print_info(name, out.get_shape(), 0)
        return out


def dense(name, input, size):
    # fully connected layer
    input_size = input.get_shape()[1]._value
    with tf.variable_scope(name):
        w_shape = [input_size, size]
        W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable("biases", size, initializer=tf.constant_initializer(0.0))
        out =  tf.add(tf.matmul(input, W), b)
        number_of_params = (input_size+1)*size
        print_info(name, out.get_shape(), number_of_params)
        return out


def relu(name, input):
    # relu activation
    out =  tf.nn.relu(input, name= name)
    print_info(name, out.get_shape(), 0)
    return out


def print_info(name, shape, params):
    print("{:20s} Shape: {:20s} Params:{:8d}".format(name,str(shape), params))

def activation_summary(tensor):
    tensor_name = tensor.op.name
    tf.histogram_summary(tensor_name + '/activations', tensor)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))
