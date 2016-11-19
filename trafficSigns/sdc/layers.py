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
    
    
def convolutional_net(x, n_classes, conv_dropout, hidden_dropout):
    # Convolution Layer
    conv1 = conv2d('conv-1', x, 3, 32)
    activation_summary(conv1)
    conv1 = relu(conv1)
    # conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    print('conv1', conv1.get_shape())
    conv2 = conv2d('conv-2', conv1, 3, 32, padding='VALID')
    activation_summary(conv2)
    conv2 = relu(conv2)
    print('conv2', conv2.get_shape())
    conv2 = maxpool2d(conv2, ksize=2)
    # conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    print('conv2 - maxpool', conv2.get_shape())
    conv2 = dropout_layer(conv2, conv_dropout)
    print('conv2 - dropout', conv2.get_shape())

    conv3 = conv2d('conv-3', conv2, 3, 64)
    activation_summary(conv3)
    conv3 = relu(conv3)
    # conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    print('conv3', conv3.get_shape())
    conv4 = conv2d('conv-4', conv3, 3, 64, padding='VALID')
    activation_summary(conv4)
    conv4 = relu(conv4)
    print('conv4', conv4.get_shape())
    conv4 = maxpool2d(conv4, ksize=2)
    # conv4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    print('conv4 - maxpool', conv4.get_shape())
    conv4 = dropout_layer(conv4, conv_dropout)
    print('conv4 - dropout', conv4.get_shape())

    # Reshape conv2 output to fit fully connected layer input
    fc1 = flatten(conv4)
    print('flattened', fc1.get_shape())
    # Fully connected layer

    d1 = dense('dense1', fc1, 512)
    activation_summary(d1)
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    d1 = relu(d1)
    d1 = dropout_layer(d1, hidden_dropout)
    print('dense', d1.get_shape())
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    # print('dense-1', fc1.get_shape())

    # Output, class prediction
    out = dense('output', d1, n_classes)
    activation_summary(out)
    print('out', out.get_shape())
    return out


# def loss(logits, labels):
#     """Add L2Loss to all the trainable variables.
#
#     Add summary for "Loss" and "Loss/avg".
#     Args:
#       logits: Logits from inference().
#       labels: Labels from distorted_inputs or inputs(). 1-D tensor
#               of shape [batch_size]
#
#     Returns:
#       Loss tensor of type float.
#     """
#     # Calculate the average cross entropy loss across the batch.
#     labels = tf.cast(labels, tf.int64)
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits, labels, name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
#
#     # The total loss is defined as the cross entropy loss plus all of the weight
#     # decay terms (L2 loss).
#     return tf.add_n(tf.get_collection('losses'), name='total_loss')


# def _add_loss_summaries(total_loss):
#     """Add summaries for losses in CIFAR-10 model.
#
#     Generates moving average for all losses and associated summaries for
#     visualizing the performance of the network.
#
#     Args:
#       total_loss: Total loss from loss().
#     Returns:
#       loss_averages_op: op for generating moving averages of losses.
#     """
#     # Compute the moving average of all individual losses and the total loss.
#     loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
#     losses = tf.get_collection('losses')
#     loss_averages_op = loss_averages.apply(losses + [total_loss])
#
#     # Attach a scalar summary to all individual losses and the total loss; do the
#     # same for the averaged version of the losses.
#     for l in losses + [total_loss]:
#         # Name each loss as '(raw)' and name the moving average version of the loss
#         # as the original loss name.
#         tf.scalar_summary(l.op.name + ' (raw)', l)
#         tf.scalar_summary(l.op.name, loss_averages.average(l))
#
#     return loss_averages_op
