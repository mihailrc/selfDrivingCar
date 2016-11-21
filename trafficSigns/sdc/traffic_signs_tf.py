from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import os

from trafficSigns.sdc import layers
from trafficSigns.sdc import data_batching
from trafficSigns.sdc import model

training_iters = 200000
learning_rate = 0.001
batch_size = 32
display_step = 10

img_size = 32
img_channels = 3
n_classes = 43


def one_hot_encoding(original_labels, number_of_classes):
    encoded_labels = np.zeros((len(original_labels), number_of_classes))
    for i in range(len(original_labels)):
        encoded_labels[i, original_labels[i]] = 1.
    return encoded_labels


def get_cifar_data(filename):
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
        X, y = data['features'], data['labels']
        X = X.astype('float32')
        X /= 255
        Y = one_hot_encoding(y, n_classes)
        return X, Y


data_dir = "../data/"
training_file = data_dir + 'train.p'
testing_file = data_dir + 'test.p'

X_train, y_train = get_cifar_data(training_file)
X_test, y_test = get_cifar_data(testing_file)
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

# define the network
def convolutional_net(x, n_classes, conv_dropout, hidden_dropout):
    # Convolution with Relu
    conv1 = layers.conv2d('conv1', x, 3, 32)
    layers.activation_summary(conv1)
    conv1 = layers.relu('conv1-relu',conv1)

    # Convolution with Relu and Max Pool
    conv2 = layers.conv2d('conv2', conv1, 3, 32, padding='VALID')
    layers.activation_summary(conv2)
    conv2 = layers.relu('conv2-relu',conv2)
    conv2 = layers.maxpool2d('conv2-maxpool',conv2, ksize=2)
    conv2 = layers.dropout_layer('conv2-dropout',conv2, conv_dropout)

    # Convolution with Relu
    conv3 = layers.conv2d('conv-3', conv2, 3, 64)
    layers.activation_summary(conv3)
    conv3 = layers.relu('conv3-relu',conv3)

    # Convolution with Relu and Max Pool
    conv4 = layers.conv2d('conv-4', conv3, 3, 64, padding='VALID')
    layers.activation_summary(conv4)
    conv4 = layers.relu('conv4-relu',conv4)
    conv4 = layers.maxpool2d('conv4-maxpool',conv4, ksize=2)
    conv4 = layers.dropout_layer('conv4-dropout',conv4, conv_dropout)

    # Reshape conv output to fit fully connected layer input
    fc1 = layers.flatten('flatten',conv4)

    # Fully connected hidden layer
    d1 = layers.dense('hidden', fc1, 512)
    layers.activation_summary(d1)
    d1 = layers.relu('hidden-relu',d1)
    d1 = layers.dropout_layer('hidden-dropout',d1, hidden_dropout)

    # Fully connected output layer
    out = layers.dense('output', d1, n_classes)
    layers.activation_summary(out)
    return out

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
conv_prob = tf.placeholder(tf.float32)
hidden_prob = tf.placeholder(tf.float32)

predictions = convolutional_net(x, n_classes, conv_prob, hidden_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

summary_op = tf.merge_all_summaries()
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
test_dataset = data_batching.DataSet(X_test, y_test)
training_epochs = 2

# Initializing the variables
init = tf.initialize_all_variables()
train_dataset = data_batching.DataSet(X_train, y_train)

train_dict = {conv_prob: 0.75, hidden_prob: 0.5}
test_dict = {conv_prob: 1.0, hidden_prob: 1.0}
model = model.Model(x, y,n_classes,train_dict, test_dict, "/tmp/traffic_signs")
with tf.Session() as sess:
    sess.run(init)
    model.train(sess, train_dataset, optimizer, cost, accuracy, summary_op,training_epochs, generate_image=False, checkpoint_step=1)
    model.evaluate(test_dataset, predictions, sess)
