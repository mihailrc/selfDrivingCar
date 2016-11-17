from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle

from trafficSigns.sdc import image_processing as ip
from trafficSigns.sdc import layers

from tensorflow.examples.tutorials.mnist import input_data

# training_iters = 1225*20
# batch_size = 32
training_iters = 10000
learning_rate = 0.001
batch_size = 32
display_step = 10


# img_size = 28
# img_channels = 1
# n_classes = 10
#
# def get_mnist_test_data():
#     mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#     X_test=mnist.test.images
#     X_test = X_test.reshape(X_test.shape[0], img_size,img_size,img_channels)
#     y_test = mnist.test.labels
#     return X_test, y_test
#
# def get_mnist_train_data():
#     mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#     X_train = mnist.train.images
#     X_train = X_train.reshape(X_train.shape[0], img_size,img_size,img_channels)
#     y_train = mnist.train.labels
#     return X_train, y_train
#
# X_train, y_train = get_mnist_train_data()
# X_test, y_test = get_mnist_test_data()


from keras.utils import np_utils

img_size = 32
img_channels = 3
n_classes = 43


def get_cifar_data(filename):
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
        X, y = data['features'], data['labels']
        X = X.astype('float32')
        X /= 255
        Y = np_utils.to_categorical(y, n_classes)
        return X, Y


data_dir = "../data/"
training_file = data_dir + 'train.p'
testing_file = data_dir + 'test.p'

X_train, y_train = get_cifar_data(training_file)
X_test, y_test = get_cifar_data(testing_file)
#



print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
conv_prob = tf.placeholder(tf.float32)
hidden_prob = tf.placeholder(tf.float32)


# def conv2d_withScope(scope, input, filter_length, nb_filter, strides=1, padding='SAME'):
#     # accepts tensor wiht shape [number_of_images, width, length, channels]
#     channels = input.get_shape()[3]
#     with tf.variable_scope(scope):
#         w_shape = [filter_length, filter_length, channels, nb_filter]
#
#         W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
#         b = tf.get_variable("biases", nb_filter, initializer=tf.constant_initializer(0.0))
#
#         x = tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding=padding)
#
#         return tf.nn.bias_add(x, b)
#
#
# def maxpool2d(x, ksize=2, padding='VALID'):
#     return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1],
#                           padding=padding, name=x.op.name + "-pool")
#
#
# def dropout_layer(input, fraction):
#     return tf.nn.dropout(input, fraction, name=input.op.name + '-dropout')
#
#
# def flatten(input):
#     scopeName = input.op.name + '-flatten'
#     with tf.variable_scope(scopeName) as scope:
#         # turns tensor with shape [batch_size, a, b, c, ...] into tensor with shape [-1, a*b*c*...]
#         flattenedSize = np.prod(input.get_shape().as_list()[1:])
#         return tf.reshape(input, [-1, flattenedSize])
#
#
# def dense(scope, input, size):
#     # fully connected layer
#     input_size = input.get_shape()[1]
#     with tf.variable_scope(scope):
#         w_shape = [input_size, size]
#         W = tf.get_variable("weights", w_shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
#         b = tf.get_variable("biases", size, initializer=tf.constant_initializer(0.0))
#         return tf.add(tf.matmul(input, W), b)
#
#
# def relu(input):
#     # relu activation
#     return tf.nn.relu(input, name=input.op.name + 'relu')
#
#
# def activation_summary(tensor):
#     tensor_name = tensor.op.name
#     tf.histogram_summary(tensor_name + '/activations', tensor)
#     tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))


# # Create model
# def conv_net(x, dropout_fraction):
#     # Convolution Layer
#     conv1 = layers.conv2d_withScope('conv-1', x, 5, 32)
#     layers.activation_summary(conv1)
#     conv1 = layers.relu(conv1)
#     print('conv1', conv1.get_shape())
#
#     conv1 = layers.maxpool2d(conv1, ksize=2)
#     conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#     print('conv1 - maxpool', conv1.get_shape())
#
#     conv2 = layers.conv2d_withScope('conv-2', conv1, 5, 64)
#     activation_summary(conv2)
#     conv2 = relu(conv2)
#     print('conv2', conv2.get_shape())
#     conv2 = maxpool2d(conv2, ksize=2)
#     conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#     print('conv2 - maxpool', conv2.get_shape())
#
#     # Reshape conv2 output to fit fully connected layer input
#     fc1 = flatten(conv2)
#     print('flattened', fc1.get_shape())
#     # Fully connected layer
#
#     fc1 = dense('hidden', fc1, 1024)
#     # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#     print('dense', fc1.get_shape())
#     activation_summary(fc1)
#     fc1 = relu(fc1)
#     # Apply Dropout
#     fc1 = dropout_layer(fc1, dropout_fraction)
#     # fc1 = tf.nn.dropout(fc1, dropout)
#     print('dense-2', fc1.get_shape())
#
#     # Output, class prediction
#     out = dense('output', fc1, n_classes)
#     activation_summary(out)
#     # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
#     print('out', out.get_shape())
#     return out


def cifar_net(x, conv_dropout, hidden_dropout):
    # Convolution Layer
    conv1 = layers.conv2d('conv-1', x, 3, 32)
    layers.activation_summary(conv1)
    conv1 = layers.relu(conv1)
    # conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    print('conv1', conv1.get_shape())
    conv2 = layers.conv2d('conv-2', conv1, 3, 32, padding='VALID')
    layers.activation_summary(conv2)
    conv2 = layers.relu(conv2)
    print('conv2', conv2.get_shape())
    conv2 = layers.maxpool2d(conv2, ksize=2)
    # conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    print('conv2 - maxpool', conv2.get_shape())
    conv2 = layers.dropout_layer(conv2, conv_dropout)
    print('conv2 - dropout', conv2.get_shape())

    conv3 = layers.conv2d('conv-3', conv2, 3, 64)
    layers.activation_summary(conv3)
    conv3 = layers.relu(conv3)
    # conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    print('conv3', conv3.get_shape())
    conv4 = layers.conv2d('conv-4', conv3, 3, 64, padding='VALID')
    layers.activation_summary(conv4)
    conv4 = layers.relu(conv4)
    print('conv4', conv4.get_shape())
    conv4 = layers.maxpool2d(conv4, ksize=2)
    # conv4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    print('conv4 - maxpool', conv4.get_shape())
    conv4 = layers.dropout_layer(conv4, conv_dropout)
    print('conv4 - dropout', conv4.get_shape())

    # Reshape conv2 output to fit fully connected layer input
    fc1 = layers.flatten(conv4)
    print('flattened', fc1.get_shape())
    # Fully connected layer

    d1 = layers.dense('dense1', fc1, 512)
    layers.activation_summary(d1)
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    d1 = layers.relu(d1)
    d1 = layers.dropout_layer(d1, hidden_dropout)
    print('dense', d1.get_shape())
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    # print('dense-1', fc1.get_shape())

    # Output, class prediction
    out = layers.dense('output', d1, n_classes)
    layers.activation_summary(out)
    print('out', out.get_shape())
    return out


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


from tensorflow.python.framework import dtypes

datagen = ip.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)

def generate_images(generator, images):
    generated_images = np.empty(np.append([0], images.shape[1:4]))
    for image in images:
        trsf = generator.random_transform(image.astype('float32'))
        # trsf = ip.rescale_array(trsf)
        # trsf/= 255
        generated_images = np.append(generated_images, [trsf], axis=0)

    return generated_images

class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.shuffle_data()

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            self.shuffle_data()
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        new_images = generate_images(datagen, self._images[start:end])
        return new_images, self._labels[start:end]

    def shuffle_data(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


# predictions = conv_net(x, keep_prob)
predictions = cifar_net(x, conv_prob, hidden_prob)

# Define loss and optimizer
# renomalize predictions
# predictions = -np.amax(predictions)

# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(predictions,1e-10,1.0)),
#                                               reduction_indices=[1]))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
summary_op = tf.merge_all_summaries()

init = tf.initialize_all_variables()

train_dataset = DataSet(X_train, y_train)


def shuffle_data(images, labels):
    perm = np.arange(labels.shape[0])
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    return images, labels


X_test, y_test = shuffle_data(X_test, y_test)

# eval_prediction = tf.nn.softmax(model(eval_data))
EVAL_BATCH_SIZE = 100


def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    evals = np.ndarray(shape=(size, n_classes), dtype=np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            evals[begin:end, :] = sess.run(
                predictions,
                feed_dict={x: X_test[begin:end],
                           y: y_test[begin:end],
                           keep_prob: 1.0,
                           conv_prob: 1.0,
                           hidden_prob: 1.0})
        else:
            batch_predictions = sess.run(
                predictions,
                feed_dict={x: X_test[begin:],
                           y: y_test[begin:],
                           keep_prob: 1.0,
                           conv_prob: 1.0,
                           hidden_prob: 1.0})
            evals[begin:, :] = batch_predictions[begin - size:, :]
    return evals

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter("/tmp/traffic_signs", sess.graph)

    step = 1

    while step * batch_size < training_iters:

        batch_x, batch_y = train_dataset.next_batch(batch_size)

        # batch_x = batch_x.reshape(batch_x.shape[0], 28,28,1)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: 0.75, conv_prob: 0.75, hidden_prob: 0.5})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, preds = sess.run([cost, accuracy, predictions], feed_dict={x: batch_x,
                                                                                  y: batch_y,
                                                                                  keep_prob: 1.0,
                                                                                  conv_prob: 1.0,
                                                                                  hidden_prob: 1.0})
            summary_str = sess.run(summary_op, feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 1.0,
                                                          conv_prob: 1.0,
                                                          hidden_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Finished Training")
    del train_dataset
#
    preds_2 = eval_in_batches(y_test, sess)
    correct_pred_2 = np.equal(np.argmax(preds_2, 1), np.argmax(y_test, 1))
    testing_accuracy = np.sum(correct_pred_2)/y_test.shape[0]
    print('Testing accuracy', testing_accuracy)

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #       sess.run(accuracy, feed_dict={x: X_test[:1000],
    #                                     y: y_test[:1000],
    #                                     keep_prob: 1.0,
    #                                     conv_prob: 1.0,
    #                                     hidden_prob: 1.0}))
