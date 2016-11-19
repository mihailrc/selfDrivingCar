from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle

from trafficSigns.sdc import layers
from trafficSigns.sdc import data_batching
import time

training_iters = 200000
learning_rate = 0.001
batch_size = 32
display_step = 10

img_size = 32
img_channels = 3
n_classes = 43


def one_hot_encoding(y, nb_classes):
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


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

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, img_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
conv_prob = tf.placeholder(tf.float32)
hidden_prob = tf.placeholder(tf.float32)

predictions = layers.cifar_net(x, n_classes, conv_prob, hidden_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

summary_op = tf.merge_all_summaries()

test_dataset = data_batching.DataSet(X_test, y_test)


def evaluate_model(dataset, logits, session):
    start_time = time.time()
    evals = np.ndarray(shape=(0, n_classes), dtype=np.float32)
    # data gets shuffled so it is important to set it before it gets shuffled
    labels = dataset.labels
    while test_dataset.epochs_completed == 0:
        batch_images, batch_labels = test_dataset.next_batch(32, shuffle_between_epochs=False)
        preds = session.run(logits, feed_dict={x: batch_images, y: batch_labels, conv_prob: 1.0, hidden_prob: 1.0})
        evals = np.append(evals, preds, axis=0)
    correct_pred_2 = np.equal(np.argmax(evals, 1), np.argmax(labels, 1))
    testing_accuracy = np.sum(correct_pred_2) / dataset.num_examples
    print(evals.shape)
    print("Testing accuracy {:.4f}%".format(testing_accuracy * 100))
    print("Calculated accuracy in {:.2f} seconds".format(time.time() - start_time))


training_epochs = 1


def train(session, training_dataset, optimizer, cost, summary_operation):
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    start_time = time.time()
    summary_writer = tf.train.SummaryWriter("/tmp/traffic_signs", session.graph)
    step = 1
    while training_dataset.epochs_completed < training_epochs:

        batch_x, batch_y = training_dataset.next_batch(batch_size, True)

        session.run(optimizer, feed_dict={x: batch_x, y: batch_y, conv_prob: 0.75, hidden_prob: 0.5})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = session.run([cost, accuracy], feed_dict={x: batch_x,
                                                                 y: batch_y,
                                                                 conv_prob: 1.0,
                                                                 hidden_prob: 1.0})
            summary_str = session.run(summary_operation, feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    conv_prob: 1.0,
                                                                    hidden_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    end_time = time.time()
    print("Training for {:d} epochs took {:.2f} seconds".format(training_epochs, end_time - start_time))


# Initializing the variables
init = tf.initialize_all_variables()
train_dataset = data_batching.DataSet(X_train, y_train)

with tf.Session() as sess:
    sess.run(init)
    train(sess, train_dataset, optimizer, cost, summary_op)
    evaluate_model(test_dataset, predictions, sess)
