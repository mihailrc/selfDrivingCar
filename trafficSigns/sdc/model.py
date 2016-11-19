from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import time

class Model():

    def __init__(self, x, y, n_classes, train_dict, test_dict, model_directory):
        self.x = x
        self.y = y
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.n_classes = n_classes
        self.model_directory = self._set_directory(model_directory)
        self.checkpoints_dir = self._set_directory(model_directory + "/checkpoints/")
        self.saver = tf.train.Saver()

    def _set_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def evaluate(self, dataset, logits, session):
        start_time = time.time()
        evals = np.ndarray(shape=(0, self.n_classes), dtype=np.float32)
        labels = dataset.labels
        while dataset.epochs_completed == 0:
            batch_images, batch_labels = dataset.next_batch(32, shuffle_between_epochs=False)
            preds = session.run(logits, feed_dict=self.merge_dictionaries({self.x: batch_images, self.y: batch_labels},
                                                                          self.test_dict))
            evals = np.append(evals, preds, axis=0)
        correct_pred_2 = np.equal(np.argmax(evals, 1), np.argmax(labels, 1))
        testing_accuracy = np.sum(correct_pred_2) / dataset.num_examples
        print("Testing accuracy {:.4f}%".format(testing_accuracy * 100))
        print("Calculated accuracy in {:.2f} seconds".format(time.time() - start_time))

    def train(self, session, training_dataset, optimizer, cost, accuracy, summary_op,
              training_epochs, batch_size=32, display_step=10, checkpoint_step=10):
        start_time = time.time()
        summary_writer = tf.train.SummaryWriter(self.model_directory, session.graph)
        step = 1
        while training_dataset.epochs_completed < training_epochs:

            batch_x, batch_y = training_dataset.next_batch(batch_size, True)

            session.run(optimizer,
                        feed_dict=self.merge_dictionaries({self.x: batch_x, self.y: batch_y}, self.train_dict))
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                feed_dict = self.merge_dictionaries({self.x: batch_x, self.y: batch_y}, self.test_dict)

                loss, acc = session.run([cost, accuracy], feed_dict=feed_dict)
                summary_str = session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            if (training_dataset.epochs_completed % checkpoint_step == 0 and training_dataset.is_first_batch):
                self.saver.save(session, self.checkpoints_dir + 'model', global_step=training_dataset.epochs_completed-1)
            step += 1
        end_time = time.time()
        print("Training for {:d} epochs took {:.2f} seconds".format(training_epochs, end_time - start_time))

    def merge_dictionaries(self, x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z
