from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
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
        self.saver = tf.train.Saver(max_to_keep=10)

    def _set_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def evaluate(self, dataset, logits, session, epoch=0):
        start_time = time.time()
        evals = np.ndarray(shape=(0, self.n_classes), dtype=np.float32)
        labels = dataset.labels
        while dataset.epochs_completed == epoch:
            batch_images, batch_labels = dataset.next_batch(32, generate_image=False, shuffle_between_epochs=False)
            preds = session.run(logits, feed_dict=self.merge_dictionaries({self.x: batch_images, self.y: batch_labels},
                                                                          self.test_dict))
            evals = np.append(evals, preds, axis=0)
        correct_pred_2 = np.equal(np.argmax(evals, 1), np.argmax(labels, 1))
        testing_accuracy = np.sum(correct_pred_2) / dataset.num_examples
        return np.argmax(evals,1),  np.argmax(labels, 1), testing_accuracy
        # print(type + " accuracy {:.4f}%".format(testing_accuracy * 100))
        # print("Calculated accuracy in {:.2f} seconds".format(time.time() - start_time))

    def train(self, session, training_dataset, validation_dataset, testing_dataset, logits, optimizer, cost, accuracy, summary_op,
              training_epochs, batch_size=32, display_step=50, checkpoint_step=10, generate_image=True):
        start_time = time.time()
        summary_writer = tf.train.SummaryWriter(self.model_directory, session.graph)
        step = 1

        while training_dataset.epochs_completed < training_epochs:

            batch_x, batch_y = training_dataset.next_batch(batch_size, generate_image=generate_image, shuffle_between_epochs=True)
            batches_per_epoch = math.ceil(training_dataset.num_examples/batch_size)
            session.run(optimizer,
                        feed_dict=self.merge_dictionaries({self.x: batch_x, self.y: batch_y}, self.train_dict))
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                feed_dict = self.merge_dictionaries({self.x: batch_x, self.y: batch_y}, self.test_dict)

                loss, acc = session.run([cost, accuracy], feed_dict=feed_dict)
                summary_str = session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                print("Epoch {:d}/{:d}".format(training_dataset.epochs_completed+1, training_epochs) +
                      ", Batch {:4d}/{:4d}".format(training_dataset.current_batch,batches_per_epoch) +
                      ", Minibatch Loss= {:.6f}".format(loss) +
                      ", Training Accuracy= {:.5f}".format(acc) +
                      ", Total Training time = {:4.2f}".format(time.time() - start_time) )

            if (training_dataset.epochs_completed % checkpoint_step == 0 and training_dataset.current_batch==0):
                self.saver.save(session, self.checkpoints_dir + 'model', global_step=training_dataset.epochs_completed)
            if (training_dataset.current_batch == 0):
                epoch = training_dataset.epochs_completed-1
                print('Summary for epoch {:d}'.format(epoch+1))
                _,_,val_acc = self.evaluate(validation_dataset, logits,session, epoch=epoch)
                print("Validation accuracy {:.4f}%".format(val_acc * 100))
                _,_,test_acc = self.evaluate(testing_dataset, logits,session, epoch=epoch)
                print("Test accuracy {:.4f}%".format(test_acc * 100))
                # print("Total Training time = {:4.2f}".format(time.time() - start_time))

            step += 1
        end_time = time.time()
        print("Trained {:d} epochs in {:.2f} seconds".format(training_epochs, end_time - start_time))

    def restore_model(self, path, session):
        self.saver.restore(sess=session, save_path=path)

    def merge_dictionaries(self, x, y):
        '''Given two dicts, merge them into a new dict as a shallow copy.'''
        z = x.copy()
        z.update(y)
        return z
