from __future__ import absolute_import
from __future__ import print_function

from trafficSigns.sdc import image_processing as ip
import numpy as np

datagen = ip.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)

def generate_images(generator, images):
    generated_images = np.empty(np.append([0], images.shape[1:4]))
    for image in images:
        trsf = generator.random_transform(image.astype('float32'))
        generated_images = np.append(generated_images, [trsf], axis=0)

    return generated_images

class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self.current_batch = 0
        self.num_examples = images.shape[0]
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

    # @property
    # def num_examples(self):
    #     return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, generate_image=False, shuffle_between_epochs=True):
        self.current_batch += 1
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        end = min(self._index_in_epoch, self.num_examples)
        imgs = self._images[start:end]
        lbls =  self._labels[start:end]
        if generate_image:
            imgs = generate_images(datagen, imgs)

        if self._index_in_epoch > self.num_examples:
            self.current_batch = 0
            self._epochs_completed += 1
            self._index_in_epoch = 0
            if shuffle_between_epochs:
                self.shuffle_data()

        return imgs,lbls

    def shuffle_data(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
