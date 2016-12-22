from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, ELU, MaxPooling2D
from keras.layers import Convolution2D, Lambda
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
from behavioralCloning.model_checkpoint import ModelCheckpointWithJson

import matplotlib.image as mpimg
import numpy as np
import pandas
import os
import json
import cv2

import matplotlib
import scipy.misc

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def show_image(img):
    plt.imshow(img)
    plt.show()


def load_image(imagepath, data_dir):
    imagepath = imagepath.replace(' ', '')
    return mpimg.imread(data_dir + imagepath)


def process_image(img, rows, cols):
    cropped = img[30:137, ]
    return scipy.misc.imresize(cropped, [rows, cols])


def generate_training_data(line_data, data_dir, rows, cols):
    # randomly select left, right or center. Adjust steering if left or right
    selected_image = np.random.randint(3)
    steering_adjustment = 0.0
    if (selected_image == 0):
        image_path = line_data['center'][0].strip()
    if (selected_image == 1):
        image_path = line_data['left'][0].strip()
        steering_adjustment = 0.25
    if (selected_image == 2):
        image_path = line_data['right'][0].strip()
        steering_adjustment = -0.25
    steering = line_data['steering'][0] + steering_adjustment
    image = load_image(image_path, data_dir)
    # crop and resize image
    image = process_image(image, rows, cols)
    image = np.array(image)
    # randomly flip image
    flip_it = np.random.randint(2)
    if flip_it == 1:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering


# bias selection of images so images with larger steering value are preferred
def biased_images(line_data, data_dir, rows, cols, threshold, probability):
    image, steering = generate_training_data(line_data, data_dir, rows, cols)
    prob = np.random.uniform()
    while (True):
        if (abs(steering) > threshold or prob > probability):
            return image, steering
        else:
            image, steering = generate_training_data(line_data, data_dir, rows, cols)
            prob = np.random.uniform()


# generate batches of images and steering values.
def generate_batch(data, data_dir, rows, cols, batch_size=32):
    batch_images = np.zeros((batch_size, rows, cols, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for current in range(batch_size):
            line_index = np.random.randint(len(data))
            line_data = data.iloc[[line_index]].reset_index()
            image, steering = biased_images(line_data, data_dir, rows, cols, 0.1, 0.8)
            batch_images[current] = image
            batch_steering[current] = steering
        yield batch_images, batch_steering


def create_model(img_rows, img_cols):
    model = Sequential()
    # normalize input to (-1, 1)
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(img_rows, img_cols, 3), name='Normalization'))
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_rows, img_cols), name='Conv1'))
    model.add(LeakyReLU(name="LeakyRelu1"))
    model.add(Convolution2D(32, 3, 3, name='Conv2'))
    model.add(LeakyReLU(name="LeakyRelu2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool1"))
    model.add(Dropout(0.5, name="Dropout_0.5_1"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', name='Conv3'))
    model.add(LeakyReLU(name="Leaky Relu 3"))
    model.add(Convolution2D(64, 3, 3, name='Conv4'))
    model.add(LeakyReLU(name="Leaky Relu 4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool2"))
    model.add(Dropout(0.5, name="Dropout_0.5_2"))

    model.add(Flatten(name="Flatten"))
    model.add(Dense(512, name="Dense512"))
    model.add(LeakyReLU(name="Leaky Relu 5"))
    model.add(Dropout(0.5, name="Dropout_0.5_3"))
    model.add(Dense(1, name="Output"))

    return model



data_dir = "data/"
data = pandas.read_csv(data_dir + "/driving_log.csv")

def split_training_data(data):
    validationIndexes = int(data.shape[0] / 10)
    #shuffle the dataframe
    _data = data.reindex(np.random.permutation(data.index))
    #return training and validation data
    return _data[validationIndexes:], _data[:validationIndexes]

training_data, validation_data = split_training_data(data)

img_rows = 32
img_cols = 64
model = create_model(img_rows, img_cols)
model.summary()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[])
nb_epoch = 50

checkpointer = ModelCheckpointWithJson(filepath="checkpoints/model-{epoch:02d}", verbose=0)

history = model.fit_generator(generate_batch(training_data, data_dir, img_rows, img_cols), samples_per_epoch=300*32,
                              validation_data=generate_batch(validation_data, data_dir, img_rows, img_cols), nb_val_samples=30*32,
                              nb_epoch=nb_epoch, verbose=1, callbacks=[checkpointer])


# losses = pandas.read_csv('losses.txt')
#
# plt.plot(losses['epoch'].values, losses['loss'].values, label='Training Loss')
# plt.plot(losses['epoch'].values, losses['validation_loss'].values, label='Validation Loss')
# plt.title('Training vs Validation Loss')
# plt.xlabel('Epochs')
# plt.legend(bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.)
# plt.show()

