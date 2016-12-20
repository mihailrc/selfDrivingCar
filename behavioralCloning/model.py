from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, ELU, MaxPooling2D
from keras.layers import Convolution2D, Lambda
from keras.optimizers import Adam

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
    cropped = img[50:137, ]
    # cropped = img[55:, ]
    return scipy.misc.imresize(cropped, [rows, cols])


def generate_training_data(line_data, data_dir, rows, cols):
    # randomly select left, right or center. Adjust steering if left or right
    selected_image = np.random.randint(3)
    steering_adjustment = 0.0
    if (selected_image == 0):
        image_path = line_data['center'][0].strip()
    if (selected_image == 1):
        image_path = line_data['left'][0].strip()
        steering_adjustment = 0.2
    if (selected_image == 2):
        image_path = line_data['right'][0].strip()
        steering_adjustment = -0.2
    steering = line_data['steering'][0] + steering_adjustment
    image = load_image(image_path, data_dir)
    # crop and resize image
    image = process_image(image, rows, cols)
    image = np.array(image)
    #randomly flip image
    flip_it = np.random.randint(2)
    if flip_it == 1:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering

#bias selection of images so images with larger steering value are preferred
def biased_images(line_data, data_dir, rows, cols, threshold, probability):
    image, steering = generate_training_data(line_data, data_dir, rows, cols)
    prob = np.random.uniform()
    while (True):
        if (abs(steering) > threshold or prob > probability):
            return image, steering
        else:
            image, steering = generate_training_data(line_data, data_dir, rows, cols)
            prob = np.random.uniform()


#generate batches of images and steering values.
def generate_batch(data, data_dir, rows, cols, batch_size=32):
    batch_images = np.zeros((batch_size, rows, cols, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for current in range(batch_size):
            line_index = np.random.randint(len(data))
            line_data = data.iloc[[line_index]].reset_index()
            image, steering = biased_images(line_data, data_dir, rows, cols, 0.01, 0.8)
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

    model.add(Convolution2D(64, 3, 3, border_mode='same',name='Conv3'))
    model.add(LeakyReLU(name="Leaky Relu 3"))
    model.add(Convolution2D(64, 3, 3,name='Conv4'))
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

img_rows = 32
img_cols = 64
model = create_model(img_rows, img_cols)
model.summary()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[])

nb_epoch = 20

model_dir = "model_small/"
if os.path.exists(model_dir):
    os.rmdir(model_dir)

os.mkdir(model_dir)

for epoch in range(nb_epoch):
    #train the model
    history = model.fit_generator(generate_batch(data, data_dir, img_rows, img_cols),
                                  samples_per_epoch=10016, nb_epoch=1,
                                  verbose=1)
    #save after each epoch
    json_string = model.to_json()
    with open("{}model_{}.json".format(model_dir, epoch), 'w') as f:
        json.dump(json_string, f)
    model.save_weights("{}model_{}.h5".format(model_dir, epoch))
