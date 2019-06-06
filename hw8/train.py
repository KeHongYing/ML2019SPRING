import numpy as np
import sys, os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, LeakyReLU, PReLU, Reshape
from keras.optimizers import Adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle

x_data = np.load("X_train.npy")
y_data = np.load("Y_train.npy")
x_val_data = np.load("X_val_train.npy")
y_val_data = np.load("Y_val_train.npy")

datagen = ImageDataGenerator(rotation_range = np.exp(1) * np.pi, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5, 5), input_shape = (48, 48, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (5, 5), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (1, 1), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (5, 5), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (1, 1), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 48, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 48, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 48, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (1, 1),  padding = "same"))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(units = 7, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(datagen.flow(x_data, y_data, batch_size = 128), steps_per_epoch = len(x_data) / 128, epochs = 300, validation_data = (x_val_data, y_val_data), callbacks = callbacks_list)

