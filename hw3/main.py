import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, PReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
#import pickle

x_data = np.load("X_train.npy")
y_data = np.load("Y_train.npy")
x_val_data = np.load("X_val_train.npy")
y_val_data = np.load("Y_val_train.npy")

datagen = ImageDataGenerator(rotation_range = 10 * np.pi, horizontal_flip = True)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (4, 4), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (5, 5), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (4, 4), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = 2 * 2, padding = "same"))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 128, kernel_size = (5, 5), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (4, 4), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3, 3), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (5, 5), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (4, 4), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3, 3), input_shape = (48, 48, 1), kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5), padding = "same"))
model.add(PReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = 2 * 2, padding = "same"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 2048, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5)))
model.add(Dropout(0.5))
model.add(Dense(units = 1024, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5)))
model.add(Dropout(0.5))
model.add(Dense(units = 512, activation = "relu", kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5)))
model.add(Dropout(0.5))
model.add(Dense(units = 7, activation = "softmax", kernel_regularizer = regularizers.l1_l2(l1 = 3e-5, l2 = 5e-5)))
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("CNN_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#model.fit(x_data, y_data, batch_size = 100, epochs = 50, validation_split = 0.1)
model.fit_generator(datagen.flow(x_data, y_data, batch_size = 128), steps_per_epoch = len(x_data) / 128, epochs = 100, validation_data = (x_val_data, y_val_data), callbacks = callbacks_list)

#pickle.dump(history, open("normal_history", "wb"))
#loss = [i for i in history.history["val_loss"]]
#acc = [i for i in history.history["val_acc"]]

#np.save("normal_loss", loss)
#np.save("normal_acc", acc)

#model.save("CNN_model_backup.h5")
#model.save("CNN_model_%s.h5"%(sys.argv[1]))
