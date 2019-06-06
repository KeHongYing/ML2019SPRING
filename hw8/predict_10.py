import numpy as np
import sys, os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, DepthwiseConv2D, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

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

weight = np.load(sys.argv[1], allow_pickle = True)
model.set_weights(weight)

test_data = np.load(sys.argv[2], allow_pickle = True)
classes = model.predict_classes(test_data)

with open(sys.argv[3], "w") as f:
	f.write("id,label\n")
	idx = 0
	for i in classes:
		f.write("%d,%d\n"%(idx, i))
		idx += 1
