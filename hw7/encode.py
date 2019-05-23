from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Reshape, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta, Adam
import numpy as np
import sys

input_img = Input(shape = (32, 32, 3))

encoded = Conv2D(16, (4, 4), padding = "same", activation = "relu")(input_img)
encoded = Conv2D(32, (4, 4), padding = "same", activation = "relu")(encoded)
encoded = MaxPooling2D((2, 2), padding = "same")(encoded)
encoded = Conv2D(64, (4, 4), padding = "same", activation = "relu")(encoded)
encoded = Conv2D(128, (4, 4), padding = "same", activation = "relu")(encoded)
encoded = MaxPooling2D((2, 2), padding = "same")(encoded)
encoded = Flatten()(encoded)
encoded = Dense(32, activation = "relu")(encoded)

decoded = Dense(8 * 8 * 128, activation = "relu")(encoded)
decoded = Reshape((8, 8, 128))(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(128, (4, 4), activation = "relu", padding = "same")(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(64, (4, 4), activation = "relu", padding = "same")(decoded)
decoded = Conv2D(32, (4, 4), activation = "relu", padding = "same")(decoded)
decoded = Conv2D(3, (4, 4), activation = "sigmoid", padding = "same")(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.summary()
#encoded_input = Input(shape = (32, 32, 3))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))

adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer = adam, loss = "mean_squared_error")
encoder.compile(optimizer = adam, loss = "mean_squared_error")

x_train = np.load(sys.argv[1])

#x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))

checkpoint = ModelCheckpoint("model_autoencoder.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
re = ReduceLROnPlateau(monitor = "loss", patience = 2, min_lr = 0.0001, factor = 0.2)
callbacks_list = [re, checkpoint]
autoencoder.fit(x_train, x_train, epochs = 40, batch_size = 128, validation_split = 0.1, callbacks = callbacks_list)

encoder.save("model_encoder_%s.h5"%(int(sys.argv[2])))
