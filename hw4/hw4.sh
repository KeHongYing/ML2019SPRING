#! /usr/env/bin bash

wget https://www.dropbox.com/s/ffszulsbxkikrbq/CNN_model_18.h5?dl=1 -O CNN_model_18.h5
python load_train.py $1
python plot.py CNN_model_18.h5 X_train.npy Y_train.npy X_lime_train.npy $2

