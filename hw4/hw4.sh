#! /usr/env/bin bash

python load_train.py $1
python plot.py CNN_model_18.h5 X_train.npy Y_train.npy X_lime_train.npy $2

