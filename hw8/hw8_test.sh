#! /usr/bin/bash

python load_test.py $1
python predict_10.py model.npy X_test.npy $2
