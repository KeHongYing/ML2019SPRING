#! /usr/env/bin bash

python load_data.py $2 $1
python k-means_sklearn.py model_encoder_17.h5 img_data.npy test_data.npy $3
