#! /usr/env/bin bash

python load_train.py $1 $2 $4
python load_test.py $3 $4
python word2vector.py train_data.npy test_data.npy
python train.py train_data.npy word_data_2 category.npy train
