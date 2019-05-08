#! /usr/env/bin bash

python load_test.py $1 $2
python predict_blend.py $3
