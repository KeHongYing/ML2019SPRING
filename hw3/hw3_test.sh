#! /usr/env/bin bash

wget https://www.dropbox.com/sh/yv4clpdzei8emz6/AACaSr7RbsgYVvOQLEsVp3Wea?dl=1 -O ./model
unzip model

python load_test.py $1
python predict_blend.py CNN_model_5.h5 CNN_model_6.h5 CNN_model_7.h5 CNN_model_8.h5 CNN_model_12.h5 CNN_model_14.h5 CNN_model_15.h5 CNN_model_16.h5 CNN_model_17.h5 CNN_model_18.h5 $2

