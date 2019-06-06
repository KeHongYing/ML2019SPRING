from keras.models import load_model
import numpy as np
import sys

model = load_model(sys.argv[1])
np.save("model", np.array([i.astype("float16") for i in model.get_weights()]))
#np.save("model", model.get_weights())
