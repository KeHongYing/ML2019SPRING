import numpy as np
from keras.preprocessing import image
import sys
import os

pic = []
test_data = np.delete(np.delete(np.genfromtxt(sys.argv[1], delimiter = ","), 0, 0), 0, 1)

dir_path = os.path.join(sys.argv[2], "")

for i in range(1, 40001):
	img_path = dir_path + "%06d.jpg"%i
	img = image.img_to_array(image.load_img(img_path, target_size=(32, 32, 3)))
	pic.append(img / 255)

np.save("test_data", test_data)
np.save("img_data", np.array(pic))
