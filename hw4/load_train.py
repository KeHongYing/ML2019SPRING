import sys
import numpy as np
import random

y_data = []
x_data = []
#x_lime_data = []

training_data = np.loadtxt(sys.argv[1], delimiter = ",", dtype = np.str)

pic = [28, 300, 6, 8, 71, 62, 12]

for i in pic:
	y_data.append(int(training_data[i][0]))
	x_data.append(training_data[i][1].split(" "))
	for j in range(len(x_data[-1])):
		#x_data[-1][j] = [int(x_data[-1][j]) / 255 for _ in range(3)]
		x_data[-1][j] = int(x_data[-1][j]) / 255
		
y_data = np.array(y_data)
x_data = np.array(x_data)
data = []

for i in x_data:
	data.append(i.reshape(-1, 48, 48, 1))

x_data = np.array(data)
x_lime_data = np.concatenate((x_data, x_data, x_data), axis = 4)
np.save("X_train", x_data)
np.save("Y_train", y_data)
np.save("X_lime_train", x_lime_data)
