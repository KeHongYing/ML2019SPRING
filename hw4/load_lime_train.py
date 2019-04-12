import sys
import numpy as np
import random

y_data = []
x_data = []

training_data = np.loadtxt(sys.argv[1], delimiter = ",", dtype = np.str)

for i in range(1, len(training_data)):
	x_data.append(training_data[i][1].split(" "))
	for j in range(len(x_data[-1])):
		#x_data[-1][j] = [int(x_data[-1][j]) / 255 for _ in range(3)]
		x_data[-1][j] = [int(x_data[-1][j]) / 255 for _ in range(3)]
		
x_data = np.array(x_data)
data = []

for i in x_data:
	data.append(i.reshape(-1, 48, 48, 3))

x_data = np.array(data)
np.save("X_lime_train", x_data)
