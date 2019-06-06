import sys
import numpy as np

training_data = np.loadtxt(sys.argv[1], delimiter = ",", dtype = np.str)
x_data = []

for i in range(1, len(training_data)):
	x_data.append(training_data[i][1].split(" "))
	for j in range(len(x_data[-1])):
		x_data[-1][j] = int(x_data[-1][j]) / 255

x_data = np.array(x_data)
data = []

for i in x_data:
	data.append(i.reshape(48, 48, 1))

x_data = np.array(data)
np.save("X_test", x_data)
