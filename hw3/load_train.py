import sys
import numpy as np
import random

y_data = []
x_data = []
x_val_data = []
y_val_data = []

training_data = np.loadtxt(sys.argv[1], delimiter = ",", dtype = np.str)

random_set = [i for i in range(len(training_data) - 1)]
random.shuffle(random_set)
random_set = random_set[0 : 2718]
#print("success")

for i in range(1, len(training_data)):
	label = int(training_data[i][0])
	if i in random_set:
		y_val_data.append([])
		for j in range(7):
			y_val_data[-1].append(label == j)
		x_val_data.append(training_data[i][1].split(" "))
		for j in range(len(x_val_data[-1])):
			x_val_data[-1][j] = int(x_val_data[-1][j]) / 255
	else:
		y_data.append([])
		for j in range(7):
			y_data[-1].append(label == j)
		x_data.append(training_data[i][1].split(" "))
		for j in range(len(x_data[-1])):
			x_data[-1][j] = int(x_data[-1][j]) / 255

y_data = np.array(y_data)
x_data = np.array(x_data)
y_val_data = np.array(y_val_data)
x_val_data = np.array(x_val_data)
data = []
val_data = []

for i in x_data:
	data.append(i.reshape(48, 48, 1))
for i in x_val_data:
	val_data.append(i.reshape(48, 48, 1))

x_data = np.array(data)
x_val_data = np.array(val_data)
np.save("X_train", x_data)
np.save("Y_train", y_data)
np.save("X_val_train", x_val_data)
np.save("Y_val_train", y_val_data)
