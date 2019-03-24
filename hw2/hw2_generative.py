import sys
import numpy as np

def sigmoid(x):
        return 1 / (1 + np.exp(-x)) if x > -550 else 0.

def normalization(x, err):
	return x / err

def add_second(x):
	length = len(x)
	data = [[] for _ in range(length)]
	for i in range(length):
		for j in range(6):
			for k in range(6 - j):
				data[i].append(x[i][j] * x[i][j + k])
		for j in x[i]:
			data[i].append(j)
	return np.array(data)

def add_third(x):
	length = len(x)
	data = [[] for _ in range(length)]
	for i in range(length):
		for j in range(21, 27):
			for k in range(j, 27):
				for l in range(k, 27):
					data[i].append(x[i][j] * x[i][k] * x[i][l])
			for j in x[i]:
				data[i].append(j)
	return np.array(data)

x_data_test = np.delete(np.genfromtxt(sys.argv[1], delimiter = ","), 0, 0)

weight = np.load("hw2_generative_weight.npy")
bias = np.load("hw2_generative_bias.npy")
standard_err = np.load("hw2_generative_standard_error.npy")

x_data_test = normalization(add_third(add_second(x_data_test)), standard_err)

with open(sys.argv[2], "w", encoding = "utf-8", newline = "\n") as out:
	out.write("id,label\n")
	for i in range(len(x_data_test)):
		out.write("%d,%d\n"%(i + 1, sigmoid(np.dot(weight, x_data_test[i]) + bias) > 0.5))
