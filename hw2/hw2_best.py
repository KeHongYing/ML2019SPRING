import sys
import numpy as np

def normalization(x, err):
	return x / err

def add_second(x):
	length = len(x)
	data = [[] for _ in range(length)]
	for i in range(length):
		for j in range(6):
			for k in range(j, 6):
				data[i].append(sample[i][j] * sample[i][k])
		for j in x[i]:
			data[i].append(j)
	return np.array(data)

def add_third(x):
	length = len(x)
	data = [[] for _ in range(length)]
	for i in range(length):
		for j in range(6):
			for k in range(j, 6):
				for l in range(k, 6):
					data[i].append(sample[i][j] * sample[i][k] * sample[i][l])
		for j in x[i]:
			data[i].append(j)
	return np.array(data)

def add_forth(x):
	length = len(x)
	data = [[] for _ in range(length)]
	for i in range(length):
		for j in range(6):
			for k in range(j, 6):
				for l in range(k, 6):
					for m in range(l, 6):
						data[i].append(sample[i][j] * sample[i][k] * sample[i][l] * sample[i][m])
		for j in x[i]:
			data[i].append(j)
		data[i].append(1)
	return np.array(data)


x_data_test = np.delete(np.genfromtxt(sys.argv[1], delimiter = ","), 0, 0)
sample = x_data_test[:, 0:6]

weight = np.load("hw2_best_weight.npy")
standard_err = np.load("hw2_best_standard_error.npy")

x_data_test = normalization(add_forth(add_third(add_second(x_data_test))), standard_err)

with open(sys.argv[2], "w", encoding = "utf-8", newline = "\n") as out:
	out.write("id,label\n")
	for i in range(len(x_data_test)):
		out.write("%d,%d\n"%(i + 1, np.dot(weight, x_data_test[i]) > 0))
