import sys
import numpy as np

testing_data = np.loadtxt(sys.argv[1], delimiter = ",", encoding = "big5", dtype = np.str)

length = len(testing_data)

data = [[] for _ in range(18)]

for i in range(length):
	for j in testing_data[i]:
		try:
			data[i % 18].append(float(j))
		except:
			if j == "NR":
				data[i % 18].append(0)

data = np.array(data)

length = len(data[0]) // 9

x_data = np.array([data[0 : 18, 9 * i : 9 * (i + 1)].reshape((162,)) for i in range(length)])
x_data = np.insert(x_data, 162, 1, 1)

weight = np.load("hw1.npy")

with open(sys.argv[2], "w", encoding = "utf-8", newline = "\n") as out:
	out.write("id,value\n")
	l = len(x_data)
	for i in range(l):
		out.write("id_%d,%f\n"%(i, np.dot(weight, x_data[i])))
