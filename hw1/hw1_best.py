import sys
import numpy as np

def find_tail(head, data):
        tail = len(data)
        for i in range(head, tail):
                if float(data[i]) != -1:
                        return float(data[i])
        return 0.

testing_data = np.loadtxt(sys.argv[1], delimiter = ",", encoding = "big5", dtype = np.str)

length = len(testing_data)

data = [[] for _ in range(18)]

prev_pm = 16

divisor = np.sqrt(np.sqrt(np.pi * np.exp(1)))

for i in range(length):
	for j in range(len(testing_data[i])):
		try:
			data[i % 18].append(float(testing_data[i][j]) if float(testing_data[i][j]) != -1 else ((prev_pm + find_tail(j, testing_data[i])) / divisor))
			prev_pm = data[i % 18][-1]
		except:
			if testing_data[i][j] == "NR":
				data[i % 18].append(0)

data = np.array(data)

length = len(data[0]) // 9

x_data = np.array([data[0 : 18, 9 * i : 9 * (i + 1)].reshape((162,)) for i in range(length)])
x_data = np.insert(x_data, 162, 1, 1)

weight = np.load("hw1_best.npy")

with open(sys.argv[2], "w", encoding = "utf-8", newline = "\n") as out:
	out.write("id,value\n")
	l = len(x_data)
	for i in range(l):
		out.write("id_%d,%d\n"%(i, round(np.dot(weight, x_data[i]))))
