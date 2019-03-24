import sys
import numpy as np

x_data_test = np.delete(np.genfromtxt(sys.argv[1], delimiter = ","), 0, 0)

weight = np.load("hw2_logistic_weight.npy")

with open(sys.argv[2], "w", encoding = "utf-8", newline = "\n") as out:
	out.write("id,label\n")
	for i in range(len(x_data_test)):
		out.write("%d,%d\n"%(i + 1, np.dot(weight, x_data_test[i]) > 0))
