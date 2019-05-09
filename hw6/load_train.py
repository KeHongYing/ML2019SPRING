import numpy as np
import sys
import jieba
import re

def has_num(s):
	for i in range(10):
		if str(i) in s:
			return True
	return False

def is_BX(s):
	if "B" in s:
		if has_num(s):
			return True
	return False

def remove(x):
	for i in range(len(x) - 1, -1, -1):
		if " " in x[i] or is_BX(x[i]):
			x.remove(x[i])
	return x


train_data = np.delete(np.delete(np.loadtxt(sys.argv[1], delimiter = ",", dtype = np.str), 0, 0), 0, 1)
label_data = np.delete(np.delete(np.genfromtxt(sys.argv[2], delimiter = ","), 0, 0), 0, 1)

jieba.set_dictionary(sys.argv[3])

data = []

for i in range(len(train_data)):
	data.append(remove(jieba.lcut(train_data[i][0], cut_all = False)))
	
np.save("train_data", np.array(data))
np.save("category", label_data.reshape(-1, ).astype("int"))
