from keras.models import load_model
import numpy as np
import sys

model = []
for i in range(1, len(sys.argv) - 1):
	model.append(load_model(sys.argv[i]))
	model[-1].compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
x_test = np.load("X_test.npy")
classes = 0
for i in model:
	classes += i.predict(x_test)

with open(sys.argv[-1], "w", newline = "\n") as out:
	out.write("id,label\n")
	for i in range(len(classes)):
		out.write("%d,%d\n"%(i, np.argmax(classes[i])))
