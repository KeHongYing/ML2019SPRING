from keras import models
import numpy as np
import sys, os
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE

encoder = models.load_model(sys.argv[1])
img = encoder.predict(np.load(sys.argv[2]))
test_data = np.load(sys.argv[3])

tsne = TSNE(n_components = 2, n_jobs = 16, verbose = True)
img = tsne.fit_transform(img)
kmeans = KMeans(n_clusters = 2, n_init = 30, max_iter = 500).fit(img)

with open(sys.argv[4], "w") as output:
	output.write("id,label\n")
	idx = 0
	for i in test_data:
		output.write("%d,%d\n"%(idx, kmeans.labels_[int(i[0]) - 1] == kmeans.labels_[int(i[1]) - 1]))
		idx += 1
