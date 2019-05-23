import os
import sys
import numpy as np 
from skimage.io import imread, imsave

def process(M): 
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	return M

IMAGE_PATH = sys.argv[1]
input_img = sys.argv[2]
reconstruct_img = sys.argv[3]

# Number of principal components used
k = 5

filelist = os.listdir(IMAGE_PATH)

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape

img_data = []
for filename in filelist:
	tmp = imread(os.path.join(IMAGE_PATH,filename))
	img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 
# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False)

#problem 1.c
# Load image & Normalize
picked_img = imread(os.path.join(IMAGE_PATH, input_img))
X = picked_img.flatten().astype('float32')
X -= mean

# Compression
#weight = np.array([.dot() for i in range(k)])

# Reconstruction
reconstruct = process(X.dot(v[:k].T).dot(v[:k]) + mean)
imsave(reconstruct_img, reconstruct.reshape(img_shape))

