import os, itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
import sys
from skimage.segmentation import slic
from lime import lime_image

np.random.seed(0)

def predict(data):
	return model.predict(np.mean(data.reshape(-1, 48, 48, 3), axis = 3).reshape(-1, 48, 48, 1)).reshape(-1, 7)
def segmentation(data):
	return slic(data, n_segments = 100)

def plotLime(arrayX, output_path):
	cnt = 0
	for i in arrayX:
		explainer = lime_image.LimeImageExplainer()
		explaination = explainer.explain_instance(image = i.reshape(48, 48, 3), classifier_fn = predict, hide_color = None, segmentation_fn = segmentation, random_seed = np.exp(1) * 1000)
		image, mask = explaination.get_image_and_mask(label = cnt, positive_only = False, hide_rest = False, num_features = 5, min_weight = 0.0)
		#plt.imsave("img_%d"%(cnt), image)
		plt.imsave(os.path.join(output_path, "fig3_%d.jpg"%(cnt)), image)
		#os.rename(os.path.join(output_path, "fig3_%d.png"%(cnt)), os.path.join(output_path, "fig3_%d.jpg"%(cnt)))
		cnt += 1

def plotImageFiltersResult(arrayX, intChooseId, output_path):
	"""
	This function plot the output of convolution layer in valid data image.
	"""
	intImageHeight = 48
	intImageWidth = 48

	model = load_model(sys.argv[1])
	dictLayer = dict([layer.name, layer] for layer in model.layers)
	inputImage = model.input
	listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
	# define the function that input is an image and calculate the image through each layer until the output layer that we choose
	listCollectLayers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in listLayerNames]
	
	cnt = 2
	fn = listCollectLayers[cnt]
	#for cnt, fn in enumerate(listCollectLayers):
	arrayPhoto = arrayX[intChooseId].reshape(1, intImageWidth, intImageHeight, 1)
	listLayerImage = fn([arrayPhoto, 0]) # get the output of that layer list (1, 1, 48, 48, 64)
	
	fig = plt.figure(figsize=(16, 17))
	intFilters = 32
	for i in range(intFilters):
		ax = fig.add_subplot(intFilters/8, 8, i+1)
		ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.xlabel("filter {}".format(i))
		plt.tight_layout()
	fig.suptitle("Output of {} (Given image{})".format(listLayerNames[cnt], intChooseId))
	plt.savefig(os.path.join(output_path, "fig2_2"))
	os.rename(os.path.join(output_path, "fig2_2.png"), os.path.join(output_path, "fig2_2.jpg"))

def deprocessImage(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to array
	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	# print(x.shape)
	return x

def makeNormalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def trainGradAscent(intIterationSteps, arrayInputImageData, targetFunction, intRecordFrequent):
	"""
	Implement gradient ascent in targetFunction
	"""
	listFilterImages = []
	floatLearningRate = 1e-2
	for i in range(intIterationSteps):
		floatLossValue, arrayGradientsValue = targetFunction([arrayInputImageData, 0])
		arrayInputImageData += arrayGradientsValue * floatLearningRate
		if i % intRecordFrequent == 0:
			listFilterImages.append((arrayInputImageData, floatLossValue))
			#print("#{}, loss rate: {}".format(i, floatLossValue))
	return listFilterImages

def plotWhiteNoiseActivateFilters(output_path):
	"""
	This function plot Activate Filters with white noise as input images
	"""
	intRecordFrequent = 20
	intNumberSteps = 160
	intIterationSteps = 160

	dictLayer = dict([layer.name, layer] for layer in model.layers)
	inputImage = model.input
	listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
	listCollectLayers = [dictLayer[name].output for name in listLayerNames]

	filter_num = [32, 32, 32, 64, 64, 64, 128, 128]

	cnt = 2
	fn = listCollectLayers[cnt]
	#for cnt, fn in enumerate(listCollectLayers):
	listFilterImages = []
	intFilters = filter_num[cnt]
	for i in range(intFilters):
		arrayInputImage = np.random.random((1, 48, 48, 1)) # random noise
		tensorTarget = K.mean(fn[:, :, :, i])

		tensorGradients = makeNormalize(K.gradients(tensorTarget, inputImage)[0])
		targetFunction = K.function([inputImage, K.learning_phase()], [tensorTarget, tensorGradients])

		# activate filters
		listFilterImages.append(trainGradAscent(intIterationSteps, arrayInputImage, targetFunction, intRecordFrequent))

	for it in range(8):
		#print("In the #{}".format(it))
		fig = plt.figure(figsize=(16, 17))
		for i in range(intFilters):
			ax = fig.add_subplot(intFilters/8, 8, i+1)
			arrayRawImage = listFilterImages[i][it][0].squeeze()
			ax.imshow(deprocessImage(arrayRawImage), cmap="Blues")
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.xlabel("{:.3f}".format(listFilterImages[i][it][1]))
			plt.tight_layout()
	fig.suptitle("Filters of layer {} (# Ascent Epoch {} )".format(listLayerNames[cnt], it*intRecordFrequent))
	plt.savefig("fig2_1")
	plt.savefig(os.path.join(output_path, "fig2_1"))
	os.rename(os.path.join(output_path, "fig2_1.png"), os.path.join(output_path, "fig2_1.jpg"))

def plotSaliencyMap(arrayX, arrayYLabel, listClasses, output_path):

	inputImage = model.input

	listImageIDs = [i for i in range(7)]
	title = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
	for idx in listImageIDs:
		arrayProbability = model.predict(arrayX[idx])
		arrayPredictLabel = arrayProbability.argmax()
		tensorTarget = model.output[:, arrayPredictLabel]
		tensorGradients = K.gradients(tensorTarget, inputImage)[0]
		fn = K.function([inputImage, K.learning_phase()], [tensorGradients])

		### start heatmap processing ###
		arrayGradients = fn([arrayX[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
	   
		arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)

		# normalize center on 0., ensure std is 0.1
		arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
		arrayGradients *= 0.1

		# clip to [0, 1]
		arrayGradients += 0.5
		arrayGradients = np.clip(arrayGradients, 0, 1)

		arrayHeatMap = arrayGradients.reshape(48, 48)
		### End heatmap processing ###
		
		#print("ID: {}, Truth: {}, Prediction: {}".format(idx, arrayYLabel[idx], arrayPredictLabel))
		
		# show original image
		fig = plt.figure()
		'''
		ax = fig.add_subplot(1, 3, 1)
		axx = ax.imshow((arrayX[idx]*255).reshape(48, 48), cmap="gray")
		plt.tight_layout()
		# show Heat Map
		ax = fig.add_subplot(1, 3, 2)
		axx = ax.imshow(arrayHeatMap, cmap=plt.cm.jet)
		plt.colorbar(axx)
		plt.tight_layout()
		'''
		# show Saliency Map
		floatThreshold = 0.55
		arraySee = (arrayX[idx]*255).reshape(48, 48)
		arraySee[np.where(arrayHeatMap <= floatThreshold)] = np.mean(arraySee)
		#ax = fig.add_subplot(1, 3, 3)
		plt.imshow(arraySee, cmap="jet")
		plt.colorbar()
		plt.tight_layout()
		fig.suptitle(title[idx])
		plt.savefig(os.path.join(output_path, "fig1_%d"%(idx)))
		os.rename(os.path.join(output_path, "fig1_%d.png"%(idx)), os.path.join(output_path, "fig1_%d.jpg"%(idx)))

model = load_model(sys.argv[1])
X_data = np.load(sys.argv[2])
Y_data = np.load(sys.argv[3])
X_lime_data = np.load(sys.argv[4])
output_path = sys.argv[5]

plotSaliencyMap(X_data, Y_data, [str(i) for i in range(7)], output_path)
plotWhiteNoiseActivateFilters(output_path)
plotImageFiltersResult(X_data, 0, output_path)
plotLime(X_lime_data, output_path)
