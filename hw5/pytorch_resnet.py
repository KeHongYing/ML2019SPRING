import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from keras.preprocessing import image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import os
import sys

def inverse_transform(img):
	img = np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0))
	img[:, :, 0] = img[:, :, 0] * 0.229 + 0.485
	img[:, :, 1] = img[:, :, 1] * 0.224 + 0.456
	img[:, :, 2] = img[:, :, 2] * 0.225 + 0.406
	
	#img = image.array_to_img(np.clip(img * 255, 0, 255))
	img = np.round(np.clip(img * 255, 0, 255)).astype("uint8")
	return img

# using pretrain proxy model, ex. VGG16, VGG19...
model = resnet50(pretrained = True).cuda()
# use eval mode
model.eval()
epsilon = 5

# loss criterion
criterion = nn.CrossEntropyLoss()
trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

input_dir = os.path.join(sys.argv[1], "")
output_dir = os.path.join(sys.argv[2], "")

for index in range(200):
	print(index, end = "\r")
	img_path = input_dir + "%03d.png"%(index)
	img = image.load_img(img_path, target_size = (224, 224))

	raw = image.img_to_array(img)
	
	org = trans(img).unsqueeze(0).cuda()
	org.requires_grad = True
	zero_gradients(org)
	output = model(org)
	
	label = np.argmax(output.data.cpu().numpy())

	target = torch.tensor([label]).cuda()
	loss = criterion(output, target)
	loss.backward()

	atk = org + epsilon * org.grad.sign_() / 255 / 0.225

	output_image = inverse_transform(atk)
	
	plt.imsave(output_dir + "%03d.png"%index, output_image)
