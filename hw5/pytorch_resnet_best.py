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
epsilon = 1

# loss criterion
criterion = nn.CrossEntropyLoss()
trans = transform.Compose([transform.ToTensor(), transform.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

result = []

#wtf_cnt = 0
#wtf = [6, 30, 50, 52, 60, 61, 66, 80, 92, 94, 102, 103, 110, 120, 121, 138, 142, 146, 153, 156, 165, 166, 170, 184, 185, 197, -1]

#atk_label = []

input_dir = os.path.join(sys.argv[1], "")
output_dir = os.path.join(sys.argv[2], "")

for index in range(200):
	print(index, end = "\r")
	img_path = input_dir + "%03d.png"%(index)
	img = image.load_img(img_path, target_size = (224, 224))

	raw = image.img_to_array(img)
	bond1 = np.clip(raw - 1, 0, 255)
	bond2 = np.clip(raw + 1, 0, 255)
	
	org = trans(img).unsqueeze(0).cuda()
	org.requires_grad = True
	zero_gradients(org)
	output = model(org)
	
	#outputs = nn.Softmax(dim = -1)(output).detach()
	label = np.argmax(output.data.cpu().numpy())

	target = torch.tensor([label]).cuda()
	loss = criterion(output, target)
	loss.backward()

	atk = org + epsilon * org.grad.sign_() / 255 / 0.225

	output_image = inverse_transform(atk)
	
	cnt = 1
	while np.argmax(model(trans(image.array_to_img(output_image)).unsqueeze(0).cuda()).data.cpu().numpy()) == label:
		#		if index != wtf[wtf_cnt]:
		#			atk -= epsilon * org.grad.sign_() / 255 / 0.225
		#		else:
		#			atk += epsilon * org.grad.sign_() / 255 / 0.225
		atk += epsilon * org.grad.sign_() / 255 / 0.225

		output_image = inverse_transform(atk)
		cnt += 1
	#if index == wtf[wtf_cnt]:
	#	wtf_cnt += 1

	#atk_label.append(np.argmax(model(trans(image.array_to_img(output_image)).unsqueeze(0).cuda()).data.cpu().numpy()))
	result.append(cnt)
	
	#image.array_to_img(output_image).save("./result/%03d.png"%index)
	plt.imsave(output_dir + "%03d.png"%index, output_image)

	#print(nn.Softmax(dim = -1)(model(atk))[0][label], nn.Softmax(dim = -1)(output)[0][label])

#print(result)
#np.save("atk_label", np.array(atk_label))
