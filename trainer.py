import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
from torchvision import datasets, utils
import torchvision.transforms as transforms

from model import StegNet

TRAIN_PATH = '/home/vkv/Downloads/tiny_imagenet/train'
TEST_PATH = 'home/vkv/Downloads/test'
MODEL_PATH = 'home/vkv/Downloads/DeepSteg/checkpoint'
epochs = 100

def denormalize(image, std, mean):
 	for t in range(3):
 		image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
 	return image

def steg_loss(S_prime, C_prime, S, C, beta):
	loss_cover = F.mse_loss(C_prime, C)
	loss_secret = F.mse_loss(S_prime, S)
	loss = loss_cover + B*loss_secret
	return loss, loss_cover, loss_secret

train_loader = torch.utils.DataLoader(
	datasets.ImageFolder(
		TRAIN_PATH,
		transforms.Compose([
			transforms.Scale(256),
			transforms.RandomCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)])), 
	batch_size=10, pin_memory=True, num_workers=1,
	shuffle=True, drop_last=True)

test_loader = torch.utils.DataLoader(
	datasets.ImageFolder(
		TEST_PATH,
		transforms.Compose([
			transforms.Scale(256),
			transforms.RandomCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std)])), 
	batch_size=5, pin_memory=True, num_workers=1,
	shuffle=True, drop_last=True)

model = StegNet()

def train(train_loader, beta, lr):
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	losses = []

	for epoch in range(epochs):
		model.Train()
		train_loss = []

		for i, data in enumerate(train_loader):

			images, _ = data

			covers = images[:len(images)//2]
			secrets = images[len(images)//2:]
			covers = Variable(covers, requires_grad=False)
			secrets = Variable(secrets, requires_grad=False)

			optimizer.zero_grad()
			hidden, output = model(secrets, covers)

			loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, beta)
			loss.backward()
			optimizer.step()

			train_loss.append(loss.data[0])
			losses.append(loss.data[0])

			torch.save(model.state_dict(), MODEL_PATH+'.pkl')
			avg_train_loss = np.mean(train_loss)
			print('Train Loss {1:.4f}, cover_error {2:.4f}, secret_error{3:.4f}'. format(loss.data[0], loss_cover.data[0], loss_secret.data[0]))
		print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
            epoch+1, epochs, avg_train_loss))

		return model, avg_train_loss, losses




