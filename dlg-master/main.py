# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

import scipy.io as sio

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import cv2

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = transforms.ToPILImage()(self.imgs[idx])
        img = self.transform(img)
        return img, lab

def SVHN(svhn_path, shape_img = (32,32)):
    '''
    svhn数据集
    '''
    images_all = []
    labels_all = []
    f = os.path.join(svhn_path,"svhn")
    #train = sio.loadmat(f + "/train_32x32.mat")
    test = sio.loadmat(f + '/test_32x32.mat')

    #train_data = train['X']
    #train_y = train['y']
    test_data = test['X']
    test_y = test['y']

    #train_data = np.swapaxes(train_data, 0, 3)
    #train_data = np.swapaxes(train_data, 2, 3)
    #train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    images_all = np.swapaxes(test_data, 1, 2)
    images_all = np.array(images_all)

    #train_y = train_y.reshape(73257, )
    labels_all = np.array(labels_all)
    #train_y = np.array(train_y)
    labels_all= test_y.reshape(26032, )
    print(test_y.shape)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all,labels_all, transform=transform)
    return dst

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="5",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
#if torch.cuda.is_available():
    #device = "cuda"
print("Running on %s" % device)

#只做了CIFAR10/CIFAR100/SVHN三个数据集
#dst = datasets.CIFAR100("D:/files/datasets", download=True)
dst = datasets.CIFAR10("D:/files/datasets")
#dst = datasets.MNIST("D:/files/datasets",download=True)
#dst = SVHN("D:/files/datasets")

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init
net = LeNet().to(device)


torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label

#调整初始化方法三种：同类图片/单色RGB/随机像素
#use RGB image to init
#img = Image.new("RGB",(32,32),(255,0,0))
#tmp_tensor = tp(img)
#dummy_data = torch.squeeze(tmp_tensor).to(device)
#dummy_data = dummy_data.view(1,*dummy_data.size())
#dummy_data = Variable(dummy_data).requires_grad_(True)


#use same class to init
#idx_shuffle = np.random.permutation(len(dst))
#tem_idx = 0
#for i in range(len(idx_shuffle)):
#    if dst[idx_shuffle[i]][1] == gt_label:
#        tem_idx = idx_shuffle[i]
#        break
#dummy_data = tp(dst[tem_idx][0]).to(device)
#dummy_data = dummy_data.view(1, *dummy_data.size())
#dummy_data = dummy_data.requires_grad_(True)

#use random pil image
#dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

#use servial same class picture to init
idx_shuffle = np.random.permutation(len(dst))#16995 35748
tem1_idx = 0
tem2_idx = 0
for i in range(len(idx_shuffle)):
    if dst[idx_shuffle[i]][1] == gt_label:
        if tem1_idx == 0 and tem2_idx == 0:
            tem1_idx = idx_shuffle[i]
        elif tem1_idx != 0 and tem2_idx == 0:
            tem2_idx = idx_shuffle[i]
        else:
            break
dummy_data1= tp(dst[tem1_idx][0]).to(device)
dummy_data2= tp(dst[tem2_idx][0]).to(device)
print(tem1_idx,tem2_idx)
dummy_data = torch.mean(torch.stack((dummy_data1,dummy_data2)),dim = 0)
dummy_data = dummy_data.view(1, *dummy_data.size())
dummy_data = dummy_data.requires_grad_(True)

dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
#dummy_label = gt_onehot_label

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
history.append(tt(dummy_data[0].cpu()))
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 10 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
