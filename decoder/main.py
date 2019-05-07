# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:58:51 2019

@author: 11527
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from model import VGG

from feature_extractor import FeatureExtractor
from attack1 import Decoder
from decode_dataset import DecodeDataset


EPOCH=10
BATCH_SIZE = 10
LR = 0.01
cuda=True

if cuda:
    print('Using cuda')
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset = datasets.CIFAR10(root='./', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE,shuffle=True,**kwargs)


'''
vgg=VGG('VGG11')
vgg.cuda()
optimizer = torch.optim.Adam(vgg.parameters(), lr=LR)
loss_func = nn.NLLLoss()
for epoch in range(EPOCH):
    print(epoch)
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x.cuda())
        batch_y = Variable(y.cuda())
        output = vgg(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()'''

#attack1(vgg,train_loader,[6,13,20,27])
#return the loss of decoders in a list
def attack1(model,train_loader,cut):
    feature_extractor=FeatureExtractor(list(model._modules.values()),cut)
    feature_extractor.cuda()
    a=[np.array([])]*len(cut)
    b=np.array([])

    for i,(x,y) in enumerate(train_loader):
        if i<1000:
            batch_x = Variable(x.cuda())
            output = feature_extractor(batch_x)
            print(i)
            if b.size==0:
                b = batch_x.cpu().detach().numpy()
            else:
                b = np.vstack((b, batch_x.cpu().detach().numpy()))
            for k in range(len(cut)):
                if a[k].size == 0:
                    print(str(cut[k])+"cut_decoder dataset...")
                    a[k]=output[k].cpu().detach().numpy()
                else:
                    a[k]=np.vstack((a[k],output[k].cpu().detach().numpy()))
        else:
            break
    decoder=[Decoder(list(feature_extractor._modules.values()),i)for i in cut]
    loss=[]
    for i in range(len(cut)):
        decoder[i].cuda()
        dataset=DecodeDataset(a[i],b)
        loss+=[decoder[i].get_loss(dataset)]
    return loss