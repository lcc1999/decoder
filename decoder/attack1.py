# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:59:47 2019

@author: 11527
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

EPOCH=10
BATCH_SIZE = 10
LR = 0.01

class Decoder(nn.Module):
#encoder_structure 
#for example       if      encoder=Encoder()
#then the parameter is     list(encoder._modules.values())
    def __init__(self,encoder_structure,cut):
        super(Decoder,self).__init__()
        layers=[]
        self.cut=cut
        for i,j in enumerate(encoder_structure):
            if "Sequential" in str(j):
                for k in j:
                    layers+=[k]
            else:
                layers+=[j]
        layers=layers[:cut+1]
        layers=layers[::-1]
        for i,j in enumerate(layers):
            if "Linear" in str(j):
                a=getattr(j,"in_features")
                b=getattr(j,"out_features")
                layers[i]=nn.Linear(in_features=b,out_features=a)
            if "Conv2d" in str(j):
                a=getattr(j,"in_channels")
                b=getattr(j,"out_channels")
                c=getattr(j,"kernel_size")
                d=getattr(j,"stride")
                e=getattr(j,"padding")
                layers[i]=nn.ConvTranspose2d(b,a,c,d,e)
            if "Pool2d" in str(j):
                a=getattr(j,"kernel_size")
                layers[i]=nn.Upsample(scale_factor=a)
        self.layers=layers
        for i,j in enumerate(self.layers):
            self.add_module("layer"+str(i),j)
    def forward(self,x):
        output=x
        for i in self.layers:
            output=i(output)
        return output
    def get_loss(self,train_dataset):
        train_loader = Data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        test_loader=Data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False)
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        loss=0.0
        l=len(train_loader)
        print(l)
        for epoch in range(EPOCH):
            print("epoch:"+str(epoch))
            for i, (x, y) in enumerate(train_loader):
                batch_x = Variable(x.cuda())
                batch_y = Variable(y.cuda())
                output = self.forward(batch_x)
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for i,(x,y) in enumerate(test_loader):
            if i==0:
                batch_x = Variable(x.cuda())
                batch_y = Variable(y.cuda())
                output = self.forward(batch_x)
                self.draw_picture(batch_y.cpu(), "i")
                self.draw_picture(output.cpu(), "r")
            else:
                break

        return float(loss)

    def draw_picture(self,batch,i_r):
        for i,j in enumerate(batch):
            npimg=(j/2+0.5).detach().numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.savefig("pic/"+str(i)+"_"+i_r+"_cut_"+str(self.cut)+".png")
            plt.close()