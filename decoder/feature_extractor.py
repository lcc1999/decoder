# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:00:02 2019

@author: 11527
"""

import torch.nn as nn

class view(nn.Module):
    def __init__(self,out_features=-1):
        super(view,self).__init__()
        self.in_features=None
        self.out_features=out_features
    def forward(self,x):
        self.in_features=tuple(x.size())
        if self.out_features==-1:
            return x.view(x.size(0),-1)
        return x.view(self.out_features)

class FeatureExtractor(nn.Module):
    def __init__(self,structure,cut):
        super(FeatureExtractor,self).__init__()
        self.cut=cut
        self.layers=[]
        for i,j in enumerate(structure):
            if "Sequential" in str(j):
                for k in j:
                    self.layers+=[k]
                self.layers+=[view()]
            else:
                self.layers+=[j]
        for i,j in enumerate(self.layers):
            self.add_module("layer"+str(i),j)
    def forward(self,x):
        feature=[]
        for i,j in enumerate(self.layers):
            x=j(x)
            if i in self.cut:
                feature+=[x]
        return feature
