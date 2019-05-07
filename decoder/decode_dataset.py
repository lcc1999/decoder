# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:23:36 2019

@author: 11527
"""

import torch.utils.data as Data

class DecodeDataset(Data.Dataset):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def __len__(self):
        return len(self.b)
    def __getitem__(self, i):
        a = self.a[i]
        b = self.b[i]
        return a,b