#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:37:39 2019

@author: dogaykamar
"""

import torch 
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data.sampler import Sampler

class SuperRes(nn.Module):
    def __init__(self, upscale_factor):
        super(SuperRes, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 32, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(32, (upscale_factor ** 2)*3, 3, stride = 1, padding = 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.conv1.weight, gain)
        init.xavier_normal_(self.conv2.weight, gain)
        init.xavier_normal_(self.conv3.weight, gain)
        init.xavier_normal_(self.conv4.weight)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
    
class SmallTestSampler(Sampler):
    def __init__(self):
        self.num_samples = 8
        self.examples = [0,500,1000,1500,2000,2500,3000,3500]

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return self.num_samples