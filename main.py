#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:41:22 2019

@author: dogaykamar
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
from dataset import FaceDataset
from network import SuperRes, SmallTestSampler

import numpy as np
import time

BATCHSIZE = 32
MAX_EPOCHS = 250
start_epoch = 0
learning_rate = 1e-3
weightDecay = 0

history = {"train_loss": [], "val_loss": []}

device = torch.device("cuda:0")
image_file = 'datasets/celebahq/'

print("Start Data Load")

train_data = FaceDataset(image_file, upscale_factor = 3, mode = 'train')
val_data = FaceDataset(image_file, upscale_factor = 3, mode = 'val')

sampler = SmallTestSampler()
train_loader = DataLoader(train_data, batch_size = BATCHSIZE, shuffle=True, pin_memory = True)
val_loader = DataLoader(val_data, batch_size = BATCHSIZE, shuffle=True, pin_memory = True)

model = SuperRes(upscale_factor = 3)
model.to(device)
criteria = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
start = time.time()

print("Begin!")

for epoch in range(start_epoch, MAX_EPOCHS):
    avg_loss = 0
    
    print("Epoch: ",epoch+1,"/",MAX_EPOCHS)
    print("Training start")
    
    model.train()
    num_batch = len(train_loader)
    for i, batch in enumerate(train_loader):
        input_img = batch[0].cuda(device)
        target_img = batch[1].cuda(device)
        
        prediction = model(input_img)
        loss = criteria.forward(prediction, target_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("MiniBatch: ", i+1, "/", num_batch, "   Loss: ", loss.item(), end="\r")
        
        avg_loss = (avg_loss*i)+loss.item()
        avg_loss = avg_loss/(i+1)
        
        del input_img, target_img, prediction, loss
    
    time_since_start = time.time()-start
    print("\nTrain Loss: ", avg_loss, "  found in {:.0f}m {:.0f}s".format(time_since_start// 60, time_since_start % 60))
    history["train_loss"].append(avg_loss)
    
    
    avg_loss = 0
    
    print("Validation start")
    model.eval()
    num_batch = len(val_loader) 
    for i, batch in enumerate(val_loader):
        input_img = batch[0].cuda(device)
        target_img = batch[1].cuda(device)
        
        prediction = model(input_img)
        loss = criteria.forward(prediction, target_img)
        
        print("MiniBatch: ", i+1, "/", num_batch, "  Loss: ", loss.item(), end="\r")
        avg_loss = (avg_loss*i)+loss.item()
        avg_loss = avg_loss/(i+1)
        
        if (i == 0):
            save_image = torch.cat([target_img, prediction], dim=2).detach().cpu()
            utils.save_image(save_image, 'model_out.png')
            
        del input_img, target_img, prediction, loss
        
        
    time_since_start = time.time()-start    
    print("\nValidation Loss: ",avg_loss, "  found in {:.0f}m {:.0f}s".format(time_since_start// 60, time_since_start % 60))
    history["val_loss"].append(avg_loss)
    
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, "models/model_{}.pth".format(epoch+1))
    np.save("models/history.npy", history)   




