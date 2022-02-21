import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
from UNetDown import *
from Dataloader import *

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, t):
        t1 = self.conv1(t)
        t2 = self.conv2(t1)
        t3 = self.down1(t2)
        t4 = self.down2(t3)
        t5 = self.down3(t4)
        t6 = self.down4(t5)
        t7 = self.up1(t6, t5)
        t8 = self.up2(t7, t4)
        t9 = self.up3(t8, t3)
        t10 = self.up4(t9, t2)
        t11 = self.conv3(t10)

        return t11

unet = UNet()
a = torch.empty((1,1,200,200))
preds = unet(a)
print(preds.shape)

#test = ImageData('/Users/liamdunn/Documents/University/Y4Project/Images/ISM_Train', '/Users/liamdunn/Documents/University/Y4Project/Images/Conf_Train', 50).load()
#labels, data = test
#labels = torch.tensor(labels)
#data = torch.FloatTensor(data)
#print(labels.shape)
#print(data.shape)
#preds = unet(data[:,0].unsqueeze(dim=0))
#print(preds.shape)

#hinge_loss = nn.HingeEmbeddingLoss()
#output = hinge_loss(F.normalize(preds[0,0,:,:],dim=1), F.normalize(labels[0,0,:,:],dim=1))
#output.backward()