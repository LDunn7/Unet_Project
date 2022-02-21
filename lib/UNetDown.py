import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import math

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, int(round((out_channels+in_channels)/2)), kernel_size=3)
        self.conv2 = nn.Conv2d(int(round((out_channels+in_channels)/2)), out_channels, kernel_size=3)

    def forward(self, t):

        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv1(t)
        t = F.relu(t)
        print(t.shape)

        t = self.conv2(t)
        t = F.relu(t)
        print(t.shape)

        return t


class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, int(round((out_channels+in_channels)/2)), kernel_size=3)
        self.conv2 = nn.Conv2d(int(round((out_channels+in_channels)/2)), out_channels, kernel_size=3)

    def forward(self, t, td):

        t = self.up(t)
        print(t.shape)

        crop = torchvision.transforms.CenterCrop(len(t[0,0,:]))
        t = torch.cat((crop(td),t), 1)
        print(t.shape)

        t = self.conv1(t)
        t = F.relu(t)
        print(t.shape)

        t = self.conv2(t)
        t = F.relu(t)
        print(t.shape)

        return t