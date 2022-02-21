import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from Loading import *
from scipy.interpolate import interp2d

class ImageData(Dataset):
    def __init__(self, label_images_file, data_images_file, images_num, transform = None, target_transform = None):
        self.label_images = np.zeros([1,images_num,1200,1200])
        for i in range(images_num):
            if i == 19 or i == 20 or i == 21 or i == 22 or i == 34 or i == 35 or i == 36:
                pass
            else:
                self.label_images[0,i,:,:] = (loadtiffs(str(label_images_file)+'/'+str(i+1)+'.tif')[:,:,0])
        self.data_images = np.zeros([1,images_num,1200,1200])
        for i in range(images_num):
            x = (loadtiffs(str(data_images_file)+'/'+str(i+1)+'.tif')[0,:,0])
            y = (loadtiffs(str(data_images_file)+'/'+str(i+1)+'.tif')[:,0,0])
            X, Y = np.meshgrid(x, y)
            z = np.sin(np.pi * X / 2) * np.exp(Y / 2)
            img = interp2d(x, y, z, 'cubic')
            x2 = np.linspace(0, 4, 1200)
            y2 = np.linspace(0, 4, 1200)
            image = img(x2, y2)
            self.data_images[0,i,:,:] = image
            #self.data_images[0,i,:,:] = (loadtiffs(str(data_images_file)+'/'+str(i+1)+'.tif')[:,:,0])
        self.transform = transform
        self.target_transform = target_transform

    def load(self):
        return self.label_images, self.data_images

    def __len__(self):
        return len(self.label_images)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_images, self.label_images.iloc[item, 0])
        image = read_image(img_path)
        label = self.label_images.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

test = ImageData('/Users/liamdunn/Documents/University/Y4Project/Images/ISM_Train', '/Users/liamdunn/Documents/University/Y4Project/Images/Conf_Train', 50).load()
labels, data = test
labels = torch.tensor(labels)
data = torch.tensor(data)
print(labels.shape)
print(data.shape)
#print(data)
