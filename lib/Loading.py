import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np

def loadtiffs(file_name):
    img = Image.open(file_name)
    #print('The Image is', img.size, 'Pixels.')
    #print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return(imgArray)

def savetiffs(file_name, data):
    images = []
    for I in range(np.shape(data)[2]):
        images.append(Image.fromarray(data[:, :, I]))
        images[0].save(file_name, save_all=True, append_images=images[1:])

#data_images = []

#data_images.append(loadtiffs('/Users/liamdunn/Documents/University/Y4Project/Images/confocal-large-pinhole.tif'))
#data_images.append(loadtiffs('/Users/liamdunn/Documents/University/Y4Project/Images/confocal-small-pinhole.tif'))
#data_images.append(loadtiffs('/Users/liamdunn/Documents/University/Y4Project/Images/ISM-large-pinhole.tif'))
#data_images.append(loadtiffs('/Users/liamdunn/Documents/University/Y4Project/Images/ISM-small-pinhole.tif'))

#data_labels = ['confocal-large-pinhole','confocal-small-pinhole','ISM-large-pinhole','ISM-small-pinhole']

#print(data_images)
#trainloader = torch.utils.data.DataLoader(data_images, batch_size=2)

#class Network(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
#        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
       # self.conv3 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=3)

#        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
    #    self.fc2 = nn.Linear(in_features=120, out_features=60)
      #  self.out = nn.Linear(in_features=60, out_features=10)

#    def forward(self, t):

#        t = self.conv1(t)
#        t = F.relu(t)
#        print(t.shape)
     #   t = F.max_pool2d(t, kernel_size=2, stride=2)

#        t = self.conv2(t)
#        t = F.relu(t)
#        print(t.shape)
      #  t = F.max_pool2d(t, kernel_size=2, stride=2)

        #t = self.conv3(t)
        #t = F.relu(t)
        #print(t.shape)

#        t = F.max_pool2d(t, kernel_size=2, stride=2)


    #    t = t.reshape(-1, 12 * 4 * 4)
     #   t = self.fc1(t)
      #  t = F.relu(t)

       # t = self.fc2(t)
        #t = F.relu(t)

      #  t = self.out(t)
 #       return t

#network = Network()

#a = torch.empty((1,1,572,572))
#preds = network(a)
#print(preds.shape)