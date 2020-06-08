## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        self.drop_layer1 = nn.Dropout(p=0.1)
        self.drop_layer2 = nn.Dropout(p=0.2)
        self.drop_layer3 = nn.Dropout(p=0.3)
        self.drop_layer4 = nn.Dropout(p=0.4)
        self.drop_layer5 = nn.Dropout(p=0.5)
        self.drop_layer6 = nn.Dropout(p=0.6)
        self.max_pool2D = nn.MaxPool2d(2, 2)
     



    # Note that among the layers to add, consider including:
    # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop_layer1(self.max_pool2D(F.relu(self.conv1(x))))
        x = self.drop_layer2(self.max_pool2D(F.relu(self.conv2(x))))
        x = self.drop_layer3(self.max_pool2D(F.relu(self.conv3(x))))
        x = self.drop_layer4(self.max_pool2D(F.relu(self.conv4(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop_layer5(F.relu(self.fc1(x)))
        x = self.drop_layer6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

