import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4)) # changing kernel to 4x4
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Conv layers
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)
        
        # Max-Pool layer that we will use multiple times
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=6400, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=30)
        
        # Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv layers have weights initialized with random # drawn from uniform distribution
                m.weight = nn.init.uniform_(m.weight, a=0, b=1) # nn.init.uniform_ does not seem to work for in-place change
            elif isinstance(m, nn.Linear):
                # FC layers have weights initialized with Glorot uniform initialization
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        ## Conv layers
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout1(x)
                
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.dropout4(x)
                
        ## Flatten
        x = x.view(x.size(0), -1) # .view() can be thought as np.reshape
                
        ## Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout5(x)
                
#         x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc2(x)) # experimenting w/ different activation function instead of relu
        x = self.dropout6(x)
                
        x = self.fc3(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
