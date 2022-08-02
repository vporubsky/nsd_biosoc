"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date:
File Final Edit Date:

Description: Contains the NSDBioSocNeuralNetwork class
"""
#%% Import packages
from torch import nn

#%% Define neural network class
class NSDBioSocNeuralNetwork(nn.Module):  # Mechanisms that module defines
    """
    This class is a temporary placeholder for the model which will be designed.
    """
    def __init__(self, input_dim_1, input_dim_2):
        super(NSDBioSocNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # pytorch likes you to set up your models as a bunch of layers, will take image matrix and flatten it into a list of numbers
        self.linear_relu_stack = nn.Sequential(  # sequence of layers that you put together in a stack
            nn.Linear(input_dim_1 * input_dim_2, 512),
            # input dimesions are based on input data, nn.Linear is just a linear transformation of some input to some output, this is alinear transformation that converts our image into something else of size 512 long, 512 is arbitrary
            nn.ReLU(),  # any value in input below zero, sets to zero
            nn.Linear(512, 512),  # linearly recombine into another 512 length vector
            nn.ReLU(),  # rectify it
            nn.Linear(512, 2),
            # convert 512 length vector to 2. this is the num labels in image dataset --> vector of approximate probabilities, but may not sum to 1, can use arctan to get them to be between 0 and 1, usually take highest and assume it is the label
            nn.ReLU()  # rectify it
        )

    def forward(self, x):  # x is whatever the batch of inputs was
        """"""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
