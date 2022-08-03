"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: provides functions to construct training and test data for training NSDBioSocC3D networks.

* This file is under development.

"""
# Import packages
import os
from torch.utils.data import Dataset # Todo: confirm that Dataset constructor is sufficient, not VisionDataset
import torch
import PIL
import pandas as pd


#%%
# Todo: ensure this can handle batched data to only process small number of trials at a time

class NSDBioSocDataset(Dataset):

    def __init__(self, root, image_dir, csv_file, transform=None):
        """
        Creates an iterable dataset from a directory containing individual files to be processed in sequence.

        :param root: root directory
        :param image_dir:
        :param csv_file:
        :param transform:
        """
        self.root = root
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.data = pd.read_csv(csv_file).iloc[:, 1] #csv_file second column should contain the labels for the data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # Todo: make this compatible with fMRI data, not image data
    def __getitem__(self, index):
        """Should generate input-output pairs, where input is fMRI data, output is the social (1)/ non-social (0) label."""
        image_name = os.path.join(self.image_dir, self.image_files[index])
        image = PIL.Image.open(image_name)
        label = self.data[index]
        # Transforms
        if self.transform:
            # Transform can convert PIL image or numpy.ndarray to tensor
            image = self.transform(image)
        return (image, label)



#%% Alternate implementation
class NSDBioSocDatasetDF():

    def __init__(self, data):
        """
        Creates an iterable dataset from a pandas.DataFrame object

        :param data: a .csv file
        """
        # loading the csv file from the folder path
        df = pd.read_csv(data, delimiter=',')

        # y should contain the column index for the social/ non-social label
        # x should contain the column index for the numpy.ndarray conversion of the nifty file
        # Todo: check that conversion from pandas.DataFrame to tensor works as expected
        self.x = torch.tensor(df['Betas'])
        self.y = torch.tensor(df['final_socnonsoc'])
        self.n_samples = df.size

    # Todo: support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # we can call tensor.len() to return the size
    def __len__(self):
        return self.x.len()