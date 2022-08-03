"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu
GitHub repository: https://github.com/neat-one/nsd_biosoc

Description: script to train the NSDBioSocC3D convolutional neural network.

* This file is under development.

Inital test uses sample_input_output.csv
"""
#%% Import packages
import torch
from NSDBioSocDatasetConstructor import NSDBioSocDataset
from torch import nn # tools in the neural network module
from torch.utils.data import DataLoader # DataLoader is a class that feeds info into the model during training
from torchvision import transforms
from NSDBioSocNeuralNetwork import NSDBioSocC3D
from pathlib import Path
import os

#%% Set up paths to access data
root = Path(os.getcwd())
image_dir = root/'nsd_biosoc_fMRI'
csv_file = root/'nsd_biosoc_data.csv'

# Call dataset constructor to construct training and test datasets
dataset = NSDBioSocDataset(root=root, image_dir=image_dir, csv_file=csv_file, transform = transforms.ToTensor())
training_data, test_data = torch.utils.data.random_split(dataset, [50, 18]) # This will randomly split the data into train and test

#%% Train CNN
# Todo: determine optimal batch size depending on amount of fMRI data that can be processed at once, and with hyperparameter tuning
batch_size = 64  # this is the number of examples that are passed to the neural network at once for better, more efficient training
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# iterate over a dataloader, gives back tuples that include the true image you are training on and the the label for that image
for (x,y) in test_dataloader:
    print(f"Shape of x: [Batch-size, Image-channels, height, width]:")
    print(f"{x.shape}, {x.dtype}")
    print("Shape of y: [1 label per Batch]")
    print(f"{y.shape}, {y.dtype}")
    break

labels = [
    "non-social",
    "social",
]

# Todo: update input dimension arguments based on data
model = NSDBioSocC3D(input_dim_1=50, input_dim_2=50)
print(model)

#%%
loss_fn = nn.CrossEntropyLoss() # when you are producing a probability that something matches a label, this is a good function
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

# Perform the optimization
size = len(train_dataloader.dataset)

# iterate
for (batch_num, (x, y)) in enumerate(train_dataloader):
    optimizer.zero_grad()  # zero the gradient
    pred = model(x)  # calculate the label for the input image -- will be wrong at first due to random params
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if batch_num % 100 == 0:
        (loss, current) = (loss.item(), batch_num * len(x))
        print(f"loss {loss: >7f} [{current:>5d}/{size:>5d}]")

#%% Save trained network
# Todo: save a .pkl file version of the trained network so that it can be easily reloaded and distributed without requiring retraining
import os
PATH = os.getcwd() + '/cnn/trained_cnn/trained_model.pkl'
torch.save(model.state_dict(), PATH)


# Load trained network to check that it was saved properly
model = NSDBioSocC3D()
model.load_state_dict(torch.load(PATH))
model.eval()


#%% Saving and loading models during training
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, PATH)


model = NSDBioSocC3D()
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()