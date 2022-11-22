"""
Template for training data with a NN model.
"""
import os
from math import floor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(3*640*480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

DEVICE = torch.device("cuda")

# Class for creating a dataset from our collected data
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255 
        #print(image.float().size())
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image.float(), steering, throttle


# Create a dataset
annotations_file = "labels.csv"  # the name of the csv file
img_dir = "images"  # the name of the folder with all the images in it
collected_data = CustomImageDataset(annotations_file, img_dir)

print("data length: ", len(collected_data))

# Define Training Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, steering, throttle) in enumerate(dataloader):
        #Combine steering and throttle into one tensor (2 columns, X rows)
        y = torch.stack((steering, throttle), -1) 
        #y = y.float()

        X, y = X.to(DEVICE), y.to(DEVICE)
        #print("Size X: ", X.size()) # torch.Size([BATCHSIZE, 3, 480, 640])

        # Compute prediction error
        pred = model(X)  # forward propagation
        loss = loss_fn(pred, y)  # compute loss
        optimizer.zero_grad()  # zero previous gradient
        loss.backward()  # back propagatin
        optimizer.step()  # update parameters
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
# Define a test function to evaluate model performance
def test(dataloader, model, loss_fn):
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, steering, throttle in dataloader:

            #Combine steering and throttle into one tensor (2 columns, X rows)
            y = torch.stack((steering, throttle), -1) 
            y = y.float()
            
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss



# Start training
train_data_len = len(collected_data)
train_data_size = round(train_data_len*0.9)
test_data_size = round(train_data_len*0.1) 
print("len and train and test: ", train_data_len, " ", train_data_size, " ", test_data_size)

test_loss = []

# Initialize the model
model = NeuralNetwork().to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

# Load the datset (split into train and test)
train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
train_dataloader = DataLoader(train_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)
epochs = 5

# Optimize the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    loss = test(test_dataloader, model, loss_fn)
    test_loss.append(loss)    

print(f"Optimize Done!")

print("test lost: ", test_loss)


# Load an image from the dataset and make a prediction
image = read_image('images/200.jpg').to(DEVICE)  # read image to tensor
image = (image.float() / 255 ) # convert to float and standardize between 0 and 1
print("loaded image after divide and float: ", image.size())
image = image.unsqueeze(dim=0) # add an extra dimension that is needed in order to make a prediction
print("loaded image after unsqueeze: ", image.size())
pred = model(image)
print(pred)

# Save the model
torch.save(model.state_dict(), "basement_cnn_model.pth")
print("Saved PyTorch Model State to basement_cnn_model.pth")

