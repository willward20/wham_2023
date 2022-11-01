"""
Template for training data with a NN model.
"""
import os
import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from math import floor


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
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image.float(), steering, throttle


# Create a dataset
annotations_file = "data2022-10-18-16-00/labels.csv"  # the name of the csv file
img_dir = "data2022-10-18-16-00/images"  # the name of the folder with all the images in it
collected_data = CustomImageDataset(annotations_file, img_dir)


"""
# Test
i = 1
for X, steering, throttle in train_data:
    #print(f"Shape of X in batch {i} [N, C, H, W]: {X.shape}")
    #print(f"Shape of steering in batch {i}: {steering.shape}")
    #print(f"Shape of throttle in batch {i}: {throttle.shape}")
    i += 1
print(i)
"""


# Define Neural Network
class NeuralNetwork(nn.Module):

    def __init__(self, hidden_layer_sizes):
        super().__init__()
        self.flatten = nn.Flatten()
        modules = []
        modules.append(nn.Linear(60*80, hidden_layer_sizes[0]))
        modules.append(nn.ReLU())
        if len(hidden_layer_sizes) > 1:
            for i in range(1, len(hidden_layer_sizes)):
                modules.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
                modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_layer_sizes[-1], 2))
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.flatten(x)
        y_predicted = self.linear_relu_stack(x) 
        return y_predicted # AKA logits
    

# Define Training Function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, steering, throttle) in enumerate(dataloader):
        #Combine steering and throttle into one tensor (2 columns, X rows)
        y = torch.stack((steering, throttle), -1) 
        #y = y.float()
        
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
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss



# Start training
train_data_len = len(collected_data)
train_data_size = floor(train_data_len*0.9)
test_data_size = round(train_data_len*0.1) 

test_loss = []

# Initialize the model
model = NeuralNetwork([500, 500])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

# Load the datset (split into train and test)
train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
train_dataloader = DataLoader(train_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=10)
epochs = 5

# Optimize the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    loss = test(test_dataloader, model, loss_fn)
    test_loss.append(loss)    

print(f"Optimize Done!")

print("test lost: ", test_loss)

"""
learning_rate = 0.001

model = NeuralNetwork([256, 256])
loss_fn = nn.MSELoss() # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

epochs = 5
test_acc = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    #acc = test(test_dataloader, model, loss_fn)
    #test_acc.append(acc)
print("Done!")

#plt.plot(test_acc)
"""
"""
# Scale our features
sc = StandardScaler() # makes features to have zero mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert test/train data back to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
"""