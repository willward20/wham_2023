"""
Template for training data with a NN model.
"""
import os
import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
Step 1: Create and configure our dataset
"""

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
        image = read_image(img_path)
        steering = self.img_labels.iloc[idx, 1]
        throttle = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return image, steering, throttle


# Create a dataset
annotations_file = "data.csv"  # the name of the csv file
img_dir = "images"  # the name of the folder with all the images in it
donkey_data = CustomImageDataset(annotations_file, img_dir)

# Load the dataset
batch_size = 64
train_dataloader = DataLoader(donkey_data, batch_size=batch_size, shuffle=False)
train_features, steering, throttle = next(iter(train_dataloader))
#train_features, steering, throttle = next(iter(train_dataloader))
#train_features, steering, throttle = next(iter(train_dataloader))
#train_features, steering, throttle = next(iter(train_dataloader))

# Set X and y
X = train_features
print(X.size())
y = torch.stack((steering, throttle), -1) # combines steering and throttle outputs (2 columns, X rows)
y = y.float()

# Flatten and standardize
M_train = X.shape[0]
X_flatten = X.reshape(M_train, -1) # torch.Size([M_train, 3*120*160])
X = X_flatten / 255.0

# Split data into test and train
test_size = 0.2 # percentage of data for testing
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=test_size, random_state=1234)

# Scale our features
sc = StandardScaler() # makes features to have zero mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert test/train data back to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

# Set constants
n_features = X_train.shape[1]
print("n_features: ", n_features)
output_size = y_train.shape[1]
print("output_size: ", output_size)

"""
Step 2: Building the Neural Network Model
"""

# Step 1: Set up the model
# model is linear combination of weights and bias
# then apply sigmoid function at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, n_output_features) # 2 values at the end of the model

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features, output_size)

# Step 2: Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step 3: Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss calcualtion
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward() # calculates gradients

    # updates
    optimizer.step() # pytorch does all udpate calcualtions for us

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# evaluate the model
with torch.no_grad():
    y_predicted = model(X_test)

    # sigmoid returns value between 0 and 1, so convert
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) # sum all the correct predictions 
    print(f'accuracy = {acc:.4f}')
