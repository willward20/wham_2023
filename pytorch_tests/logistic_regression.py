"""
This program is based on PythonEngineer PyTorch tutorials on YouTube
and Assignment number 4 from Applied Deep Learning
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


# Class for creating a custom dataset
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

# Create a dataset using donkey car data
annotations_file = "data.csv"
img_dir = "images"
donkey_data = CustomImageDataset(annotations_file, img_dir)

train_dataloader = DataLoader(donkey_data, batch_size=999, shuffle=False)

train_features, steering, throttle = next(iter(train_dataloader))

X = train_features # the images; torch.Size([# images, 3, 120, 160]);  type: torch.ByteTensor
y = steering # torch.Size([# images]);  type: torch.DoubleTensor
y = y.float() # convert y to type: torch.FloatTensor

# process / reshape the images 
# Explore your dataset
M_train = X.shape[0]  # Number of training examples
image_size = X.shape[1:]  # size of each picture torch.Size([3, 120, 160])

# Flatten and standardize
X_flatten = X.reshape(M_train, -1) # new shape: torch.Size([# images, 3*120*160])
X = X_flatten / 255.0

n_samples, n_features = X.shape

# split data into test and train
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1234)
                                                            # test size = 20%

# scale our features
sc = StandardScaler() # makes features to have zero mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert data to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
#y_train = torch.from_numpy(y_train.astype(np.float32))
#y_test = torch.from_numpy(y_test.astype(np.float32))


# reshape y tensors as a columne vecotr (one column)
y_train = y_train.view(y_train.shape[0], 1) # built in function from pytorch that reshapes tensor with given size
y_test = y_test.view(y_test.shape[0], 1)




# Step 1: Set up the model
# model is linear combination of weights and bias
# then apply sigmoid function at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 1 value at the end of the model

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

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