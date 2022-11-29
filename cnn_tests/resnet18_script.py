import numpy as np
import os
import pandas as pd
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, random_split
from torchvision import models
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torchvision.transforms import transforms

torch.cuda.empty_cache()
device = torch.device("cuda")

#applying a transform on all images and building the dataset
transforms = transforms.Compose(
[
    transforms.ToTensor()
])

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


annotations_file = "data2022-11-29-14-31/labels.csv"  # the name of the csv file
img_dir = "data2022-11-29-14-31/images"  # the name of the folder with all the images in it
collected_data = CustomImageDataset(annotations_file, img_dir)

# define parameters
learning_rate = 0.001
batch_size = 50
epochs = 5


# loop over optimization code
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, steering, throttle) in enumerate(dataloader):
        # combine steering and throttle in one tensor
        y = torch.stack((steering, throttle), -1)

        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Evaluate model performance
def test_loop(dataloader, model, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, steering, throttle in dataloader:
            # combine steering and throttle in one tensor
            y = torch.stack((steering, throttle), -1)
            y = y.float()

            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss

test_loss = []
train_data_len = len(collected_data)
train_data_size = round(train_data_len*0.9)
test_data_size = round(train_data_len*0.1)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
print(num_ftrs)
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2 ).to(device)

model = model.to(device)

# initialize loss function and optimizer and pass it to train_loop and test_loop
loss_fn = nn.MSELoss()
# Reduce module error in each training step
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Load the datset (split into train and test)
train_data, test_data = random_split(collected_data, [train_data_size, test_data_size])
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss = test_loop(test_dataloader, model, loss_fn)
    test_loss.append(loss)
print("Done!")
print(f"test loss: {test_loss}")

# Load an image from the dataset and make a prediction
image = read_image('data2022-11-29-14-31/images/2053.jpg').to(device)  # read image to tensor
image = (image.float() / 255 ) # convert to float and standardize between 0 and 1
print("loaded image after divide and float: ", image.size())
image = image.unsqueeze(dim=0) # add an extra dimension that is needed in order to make a prediction
print("loaded image after unsqueeze: ", image.size())
pred = model(image)
print(pred)

# Save the model
torch.save(model.state_dict(), "resnet_cnn_model.pth")
print("Saved PyTorch Model State to resnet_cnn_model.pth")