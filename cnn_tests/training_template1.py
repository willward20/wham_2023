import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class CustomImageDataset(Dataset):
    def __init__(self, transform=None):
        self.img_labels = pd.read_csv("../cnn/labels.csv")
        self.img_dir = "../cnn/images"
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) / 255
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if self.transform:
            image = self.transform(image)
        return image.float(), steering, throttle


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


model = NeuralNetwork()

# define parameters
learning_rate = 1e-100
batch_size = 4
epochs = 10

dataset = CustomImageDataset()
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# loop over optimization code
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, steering, throttle) in enumerate(dataloader):
        # combine steering and throttle in one tensor
        y = torch.stack((steering, throttle), -1)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Evaluate model performance
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, steering, throttle in dataloader:
            # combine steering and throttle in one tensor
            y = torch.stack((steering, throttle), -1)
            y = y.float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# initialize loss function and optimizer and pass it to train_loop and test_loop
loss_fn = nn.MSELoss()
# Reduce module error in each training step
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "model.pth")
