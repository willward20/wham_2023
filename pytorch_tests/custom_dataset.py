import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

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

annotations_file = "donkey_data.csv"
img_dir = "donkey_images"
donkey_data = CustomImageDataset(annotations_file, img_dir)

train_dataloader = DataLoader(donkey_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(donkey_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, steering, throttle = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Steering batch shape: {steering.size()}")
print(f"Throttle batch shape: {throttle.size()}")
img = train_features[0, 0] # 120 x 160  -- you're accessing the first color channel
img2 = np.stack((train_features[0,0], train_features[0,1], train_features[0,2]), axis=-1)
steering_label = steering[0]
throttle_label = throttle[0]
plt.imshow(img2)
print(f"Steering Label: {steering_label}")
print(f"Throttle Label: {throttle_label}")
plt.show()