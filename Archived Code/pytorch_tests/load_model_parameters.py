"""
Loads the parameters of a trained model and tests on image.
"""
from torchvision.io import read_image
import torch
from neural_network_class import NeuralNetwork

model = NeuralNetwork([500, 500])
model.load_state_dict(torch.load("model.pth"))

img = read_image('data2022-10-18-16-00/images/10.jpg')  # read image to tensor
#img = totensor(frame)
image = img / 255 

with torch.no_grad():
    pred = model(image)

print(pred)
steering, throttle = pred[0][0].item(), pred[0][1].item()
print("steering: ", steering)
print("throttle: ", throttle)
#motor(throttle)
#servo(steering)