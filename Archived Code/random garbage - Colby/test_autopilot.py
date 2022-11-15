#!/usr/bin/python3
import cv2 as cv
import servo1 as servo
import motor
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize
import torch
from train_and_deploy_autopilot.neural_network_class import NeuralNetwork


# Load model
model = NeuralNetwork([500, 500])
model.load_state_dict(torch.load("train_and_deploy_autopilot/model.pth"))

# Setup Transforms
img2tensor = ToTensor()
resize = Resize(size=(80,60))

# Create video capturer
cap = cv.VideoCapture(0) #video capture from 0 or -1 should be the first camera plugged in. If passing 1 it would select the second camera
# cap.set(cv.CAP_PROP_FPS, 10)


while True:
    ret, frame = cap.read()   
    if frame is not None:
        # cv.imshow('frame', frame)  # debug
        frame = cv.resize(frame, (int(frame.shape[1]), int(frame.shape[0]))) 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img_tensor = img2tensor(gray)
        img_tensor = resize(img_tensor)
        print(img_tensor.shape)
    with torch.no_grad():
        pred = model(img_tensor)
    print(pred)
    steering, throttle = pred[0][0].item(), pred[0][1].item()
    print("steering: ", steering)
    print("throttle: ", throttle)
    th = throttle * 12
    if th > 50:
        th = 50
    motor.drive(th)  
    servo.turn(steering)
        
    if cv.waitKey(1)==ord('q'):
        motor.stop()
        motor.close()
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
        
