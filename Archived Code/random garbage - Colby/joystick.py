import numpy as np
import cv2 as cv
from vidstab.VidStab import VidStab
import servo1 as servo
import time
import motor
import RPi.GPIO as GPIO
import pygame

pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()

stabilizer = VidStab()
np_frame = []
cap = cv.VideoCapture(0) #video capture from 0 or -1 should be the first camera plugged in. If passing 1 it would select the second camera
cap.set(cv.CAP_PROP_FPS, int(10))
n = 0
steering_handler = 0
speed = 100
right_steer = 0
left_steer = 0

while True:
    #captures camera frame by frame
    
    ret, frame = cap.read() #frame gets next frame via cap, ret obtains return value from the frame either true or false depending if the frame is capture
    
    if frame is not None:
        cv.imshow('frame', frame)
        frame = cv.resize(frame, (int(frame.shape[1]/8), int(frame.shape[0]/8))) 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pass
    stabilized_frame = stabilizer.stabilize_frame(input_frame=gray,smoothing_window=4)
    if stabilized_frame is None:
        break
    cv.waitKey(1)

    #get values from driving and turning joysticks.
    pygame.event.pump()
    drive = round((pygame.joystick.Joystick(0).get_axis(1)),2)
    motor.drive(drive)
    #print(drive)
    turn = (pygame.joystick.Joystick(0).get_axis(3))
    servo.turn(turn)
    #print(turn)
        
    """
    cv.imwrite(str(n)+  ".jpg", stabilized_frame)
    n = n + 1
    """
