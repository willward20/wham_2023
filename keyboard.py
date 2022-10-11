import numpy as np
import cv2 as cv
from vidstab.VidStab import VidStab
import servo1 as servo
import time
import motor
import RPi.GPIO as GPIO

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
    #cv.imshow('frame', stabilized_frame)
    
    if stabilized_frame is None:
        break
    k = cv.waitKey(1)
    print(k)
    
    if str(k) == "81":
        servo.left(left_steer)
        if left_steer > -90:
            left_steer = left_steer - 5
        right_steer = 0
    elif str(k) == "82":
        motor.forward(speed)
        right_steer = 0
        left_steer = 0
        servo.reset()
    elif str(k) == "83":
        servo.right(right_steer)
        if right_steer < 90:
            right_steer = right_steer + 5
        left_steer = 0
    elif str(k) == "84":
        motor.stop()
    elif str(k) == "32":
        speed = 100
        motor.forward(speed)
    elif str(k) == "227":
        motor.stop()
        servo.reset()
        GPIO.cleanup()
        cap.release()
        cv.destroyAllWindows()
        False
    elif str(k) == "173":
        if speed > 0:
            speed = speed - 1
    elif str(k) == "171":
        if speed < 100:
            speed = speed + 1
    """
    cv.imwrite(str(n)+  ".jpg", stabilized_frame)
    n = n + 1
    """
