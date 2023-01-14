#!/usr/bin/python3
import os
import cv2 as cv
#from vidstab.VidStab import VidStab
from adafruit_servokit import ServoKit
import motor
import RPi.GPIO as GPIO
import pygame
import csv
from datetime import datetime
import time
from gpiozero import LED
import json
import numpy as np

# define servokit
kit = ServoKit(channels=16)

# Load the trim values from the configuration file
f = open('config.json')
data = json.load(f)
steering_trim = data['steering_trim']
throttle_lim = data['throttle_trim']

# create data storage
image_dir = 'data' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/images/'
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# initialize devices
pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()
#stabilizer = VidStab()
cap = cv.VideoCapture(0) #video capture from 0 or -1 should be the first camera plugged in. If passing 1 it would select the second camera
cap.set(cv.CAP_PROP_FPS, 30)
i = 0  # image index
action = [0., 0.]
Record_data = -1
led = LED(4)

times_to_read_image = []
times_to_resize = []
times_to_perform_action = []
times_to_print_message = []
times_to_save_data = []

while True:
    try:
        start_time = time.time()

        ret, frame = cap.read() 
        time_to_read_image = time.time() - start_time  

        if frame is not None:
            #cv.imshow('frame', frame)  # debug
            #frame = cv.resize(frame, (int(frame.shape[1]), int(frame.shape[0]))) 
            frame = cv.resize(frame, (300, 300))
            time_to_resize = time.time() - time_to_read_image
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #stabilized_frame = stabilizer.stabilize_frame(input_frame=gray,smoothing_window=4)
        # if stabilized_frame is None:
        #     print('no frame')
        #     break

        #get thorttle and steering values from joysticks.
        pygame.event.pump()
        throttle = round((pygame.joystick.Joystick(0).get_axis(1)),2) # between -1 (max reverse) and 1 (max forward), rounded to 2 sig figs 
        motor.drive(throttle * throttle_lim) # multiplies speed within range -100 to 100 (or whatever throttle_lim is)
        steer = round((pygame.joystick.Joystick(0).get_axis(3)), 2) # between -1 (left) and 1 (right)
        ang = 90 * (1 + steer) + steering_trim
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        kit.servo[15].angle = ang
    
        time_to_perform_action = time.time() - time_to_resize

        # steer = 90 + steering_trim + steer * 90
        # servo.turn(steer)
        # turn(steer)
        print(throttle*throttle_lim, ang)
        time_to_print_message = time.time() - time_to_perform_action

        ##########################################################################################################################################
        action = [steer, throttle] # this MUST be [steering, throttle] because that's the order that train.py expects (originaly it was reversed)
        ##########################################################################################################################################

        # print(f"action: {action}") # debug
        # save image
        if pygame.joystick.Joystick(0).get_button(0) == 1:
            Record_data = Record_data * -1
            if Record_data == 1:
                print("Recording Data")
                led.on()
            else:
                print("Stopping Data Logging")
                led.off()
            time.sleep(0.1)
    
        if Record_data == 1:
            cv.imwrite(image_dir + str(i)+'.jpg', frame) # changed frame to gray
            # save labels
            label = [str(i)+'.jpg'] + list(action)
            label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # write the data
            i += 1
        
            time_to_save_data = time.time() - time_to_print_message
            times_to_save_data.append(time_to_save_data)

        times_to_read_image.append(time_to_read_image)
        times_to_resize.append(time_to_resize)
        times_to_perform_action.append(time_to_perform_action)
        times_to_print_message.append(time_to_print_message)
        # times_to_save_data.append(time_to_save_data)

        #if cv.waitKey(1)==ord('q'):
            #break

    except KeyboardInterrupt:
            break

print("Iterations: ", i)
print("Average time to read image:     ", np.mean(np.array(times_to_read_image)))      
print("Average time to resize image:   ", np.mean(np.array(times_to_resize))) 
print("Average time to perform action: ", np.mean(np.array(times_to_perform_action))) 
print("Average time to print message:  ", np.mean(np.array(times_to_print_message))) 
print("Average time to save data:      ", np.mean(np.array(times_to_save_data))) 
