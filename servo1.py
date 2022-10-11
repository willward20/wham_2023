from time import time, sleep
from adafruit_servokit import ServoKit

#pca.frequency = 50
kit = ServoKit(channels=16)
calibrate = 7

def right(angle):
    kit.servo[0].angle = 90 + angle
def left(angle):
    kit.servo[0].angle = 90 + angle 
def reset():
    kit.servo[0].angle = 90 + calibrate
def turn(angle):
    turn = 90 + angle * 90 + calibrate
    if turn > 180:
        turn = 180
    elif turn < 0:
        turn = 0
    kit.servo[0].angle = turn