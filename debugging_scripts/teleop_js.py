import sys
import time
import numpy as np
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor



def get_numControllers():
    return joystick.get_count()


# SETUP
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=8, address=0x40)
steer = kit.servo[0]
MAX_THROTTLE = 0.75
STEER_CENTER = 100
MAX_STEER = 50
assert MAX_THROTTLE <= 1
steer.angle = 90
display.init()
joystick.init()
print(f"{get_numControllers()} joystick connected")
js = joystick.Joystick(0)


# MAIN
try:
    while True:
        for e in event.get():
            if e.type == QUIT:
                print("QUIT detected, terminating...")
                pygame.quit()
                sys.exit()
            if e.type == JOYAXISMOTION:
                v_0 = js.get_axis(0)
                v_4 = js.get_axis(4)
                print(f"steering joystick value: {v_0}, speed joystick value: {v_4}")
                speed = -np.clip(v_4, -MAX_THROTTLE, MAX_THROTTLE)
                if speed > 0:
                    engine.forward(speed)
                elif speed < 0:
                    engine.backward(-speed)
                else:
                    engine.stop()
                ang = STEER_CENTER - MAX_STEER * v_0
                steer.angle = ang
except KeyboardInterrupt:
    engine.stop()
    engine.close()
