import os
import numpy as np
import cv2 as cv
from datetime import datetime
from gpiozero import LED, Robot


# create data storage
now = datetime.now()  # current date and time
d = now.strftime("%m%d%H%M")
save_path = "data_" + d
classes = ["forward", "fleft", "fright"]
if not os.path.exists(save_path):
    os.makedirs(save_path)
    for c in classes:
        os.makedirs(f'{save_path}/{c}')
# init robot, Waveshare Motor Driver Board
en_l = LED(26)
en_r = LED(12)
bot = Robot((20, 21), (6, 13))
# init camera, webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FPS, 20)
i = 0
en_l.on()
en_r.on()
act = 0  # stop

# Main
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    im = cv.resize(frame, (200, 200))
    # set action according to key pressed
    k = cv.waitKey(1) & 0xFF
    if k == ord("q"):
        bot.stop()
        print("Data collection terminated!")
        break
    elif k == ord("u"):
        act = 2  # forward left
    elif k == ord("o"):
        act = 3  # forward right
    elif k == ord("i"):
        act = 1  # forward
    # send driving command and record corresponding image
    if act == 0:
        bot.stop()
    elif act == 1:
        bot.forward(0.3)
        cv.imwrite(f"{save_path}/forward/{i}.jpg", im)
        i += 1
    elif act == 2:
        bot.left_motor.forward(0.32)
        bot.right_motor.forward(0.64)
        cv.imwrite(f"{save_path}/fleft/{i}.jpg", im)
        i += 1
    elif act == 3:
        bot.left_motor.forward(0.64)
        bot.right_motor.forward(0.32)
        cv.imwrite(f"{save_path}/fright/{i}.jpg", im)
        i += 1
    # show image, TODO: headless
    cv.imshow('view', im)

# When everything done, release the capture, disable the robot
cap.release()
cv.destroyAllWindows()
bot.stop()
en_l.off()
en_r.off()
