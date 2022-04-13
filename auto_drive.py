import os
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
from gpiozero import LED, Robot


# load model
model_path = "model_04131023.tflite"
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# init robot
en_l = LED(26)
en_r = LED(12)
bot = Robot((20, 21), (6, 13))
# init camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FPS, 20)
en_l.on()
en_r.on()
act = 0

# Main
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    im = cv.resize(frame, (200, 200))
    view = np.expand_dims(im, axis=0)  #(200,200,3) -> (1,200,200,3)
    # take down key pressed
    k = cv.waitKey(1) & 0xFF
    if k == ord("q"):
        bot.stop()
        break
    else:
        interpreter.set_tensor(input_details[0]['index'], view.astype(np.float32))
        interpreter.invoke()
        act_arr = interpreter.get_tensor(output_details[0]['index'])
        act = np.argmax(np.squeeze(act_arr))
        if act == 0:  # fleft
            bot.left_motor.forward(0.32)
            bot.right_motor.forward(0.64)
            print("forward left")
        elif act == 1:  # forward
            bot.forward(0.64)
            print("forward")
        elif act == 2:  # fright
            bot.left_motor.forward(0.64)
            bot.right_motor.forward(0.32)
            print("forward right")
    # show image, TODO: headless
    cv.imshow("image", im)
# When everything done, release the capture, disable the robot
cap.release()
cv.destroyAllWindows()
robot.stop()
en_l.off()
en_r.off()
