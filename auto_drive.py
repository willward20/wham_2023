import os
import numpy as np
import cv2 as cv
# import tensorflow as tf
import tflite_runtime.interpreter as tflite



# load model
model_path = "model_04080133.tflite"
interpreter = tflite.Interpreter(model_path)
# model = tf.keras.models.load_model(model_path)
# model.summary()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

th = 0  # throttle
st = 0  # steering
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    im = cv.resize(frame, (150, 150))
    view = np.expand_dims(im, axis=0)
    # take down key pressed
    k = cv.waitKey(1) & 0xFF
    if k == ord("q"):
        th = 0
        st = 0
        break
    else:
        a = np.argmax(np.squeeze(model.predict(view)))
        if a == 0:
            left_forward()
        elif a == 1:
            forward()
        elif a == 2:
            right_forward()

    # operations on the frame
    cv.imshow("image", im)
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
