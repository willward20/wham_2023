import os
import numpy as np
import cv2 as cv
from datetime import datetime


now = datetime.now()  # current date and time
d = now.strftime("%m%d%H%M")
save_path = "data_" + d
if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

act_buf = []
th = 0  # throttle
st = 0  # steering
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    # take down key pressed
    k = cv.waitKey(1) & 0xFF
    if k == ord("i"):
        th = 1
        print("forward")
    elif k == ord("u"):
        th = 1
        st = 1
        print("forward left")
    elif k == ord("o"):
        th = 1
        st = -1
        print("forward right")
    elif k == ord("q"):
        break
    act_buf.append([th, st])
    # operations on the frame
    im = cv.resize(frame, (150, 150))
    cv.imwrite(f"{save_path}/{i}.jpg", im)
    cv.imshow("image", im)
    i += 1
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
d = now.strftime("%m%d%H%M")
with open(f"{d}.npy", "wb") as f:
    np.save(f, np.array(act_buf))
