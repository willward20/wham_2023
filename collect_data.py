import os
import numpy as np
import cv2 as cv
from datetime import datetime


now = datetime.now()  # current date and time
d = now.strftime("%m%d%H%M")
save_path = "data_" + d
classes = ["forward", "fleft", "fright", "stay"]
if not os.path.exists(save_path):
    os.makedirs(save_path)
    for c in classes:
        os.makedirs(f'{save_path}/{c}')

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
    im = cv.resize(frame, (150, 150))
    # take down key pressed
    k = cv.waitKey(1) & 0xFF
    if k == ord("i"):
        th = 1
        cv.imwrite(f"{save_path}/forward/{i}.jpg", im)
        print("forward")
    elif k == ord("u"):
        th = 1
        st = 1
        cv.imwrite(f"{save_path}/fleft/{i}.jpg", im)
        print("forward left")
    elif k == ord("o"):
        th = 1
        st = -1
        cv.imwrite(f"{save_path}/fright/{i}.jpg", im)
        print("forward right")
    elif k == ord("q"):
        th = 0
        st = 0
        break
    else:
        cv.imwrite(f"{save_path}/stay/{i}.jpg", im)

    act_buf.append([th, st])
    # operations on the frame
    cv.imshow("image", im)
    i += 1
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
d = now.strftime("%m%d%H%M")
with open(f"{d}.npy", "wb") as f:
    np.save(f, np.array(act_buf))
