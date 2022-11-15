import numpy as np
import cv2 as cv
import csv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    action = np.random.randn(2)
    data = [str(i)+'.jpg'] + list(action)
    # cv.imshow("frame", gray)  # display frame
    # save image to file
    cv.imwrite('test_images/'+str(i)+'.jpg', gray)
    # append labels to csv
    with open('test_images/labels.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)  # write the data

    if cv.waitKey(1) == ord("q"):
        break
    i += 1
    print(i)
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
