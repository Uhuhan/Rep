import numpy as np
import cv2 as cv

def camera_read():
    capture = cv.VideoCapture(0)

    while (True):
        ret, frame = capture.read()
        cv.imshow('Frame', frame)

        if cv.waitKey(30) & 0xFF == 27:
            break

    capture.release()
    cv.destroyAllWindows()

camera_read()
