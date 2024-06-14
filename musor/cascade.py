import cv2 as cv

face_cascade = cv.CascadeClassifier('E:\\python\\Projects\\TestCameraVision\\Data\\haarcascades\\haarcascade_frontalface_default.xml')

def camera_read():
    capture = cv.VideoCapture(0)
    while (True):
        ret, frame = capture.read()
        res_frame = cv.resize(frame, (800, 600))

        faces = face_cascade.detectMultiScale(res_frame, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv.rectangle(res_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.imshow('Frame', res_frame)
        if cv.waitKey(30) & 0xFF == 27:
            break

    capture.release()
    cv.destroyAllWindows()

camera_read()