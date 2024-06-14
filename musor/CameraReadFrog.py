from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os

model_dir = '/\\'
path = os.path.join(model_dir, 'my_resnet18.onnx')
args_confidence = 0.2
CLASSES = ['frog', 'raspberry']
print("[INFO] loading model...")
net = cv2.dnn.readNetFromONNX(path)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
frame = vs.read()
frame = imutils.resize(frame, width=400)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), scalefactor=1.0 / 224
                                 , size=(224, 224), mean=(104, 117, 123), swapRB=True)
    cv2.imshow("Cropped image", cv2.resize(frame, (224, 224)))
    net.setInput(blob)
    detections = net.forward()
    print(list(zip(CLASSES, detections[0])))

    confidence = abs(detections[0][0] - detections[0][1])
    print(confidence)
    if (confidence > args_confidence):
        class_mark = np.argmax(detections)
        cv2.putText(frame, CLASSES[class_mark], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)

    cv2.imshow("Web camera view", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()