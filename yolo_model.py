import os.path
import uuid
from datetime import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch



images_dir = 'datasets\\data\\images\\'
labels = ['pepe', 'keycap']
number_imgs = 25

def create_dataset():
    cap = cv2.VideoCapture(0)
    for label in labels:
        print('Collecting images for {}'.format(label))
        time.sleep(5)
        count = 0
        for img_num in range(number_imgs):
            print('Collecting images for {}, image number {}'.format(label, img_num))
            ret, frame = cap.read()
            imgname = os.path.join(images_dir, label + '.' + str(uuid.uuid1()) + '.jpg')
            cv2.imwrite(imgname, frame)
            # cv2.imwrite(images_dir + "%d.jpg" % count, frame)
            # count = count + 1
            cv2.imshow('Image Collection', frame)
            time.sleep(1)

    print(os.path.join(images_dir, labels[0] + '.' + str(uuid.uuid1()) + '.jpg'))
    for label in labels:
        print('Collecting images for {}'.format(label))
        for img_num in range(number_imgs):
            print('Collecting images for {}, image number {}'.format(label, img_num))
            imgname = os.path.join(images_dir, label + '.' + str(uuid.uuid1()) + '.jpg')
            # imgname = os.path.join(images_dir, label + '.jpg')
            print(imgname)

# create_dataset()

args_dir = "Resources\\train\\exp\\weights\\last.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", args_dir, force_reload=True)

def show_result():
    img = os.path.join('datasets\\data\\images\\', 'keycap.c38f8eab-28d3-11ef-95bf-e0d55e8eb90b.jpg')
    results = model(img)
    print(results)
    plt.imshow(np.squeeze(results.render()))
    plt.show()

# show_result()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('frame', np.squeeze(results.render()))
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()