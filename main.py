import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# args_path = "E:\\python\\Projects\\TestCameraVision\\TestCamera\\pythonProject\\yolov5\\runs\\train\\exp\\weights\\last.pt"
# model = torch.hub.load("ultralytics/yolov5", "custom", args_path, force_reload=True)



def bounding_box(image, index, box):
    #"datasets/data/labels/classes.txt"
    #"Resources/coco.names.txt"
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (255,0,0)
    final = cv2.rectangle(image, start, end, color, 2)
    start = (box[0], box[1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = classes[index]
    final = cv2.putText(final, text, start, font,1, color, 2, cv2.LINE_AA)
    return final

def detect_class(image):
    height = image.shape[0]
    width = image.shape[1]

    blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608),(0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob) #net/cv2.dnn.readNetFromDarknet(конфиг, веса)
    outs = model.forward(out_layers)
    indexes, scores, boxes = ([] for i in range(3))

    for i in outs:
        for object in i:
            nums = object[5:]
            index = np.argmax(nums)
            score = nums[index]
            if score > 0:
                center_x = int(object[0] * width)
                center_y = int(object[0] * height)
                object_width = int(object[2] * width)
                object_height = int(object[2] * height)
                box = [center_x - object_width // 2, center_y - object_height // 2, object_width, object_height]
                boxes.append(box)
                indexes.append(index)
                scores.append(float(score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        index = indexes[box_index]
        final_image = bounding_box(image, index, box)


    return final_image

def find_on_stream():
    cap = cv2.VideoCapture(0)
    while True:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = detect_class(frame)
            frame = cv2.resize(frame, (800, 600))
            cv2.imshow('Find Pepe', frame)

            if cv2.waitKey(30) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
    layer_names = model.getLayerNames() #имя всех слоёв сети
    out_layers_indexes = model.getUnconnectedOutLayers() #индексы выходных слоёв сети
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")
    find_on_stream()

