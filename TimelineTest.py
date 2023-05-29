import pprint
from threading import Thread
from threading import Lock
import time
from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv
from numpy import number

from detection_nanodet.nanodet import NanoDet
import myparser
import imutils
from collections import deque

backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


output = ""
strToParse = ""
lock = Lock()
strLock = Lock()
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
frames = deque(maxlen=SAMPLE_DURATION)
video_path = "Medicamento.mp4"
camera = cv.VideoCapture(0)
startTime = time.time()
timeline = deque()
activityLock = Lock()


def parse_timeline():
    global strToParse
    global output
    time.sleep(10)
    parser_tatsu = myparser.TatsuParser()
    semantics = myparser.TatsuSemantics()
    while True:
        time.sleep(4)
        strLock.acquire()
        ast = parser_tatsu.parse(strToParse, start='start')
        output = semantics.expression(ast)
        strLock.release()
        if output is not None:
            print(output)
            time.sleep(1)
            output = None


def constructTimeline():
    while True:
        global strToParse
        time.sleep(5)
        print(timeline)
        lock.acquire()
        strLock.acquire()
        strToParse = ""
        for x in timeline:
            for y in x:
                if not isinstance(y, float):
                    strToParse = strToParse + y + " "
        strLock.release()
        lock.release()
        strToParse = strToParse[:-1]


def cleanTimeline():
    while True:
        time.sleep(25)
        if len(timeline) > 0:
            x = timeline[-1]
            x = x[-1]  # Time of last position in deque
            y = timeline[0]
            y = y[-1]  # Time of first position in deque
            lock.acquire()
            while (y - x) > 20.0:
                timeline.pop()
                x = timeline[-1]
                x = x[-1]
            lock.release()


def drinking():
    model_path = os.path.join('./drink', 'runs', 'detect', 'train2', 'weights', 'last.pt')
    model = YOLO(model_path)
    ret, frame = camera.read()
    H, W, _ = frame.shape
    class_name_dict = {0: 'drinking', 1: 'not drinking'}
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.32:
                lock.acquire()
                timeline.appendleft([class_name_dict[int(class_id)], time.time() - startTime])
                lock.release()
        ret, frame = camera.read()


def getClasses(preds):
    arr = []
    i = 0
    for pred in preds:
        if pred[-1].astype(np.int32) == 0 or pred[-1].astype(np.int32) == 39 and i < 10:
            arr.append(classes[pred[-1].astype(np.int32)])
            i += 1

    return arr


def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()
    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT,
                                    value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale


def object_detection():
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    model = NanoDet(modelPath='detection_nanodet/object_detection_nanodet_2022nov.onnx',
                    prob_threshold=0.35,
                    iou_threshold=0.6,
                    backend_id=backend_id,
                    target_id=target_id)

    while True:
        hasFrame, frame = camera.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)
        # Inference
        preds = model.infer(input_blob)
        # Get classes and fill timeline
        arrayClasses = getClasses(preds)
        arrayClasses.append(time.time() - startTime)
        lock.acquire()
        timeline.appendleft(arrayClasses)
        lock.release()


def medicine_detection():
    model_path = os.path.join('./medicine','best.pt')
    model = YOLO(model_path)
    ret, frame = camera.read()
    H, W, _ = frame.shape
    class_name_dict = {0: 'medicine'}
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.3:
                lock.acquire()
                timeline.appendleft([class_name_dict[int(class_id)], time.time() - startTime])
                lock.release()
        ret, frame = camera.read()


if __name__ == '__main__':
    t1 = Thread(target=drinking)
    t2 = Thread(target=object_detection)
    t3 = Thread(target=constructTimeline)
    t4 = Thread(target=cleanTimeline)
    t5 = Thread(target=parse_timeline)
    t6 = Thread(target=medicine_detection)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    while True:
        hasFrame, frame = camera.read()
        cv.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        if output is None:
            action = "no action"
        else:
            action = output
        #cv.putText(frame, action, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #print(timeline)
        cv.imshow("model", frame)
        cv.waitKey(1)
    print("Done")
