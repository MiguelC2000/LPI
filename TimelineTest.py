import pprint
from threading import Thread
from threading import Lock
import time

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

CLASSES = (
    'abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling',
    'arranging flowers', 'assembling computer', 'auctioning', 'baby waking up', 'baking cookies',
    'balloon blowing', 'bandaging', 'barbequing', 'bartending', 'beatboxing', 'bee keeping', 'belly dancing',
    'bench pressing', 'bending back', 'bending metal', 'biking through snow', 'blasting sand', 'blowing glass',
    'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding', 'bookbinding', 'bouncing on trampoline',
    'bowling', 'braiding hair', 'breading or breadcrumbing', 'breakdancing', 'brush painting', 'brushing hair',
    'brushing teeth', 'building cabinet', 'building shed', 'bungee jumping', 'busking', 'canoeing or kayaking',
    'capoeira',
    'carrying baby', 'cartwheeling',
    'carving pumpkin', 'catching fish', 'catching or throwing baseball', 'catching or throwing frisbee ',
    'catching or throwing softball', 'celebrating', 'changing oil',
    'changing wheel', 'checking tires', 'cheerleading', 'chopping wood ', 'clapping', 'clay pottery making',
    'clean and jerk', 'cleaning floor', 'cleaning gutters', 'cleaning pool', 'cleaning shoes',
    'cleaning toilet', 'cleaning windows', 'climbing a rope', 'climbing ladder', 'climbing tree', 'contact juggling',
    'cooking chicken', 'cooking egg', 'cooking on campfire', 'cooking sausages', 'counting money',
    'country line dancing', 'cracking neck', 'crawling baby', 'crossing river', 'crying', 'curling hair',
    'cutting nails',
    'cutting pineapple', 'cutting watermelon', 'dancing ballet', 'dancing charleston',
    'dancing gangnam style', 'dancing macarena', 'deadlifting', 'decorating the christmas tree', 'digging', 'dining',
    'disc golfing', 'diving cliff', 'dodgeball', 'doing aerobics', 'doing laundry', 'doing nails',
    'drawing', 'dribbling basketball', 'drinking', 'drinking', 'drinking', 'driving car', 'driving tractor',
    'drop kicking', 'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger',
    'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream',
    'eating spaghetti', 'eating watermelon', 'egg hunting', 'exercising arm', 'exercising with an exercise ball',
    'extinguishing fire', 'faceplanting', 'feeding birds', 'feeding fish', 'feeding goats', 'filling eyebrows',
    'finger snapping', 'fixing hair', 'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins',
    'folding paper', 'front raises', 'frying vegetables ', 'garbage collecting', 'drinking', 'getting a haircut',
    'getting a tattoo', 'giving or receiving award', 'golf chipping', 'golf driving', 'golf putting', 'grinding meat',
    'grooming dog', 'grooming horse', 'gymnastics tumbling', 'hammer throw', 'headbanging', 'headbutting', 'high jump',
    'high kick', 'hitting baseball', 'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging',
    'hula hooping', 'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing', 'ice skating', 'ironing',
    'javelin throw ', 'jetskiing', 'jogging', 'juggling balls', 'juggling fire', 'juggling soccer ball',
    'jumping into pool',
    'jumpstyle dancing', 'kicking field goal', 'kicking soccer ball', 'kissing', 'kitesurfing', 'knitting', 'krumping',
    'laughing', 'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich', 'making bed',
    'making jewelry',
    'making pizza', 'making snowman', 'making sushi', 'making tea', 'marching', 'massaging back', 'massaging feet',
    'massaging legs', 'massaging persons head', 'milking cow', 'mopping floor', 'motorcycling', 'moving furnitur',
    'mowing lawn',
    'news anchoring', 'opening bottle', 'opening present', 'paragliding', 'parasailing', 'parkour',
    'passing American football (in game)', 'passing American football (not in game)', 'peeling apples',
    'peeling potatoes',
    'petting animal (not cat)',
    'petting cat', 'picking fruit', 'planting trees', 'plastering', 'playing accordion', 'playing badminton',
    'playing bagpipes', 'playing basketball', 'playing bass guitar', 'playing cards', 'playing cello', 'playing chess',
    'playing clarinet', 'playing controller',
    'playing cricket', 'playing cymbals', 'playing didgeridoo', 'playing drums', 'playing flute', 'playing guitar',
    'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard', 'playing kickball',
    'playing monopoly',
    'playing organ', 'playing paintball',
    'playing piano', 'playing poker', 'playing recorder', 'playing saxophone', 'playing squash or racquetball',
    'playing tennis', 'playing trombone', 'playing trumpet', 'playing ukulele', 'playing violin', 'playing volleyball',
    'playing xylophone',
    'pole vault', 'presenting weather forecast', 'pull ups', 'pumping fist', 'pumping gas', 'punching bag',
    'punching person (boxing)', 'push up', 'pushing car', 'pushing cart', 'pushing wheelchair', 'reading book',
    'reading newspaper',
    'recording music', 'riding a bike', 'riding camel', 'riding elephant', 'riding mechanical bull',
    'riding mountain bike',
    'riding mule', 'riding or walking with horse', 'riding scooter', 'riding unicycle', 'ripping paper',
    'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating', 'running on treadmill', 'sailing',
    'salsa dancing', 'sanding floor', 'scrambling eggs', 'scuba diving', 'setting table', 'shaking hands',
    'shaking head',
    'sharpening knives', 'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes',
    'shooting basketball', 'shooting goal (soccer)', 'shot put', 'shoveling snow ', 'shredding paper',
    'shuffling cards', 'side kick', 'sign language interpreting', 'singing', 'situp', 'skateboarding', 'ski jumping',
    'skiing (not slalom or crosscountry)', 'skiing crosscountry', 'skiing slalom', 'skipping rope', 'skydiving',
    'slacklining', 'slapping', 'sled dog racing', 'smoking', 'smoking hookah', 'snatch weight lifting', 'sneezing',
    'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spinning poi',
    'spray painting', 'spraying', 'springboard diving', 'squat', 'sticking tongue out', 'stomping grapes',
    'stretching arm',
    'stretching leg', 'strumming guitar', 'surfing crowd', 'surfing water', 'sweeping floor',
    'swimming backstroke', 'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs',
    'swinging on something', 'sword fighting', 'tai chi', 'taking a shower', 'tango dancing',
    'tap dancing', 'tapping guitar', 'tapping pen', 'drinking', 'tasting food', 'testifying', 'texting',
    'throwing axe',
    'throwing ball', 'throwing discus', 'tickling', 'tobogganing', 'tossing coin', 'tossing salad',
    'training dog', 'trapezing', 'trimming or shaving beard', 'trimming trees', 'triple jump', 'tying bow tie',
    'tying knot (not on a tie)', 'tying tie', 'unboxing', 'unloading truck', 'using computer',
    'using remote controller (not gaming)', 'using segway', 'vault', 'waiting in line', 'walking the dog',
    'washing dishes',
    'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding',
    'watering plants', 'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs', 'weaving basket', 'welding',
    'whistling', 'windsurfing', 'wrapping present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba)')

strToParse = ""
lock = Lock()
array_t1 = np.empty((20, 10), np.chararray)
array_t2 = np.empty((20, 10), np.chararray)
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
frames = deque(maxlen=SAMPLE_DURATION)
camera = cv.VideoCapture(0)
startTime = time.time()
timeline = deque()
activityLock = Lock()


def modified_human_activity_v2():
    net = cv.dnn.readNet("human-activity-recognition/resnet-34_kinetics.onnx")
    while True:
        (grabbed, frame) = camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=400)
            frames.append(frame)
            # If the queue is not filled to sample size, continue back to the top of the loop and continue
            # pooling/processing frames
            if len(frames) < SAMPLE_DURATION:
                continue
            # Now the frames array is filled, we can construct the blob
            blob = cv.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                         swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            # Pass the blob through the network to obtain the human activity recognition predictions
            net.setInput(blob)
            outputs = net.forward()
            label = CLASSES[np.argmax(outputs)]
            lock.acquire()
            timeline.appendleft([label, time.time() - startTime])
            lock.release()
            cv.imshow("Activity Recognition", frame)
            cv.waitKey(1)


def constructTimeline():
    while True:
        strToParse = ""
        time.sleep(5)
        lock.acquire()
        for x in timeline:
            for y in x:
                if not isinstance(y, float):
                    strToParse= strToParse + y + " "
        lock.release()
        strToParse = strToParse[:-1]
        print(strToParse)


def cleanTimeline():
    while True:
        time.sleep(25)
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


if __name__ == '__main__':
    t2 = Thread(target=modified_human_activity_v2)
    t3 = Thread(target=constructTimeline)
    t4 = Thread(target=cleanTimeline)
    t2.start()
    t3.start()
    t4.start()

    # t3 = threading.Thread(parse_timeline())
    # t2.join()
    # t3.join()
    print("Done")
