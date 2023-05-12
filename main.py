import pprint
import threading
import numpy as np
import cv2 as cv
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

lock = threading.Lock()
array_t1 = np.empty((20, 10), np.chararray)
array_t2 = np.empty((20, 10), np.chararray)
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
frames = deque(maxlen=SAMPLE_DURATION)
camera = cv.VideoCapture(0)


# ///////////////////OBJECT DETECTION///////////////////
def parse_timeline():
    parser_tatsu = myparser.TatsuParser()
    semantics = myparser.TatsuSemantics()
    oldstr = ''
    while True:
        lock.acquire()
        # Flatten array to create string to parse
        flattenedMatrix = array_t1.flatten()
        lock.release()
        str = ''
        for item in flattenedMatrix:
            if item is not None:
                str = str + item + ' '
        str = str[:-1]
        pprint.pprint(str)
        if str != '' or oldstr == str:
            ast = parser_tatsu.parse(str, start='start')
            output = semantics.expression(ast)
            if output is not None:
                print(output)
                output = None
        oldstr = str


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


def getClasses(preds):
    arr = np.empty(10, np.chararray)
    i = 0
    for pred in preds:
        if pred[-1].astype(np.int32) == 0 or pred[-1].astype(np.int32) == 39 and i < 10:
            # arr = np.append(arr, classes[pred[-1].astype(np.int32)])
            arr.put(i, classes[pred[-1].astype(np.int32)])
            i += 1

    return arr


def object_detection():
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    model = NanoDet(modelPath='detection_nanodet/object_detection_nanodet_2022nov.onnx',
                    prob_threshold=0.35,
                    iou_threshold=0.6,
                    backend_id=backend_id,
                    target_id=target_id)

    tm = cv.TickMeter()
    tm.reset()

    print("Press any key to stop video capture")

    cap = cv.VideoCapture(0)
    frameNumber = 0

    threading.Thread(target=parse_timeline, daemon=True).start()

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)
        # Inference
        tm.start()
        preds = model.infer(input_blob)
        tm.stop()

        cv.imshow("NanoDet Demo", frame)

        tm.reset()

        # Get classes and fill timeline
        arrayClasses = getClasses(preds)
        lock.acquire()
        array_t1[frameNumber] = arrayClasses
        lock.release()
        # Restart position in array
        frameNumber += 1

        if frameNumber % 20 == 0:
            frameNumber = 0


# ///////////////////HUMAN ACTIVITY///////////////////
def human_activity():
    #SAMPLE_DURATION = 16
    #SAMPLE_SIZE = 112

    # Initialize the frames queue used to store a rolling sample duration of frames -- this queue will automatically pop out
    # old frames and accept new ones
    frames = deque(maxlen=SAMPLE_DURATION)

    # Load the human activity recognition model
    print("[INFO] Loading the human activity recognition model...")
    net = cv.dnn.readNet("human-activity-recognition/resnet-34_kinetics.onnx")

    # Grab the pointer to the input video stream
    print("[INFO] Accessing the video stream...")
    vs = cv.VideoCapture(0)

    # Loop over the frames from the video stream
    while cv.waitKey(1) < 0:
        # Read the frame from the video stream
        (grabbed, frame) = vs.read()
        # If the frame was not grabbed then we have reached the end of the video stream
        if not grabbed:
            print("[INFO] No frame read from the video stream - Exiting...")
            break
        # Resize the frame (to ensure faster processing) and add the frame to our queue
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
        pprint.pprint(label)
        cv.imshow("Activity Recognition", frame)


def modified_human_activity(frame, net):

    # Resize the frame (to ensure faster processing) and add the frame to our queue
    frame = imutils.resize(frame, width=400)
    frames.append(frame)
    # If the queue is not filled to sample size, continue back to the top of the loop and continue
    # pooling/processing frames
    # Now the frames array is filled, we can construct the blob
    blob = cv.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                 swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    # Pass the blob through the network to obtain the human activity recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]
    cv.imshow("Activity Recognition", frame)
    return label


def modified_object_detection(frame, model):
    input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_blob, letterbox_scale = letterbox(input_blob)
    preds = model.infer(input_blob)
    arrayClasses = getClasses(preds)
    pprint.pprint(arrayClasses)
    return arrayClasses


def run_models():
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]
    model = NanoDet(modelPath='detection_nanodet/object_detection_nanodet_2022nov.onnx',
                    prob_threshold=0.35,
                    iou_threshold=0.6,
                    backend_id=backend_id,
                    target_id=target_id)

    net = cv.dnn.readNet("human-activity-recognition/resnet-34_kinetics.onnx")
    frameNumber = 0
    #threading.Thread(parse_timeline()).start()
    camera = cv.VideoCapture(0)
    while True:
        (grabbed, frame) = camera.read()
        if grabbed:
            t1 = threading.Thread(target=modified_object_detection, args=(frame,model))
            #t2 = threading.Thread(modified_human_activity(frame,net),daemon=True)
            t1.run()
            #t2.start()
            return_t1 = t1.join()
            #pprint.pprint(return_t1)
            #return_t2 = t2.join()
            #lock.acquire()
            #return_t1 = getClasses(return_t1)
            #if return_t2 == 'drinking':
            #    i=0
            #    arr = np.empty(11, np.chararray)
            #    for x in range(len(return_t1)):
            #        arr.put(i, return_t1[i])
            #        i += 1
            #    arr[i] = return_t2
            #    return_t1 = arr
            #array_t1[frameNumber] = return_t1
            #lock.release()
            ## Restart position in array
            #frameNumber += 1
            #
            #if frameNumber % 20 == 0:
            #    frameNumber = 0


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
            print(label)
            cv.imshow("Activity Recognition", frame)
            cv.waitKey(1)


if __name__ == '__main__':
    # backend_id = backend_target_pairs[0][0]
    # target_id = backend_target_pairs[0][1]
    # model = NanoDet(modelPath='detection_nanodet/object_detection_nanodet_2022nov.onnx',
    #                 prob_threshold=0.35,
    #                 iou_threshold=0.6,
    #                 backend_id=backend_id,
    #                 target_id=target_id)
    #

    t2 = threading.Thread(modified_human_activity_v2(),daemon=True)
    t2.start()
    t2.join()
    #t3 = threading.Thread(parse_timeline())
    #t3.start()
    #t3.join()
    print("Done")
