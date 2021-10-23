from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import sys
sys.path.append('C:/Users/cps_lab/darknet/build/darknet/x64')
import orderly
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load
import func_hands

def draw_boxes(detections, image, colors):
    import cv2
    if len(detections)==0:
        text = "No action"
        cv2.putText(image, text, (10,10), cv2.FONT_HERSHEY_PLAIN,1.2, (0, 0, 255), 2)
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        #===========================jerry==========================
        if label == "s":
            func_hands.space()
        if label == "l":
            func_hands.left()
        if label == "u":
            func_hands.up()
        if label == "r":
            func_hands.right()
        if label == "d":
            func_hands.down()
        #===========================jerry==========================
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image

win_title = "windows"
cfg_file = "data/hands/hand-yolov4.cfg"
data_file = "data/hands/obj.data"
weight_file = 'data/hands/hand-yolov4_6000.weights'
thre = 0.7
show_coordinates = True

# Load Network

network, class_names, class_colors = darknet.load_network(
cfg_file,
data_file,
weight_file,
batch_size=1
)
# Get Nets Input dimentions
width = darknet.network_width(network)
height = darknet.network_height(network)

# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture('CoverTestYolo/maintest.mp4')
# Video Stream
while cap.isOpened():
# Get current frame, quit if no frame
    ret, frame = cap.read()
    if not ret: 
        break
    t_prev = time.time()

# Fix image format
    frame_rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize( frame_rgb, (width, height))


# convert to darknet format, save to “ darknet_image “
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

# inference
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)
    darknet.free_image(darknet_image)

# draw bounding box
    # image = covertest(detections,frame_resized)
    image = draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show Image and FPS
    fps = int(1/(time.time()-t_prev))
    cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
    cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(win_title, image)

    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()