import torch
import psutil
import cv2
import os
import matplotlib.pyplot as plt

def get_video_filename():
    # todo you may have to change this for the deliverable
    return "/home/katop/Desktop/1678_3566_final_four.webm.mp4"

# Model
def get_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom



