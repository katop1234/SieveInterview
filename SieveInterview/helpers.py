import torch
import psutil
import shutil
import random
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt

def get_video_filename():
    # todo you may have to change this for the deliverable
    return "/home/katop/Desktop/1678_3566_final_four.webm.mp4"

def get_test_video_filename():
    # todo you may have to change this for the deliverable
    return "/home/katop/Desktop/5sec.mp4"

# Model
def get_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

def tensor_row_is_person(row):
    return row[5] == 0

def get_pixel_values_for_detections(detections, height, width):
    # original input is of the form [x1, y1, x2, y2]
    # where each value is actually a proportion from 0 to 1 of the pixel number
    # we need it to be of the form [x1, y1, w, h] where each is the
    # actual pixel count (an integer)

    new_detections = []
    for detection in detections:
        x = int(detection[0] * width)
        y = int(detection[1] * height)
        w = int((detection[2] - detection[0]) * width)
        h = int((detection[3] - detection[1]) * height)
        new_detections.append([x, y, w, h])

    return new_detections

def get_boxes_with_persons(boxes_all):
    boxes_persons = []
    for i in range(len(boxes_all)):
        person = boxes_all[i]
        if tensor_row_is_person(person):
            boxes_persons.append(person)
    return boxes_persons

#   TODO this function is too big
# TODO write docs for this function
def parse_through_video_for_cropped_objects(cap=None):
    if cap is None:
        print("WARNING: The video capture object was not specified, defaulting to that for the 5 second version")
        video_filename = get_test_video_filename()  # todo change this to get_video_filename() after
        cap = cv2.VideoCapture(video_filename)

    model = get_yolo_model()

    FRAME_NUM = 0
    while True:
        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame', frame)
        frame_name = "frame" + str(FRAME_NUM)
        if frame_name not in os.listdir():
            os.mkdir(frame_name)

        os.chdir(frame_name)
        results = model(frame)

        # Results todo this is for debugging so comment out later
        # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

        # Get boxes
        boxes = results.xyxyn[0]

        # Get unique cropped image
        height = frame.shape[0]
        width = frame.shape[1]

        for i in range(len(boxes)):
            person = boxes[i]
            if not tensor_row_is_person(person):
                continue

            x1 = int((person[0]) * width)
            y1 = int((person[1]) * height)
            x2 = int((person[2]) * width)
            y2 = int((person[3]) * height)

            cropped_image = frame[y1:y2, x1:x2]
            subfolder = "frame" + str(FRAME_NUM) + "/"
            cv2.imwrite('person' + str(i) + '.png',
                        cropped_image)  # todo use for debugging to see the cropped image

        os.chdir("..")
        FRAME_NUM += 1

def get_n_random_frames(n=100):
    PATH = home_dir()
    os.chdir(PATH + "frames/")
    result = [y for x in os.walk(PATH) for y in
              glob(os.path.join(x[0], '*.png'))]
    os.chdir(PATH)
    if n == -1: # don't randomly sample
        return result
    else:
        random.sample(result, n)

def create_testing_folder_with_n_frames(n=100):
    frames = get_n_random_frames(n)
    print(len(frames), frames)
    frames_folder = home_dir() + "frames/"
    os.chdir(frames_folder)

    sample_frames = frames_folder + "sample_frames/"

    if "sample_frames" in os.listdir(frames_folder): shutil.rmtree(sample_frames)
    os.mkdir(sample_frames)

    for count, frame in enumerate(frames):
        shutil.copyfile(frame, sample_frames + "file" + str(count))

    return None

def get_parent_dir(yourpath):
    return os.path.abspath(os.path.join(yourpath, os.pardir))

def home_dir():
    PATH = "/home/katop/Desktop/SieveInterview/"
    return PATH

def play_sound(secs=3):
    duration = secs  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def clear_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

clear_folder(home_dir() + "output/")
