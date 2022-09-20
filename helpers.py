import torch
import psutil
import cv2
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

os.chdir("frames")
for file in os.listdir():
    if file[0:6] == "person" and file[-4:] == ".png":
        os.remove(file)


