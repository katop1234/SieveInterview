import torch
import psutil
import shutil
import random
import os
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
import torch
from random import randint
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
from glob import glob
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model

def get_video_filename():
    # todo you may have to change this for the deliverable
    return "/home/katop/Desktop/1678_3566_final_four.webm.mp4"

def home_dir():
    PATH = "/home/katop/Desktop/SieveInterview/"
    return PATH
def get_test_video_filename():
    # todo you may have to change this for the deliverable
    return "/home/katop/Desktop/5sec.mp4"

# Model

def read_obj(file_name):
    file_path = home_dir() + "serialized/" + file_name
    file = open(file_path, "rb")
    obj = pickle.load(file)
    return obj

def write_obj(obj, file_name):
    file_path = home_dir() + "serialized/" + file_name
    if not os.path.exists(file_path):
        file = open(file_path, "xb")
    else:
        file = open(file_path, "wb")
    pickle.dump(obj, file)

def get_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    return model

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
def parse_through_video_for_cropped_objects(video_filename, num_frames_to_keep = -1):
    frames_folder_path = home_dir() + "frames/"
    clear_folder(frames_folder_path)
    os.chdir(frames_folder_path)

    model = get_yolo_model()
    TOTAL_FRAMES = get_frames_of_video(get_video_filename())
    frames_to_keep = [i for i in range(TOTAL_FRAMES)]

    cap = cv2.VideoCapture(video_filename)

    print("TOTAL FRAMES DETECTED", TOTAL_FRAMES)

    if num_frames_to_keep != -1:
        frames_to_keep = random.sample(frames_to_keep, num_frames_to_keep)

    FRAME_NUM = 0
    while True and FRAME_NUM < TOTAL_FRAMES:
        ret, frame = cap.read()

        # Only work on frames we want to keep
        if FRAME_NUM not in frames_to_keep:
            FRAME_NUM += 1
            continue

        # Breaks out of video
        if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
            cap.release()
            cv2.destroyAllWindows()
            print("broke out of loop at frame", FRAME_NUM)
            break

        # Show video only if all frames selected
        if num_frames_to_keep == -1: cv2.imshow('frame', frame)

        # Create subfolder with frames
        frame_name = "frame" + str(FRAME_NUM)
        if frame_name not in os.listdir():
            os.mkdir(frame_name)

        # Prepare to put all the bounding boxes in the subfolder for that frame
        os.chdir(frame_name)
        results = model(frame)

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
            cv2.imwrite('person' + str(i) + '.png',
                        cropped_image)  # todo use for debugging to see the cropped image

        os.chdir("..")
        FRAME_NUM += 1

def get_n_random_boxes(n=1000):
    PATH = home_dir()
    os.chdir(PATH + "frames/")
    result = [y for x in os.walk(PATH) for y in
              glob(os.path.join(x[0], '*.png'))]
    os.chdir(PATH)
    if n == -1: # don't randomly sample
        return result
    else:
        return random.sample(result, n)

def get_parent_dir(yourpath):
    return os.path.abspath(os.path.join(yourpath, os.pardir))

def play_sound(secs=3):
    duration = secs  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def clear_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print('cleared folder', folder_name)
    else:
        print("WARNING: the folder you specified to clear does not exist", folder_name)
    os.mkdir(folder_name)

def get_output_folder(groups):
    output_folder = home_dir() + "output/"
    clear_folder(output_folder)
    for group in groups:
        group_folder = output_folder + str(group)
        os.mkdir(group_folder)
        for count, frame in enumerate(groups[group]):
            shutil.copyfile(frame, group_folder + "/" + str(count))
    return

def get_frames_of_video(video_filename):
    assert os.path.exists(video_filename)
    frame_count = 0
    cap = cv2.VideoCapture(video_filename)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_count += 1
    return frame_count

kmeans_centers = read_obj("kmeans.centers")

def get_center_of_centers(l):
    output = []
    coord = 0
    for j in range(len(l[0])):
        for i in range(len(l)):
            cluster = l[i]
            cluster_coord = cluster[j]
            coord += cluster_coord
        output.append(coord / len(l))
        coord = 0

    return output

def sq_dist_between_two_vectors(a, b):
    val = 0
    for i in range(min(len(a), len(b))):
        val += (a[i] - b[i]) ** 2
    if len(a) < len(b):
        val += sum([b[i]**2 for i in range(len(a), len(b))])
    else:
        val += sum([a[i] ** 2 for i in range(len(b), len(a))])
    return val

def get_variance_of_clusters(l):
    center_point = get_center_of_centers(l)
    var = 0
    for cluster_center in l:
        var += sq_dist_between_two_vectors(center_point, cluster_center)
    return var

# todo uncomment
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def get_cnn_model():
    return model

import time


def extract_features(file, model):
    # load the image as a 224x224 array
    start = time.time()
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    end = time.time()
    # print("EXTRACTING FEATURES TOOK", end - start)
    return features

def get_vector(box_file_path):
    features = extract_features(box_file_path, get_cnn_model())
    pca = read_obj("fit.pca")
    img_to_dim_reduced_vector = pca.transform(features)[0]
    return img_to_dim_reduced_vector

def get_vector_from_array(arr):
    cv2.imwrite(home_dir() + "serialized/" + 'temp.png',
                arr)  # todo use for debugging to see the cropped image
    return get_vector(home_dir() + "serialized/" + 'temp.png')

white_vector = read_obj("whites.center")
blue_vector = read_obj("blues.center")
ref_center0 = read_obj("ref.center0")
ref_center1 = read_obj("ref.center1")
ref_center2 = read_obj("ref.center2")
ref_center3 = read_obj("ref.center3")

# because i found many good clusters
def get_ref_val(vector):
    return min(
        sq_dist_between_two_vectors(vector, ref_center0),
        sq_dist_between_two_vectors(vector, ref_center1),
    sq_dist_between_two_vectors(vector, ref_center2),
    sq_dist_between_two_vectors(vector, ref_center3)
    )

def get_blue_val(vector):
    return sq_dist_between_two_vectors(vector, blue_vector)

def get_white_val(vector):
    return sq_dist_between_two_vectors(vector, white_vector)

def get_likelihoods_of_person_type(id, arr):
    vector = get_vector_from_array(arr)

    white_val = get_white_val(vector)
    blue_val = get_blue_val(vector)
    ref_val = get_ref_val(vector)

    likelihoods = {"white": white_val,
                   "blue": blue_val,
                    "ref": ref_val,
                   "id": id}
    return likelihoods

def get_predicted_type_for_each_id(seen, id_to_likelihood):
    print("ID TO LIKELIHOOD")
    id_to_likelihood.sort(key=lambda x: x["id"])
    print(id_to_likelihood)
    MAX_BLUE = 5
    MAX_WHITE = 5
    predictions = {"id": "type"}

    # if we know the type from previous frame
    for detection in id_to_likelihood:
        id = detection["id"]
        if id in seen:
            predictions[id] = seen[id]
            continue

    # check if blue
    count = 1
    id_to_likelihood.sort(key = lambda x: x["blue"])
    for detection in id_to_likelihood:
        if count > MAX_BLUE:
            break

        id = detection["id"]
        if id not in predictions: # todo set threshold
            if detection["blue"] < 2500 and detection["blue"] < detection["ref"] and detection["blue"] < detection["white"]:
                predictions[id] = "blue"

        count += 1

    # check if ref
    for detection in id_to_likelihood:
        id = detection["id"]
        if id not in predictions:
            if detection["ref"] < 2800 and detection["ref"] < detection["white"]:
                predictions[id] = "ref"

    # check if white
    count = 1
    id_to_likelihood.sort(key=lambda x: x["white"])
    for detection in id_to_likelihood:
        if count > MAX_WHITE:
            break

        # TODO i set this threshold through trial and error
        id = detection["id"]
        if id not in predictions:  # todo set threshold
            if detection["white"] < 2500 and detection["white"] < detection["ref"] and detection["white"] < detection["blue"]:
                predictions[id] = "white"

        count += 1

    print("PREDICTIONS BEFORE ASSINGING TO OTHER", "len predictions", len(predictions), "len seen", len(seen))
    print(predictions, "xxx")
    # assign remaining to other
    for detection in id_to_likelihood:
        other_id = detection["id"]
        if other_id not in predictions:
            predictions[other_id] = "other"

    return predictions




