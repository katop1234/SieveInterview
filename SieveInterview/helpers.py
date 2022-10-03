import json
import psutil
import shutil
import random
import time
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.models import Model
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import cv2
from glob import glob
import pickle
import numpy as np
import os

PATH = os.getcwd()

def home_dir():
    return PATH + "/"

def get_video_filename():
    return home_dir() + "1678_3566_final_four.webm.mp4"

def read_obj(file_name):
    '''reads a file in the serialized/ folder'''
    file_path = home_dir() + "serialized/" + file_name
    file = open(file_path, "rb")
    obj = pickle.load(file)
    return obj

def write_obj(obj, file_name):
    '''writes a file to the serialized/ folder'''
    file_path = home_dir() + "serialized/" + file_name
    if not os.path.exists(file_path):
        file = open(file_path, "xb")
    else:
        file = open(file_path, "wb")
    pickle.dump(obj, file)

def get_yolo_model():
    '''returns yolo model. Try running this as few times as possible since it takes a while'''
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    return model

def tensor_row_is_person(row):
    '''Returns whether the 6th element of results.xyxyn is 0. 0 is the id for person'''
    return row[5] == 0

def get_pixel_values_for_detections(detections, height, width):
    '''
    original input is of the form [x1, y1, x2, y2]
    where each value is actually a proportion from 0 to 1 of the pixel number
    we need it to be of the form [x1, y1, x2, y2] where each is the
    actual pixel count (an integer)
    '''

    new_detections = []
    for detection in detections:
        x1 = int(detection[0] * width)
        y1 = int(detection[1] * height)
        x2 = int(detection[2] * width)
        y2 = int(detection[3] * height)
        new_detections.append([x1, y1, x2, y2])

    return new_detections

def get_boxes_with_persons(boxes_all):
    '''Keeps only objects identified to be persons'''
    boxes_persons = []
    for i in range(len(boxes_all)):
        person = boxes_all[i]
        if tensor_row_is_person(person):
            boxes_persons.append(person)
    return boxes_persons
def parse_through_video_for_cropped_objects(video_filename, num_frames_to_keep = -1):
    '''Use this function for model tuning. Basically goes through the video and
    any time yolo detects a person, it stores the box in a specified folder. Helpful for
    debugging and seeing what the persons look like'''
    frames_folder_path = home_dir() + "boxes/"
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
    '''Use after parse_through_video_for_cropped_objects function.
    Basically goes through all the boxes stored by that function and returns a list
    of n randomly sampled from them'''
    PATH = home_dir()
    os.chdir(PATH + "boxes/")
    result = [y for x in os.walk(PATH) for y in
              glob(os.path.join(x[0], '*.png'))]
    os.chdir(PATH)
    if n == -1: # don't randomly sample
        return result
    else:
        return random.sample(result, n)

def get_parent_dir(yourpath):
    '''equal to cd .. '''
    return os.path.abspath(os.path.join(yourpath, os.pardir))

def play_sound(secs=3):
    '''if i want to do something else while the code is running'''
    duration = secs  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def clear_folder(folder_name):
    '''deletes all the contents of a folder'''
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print('cleared folder', folder_name)
    else:
        print("WARNING: the folder you specified to clear does not exist", folder_name)
    os.mkdir(folder_name)

def get_output_folder(groups):
    '''get output folder for k-means clustering groups. each folder will have
    the boxes of the identified objects for that cluster'''
    output_folder = home_dir() + "output/"
    clear_folder(output_folder)
    for group in groups:
        group_folder = output_folder + str(group)
        os.mkdir(group_folder)
        for count, frame in enumerate(groups[group]):
            shutil.copyfile(frame, group_folder + "/" + str(count))
    return

def get_frames_of_video(video_filename):
    '''returns frame count of a video'''
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
    '''returns center of a list of vectors'''
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
    '''returns the squared distance between two vectors'''
    val = 0
    for i in range(min(len(a), len(b))):
        val += (a[i] - b[i]) ** 2
    if len(a) < len(b):
        val += sum([b[i]**2 for i in range(len(a), len(b))])
    else:
        val += sum([a[i] ** 2 for i in range(len(b), len(a))])
    return val

def get_variance_of_clusters(l):
    '''returns the variance of a list of vectors (which are supposed to be a cluster).
    Variance is the average squared distance to the mean over that cluster'''
    center_point = get_center_of_centers(l)
    var = 0
    for cluster_center in l:
        var += sq_dist_between_two_vectors(center_point, cluster_center)
    return var

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def get_cnn_model():
    '''returns a vgg model that outputs embedding vectors for an image. try to run as
    few times as possible since it takes some time'''
    return model

def extract_features(file, model):
    '''takes an image file and returns vgg embeddings'''
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
    '''takes a png and extracts embeddings using a vgg model.
    then reduces it down to 100 dimensions with PCA'''
    features = extract_features(box_file_path, get_cnn_model())
    pca = read_obj("fit.pca")
    img_to_dim_reduced_vector = pca.transform(features)[0]
    return img_to_dim_reduced_vector

def get_vector_from_array(arr):
    '''Gets the feature vector for a arr object representing a frame in cv2'''
    cv2.imwrite(home_dir() + "serialized/" + 'temp.png',
                arr)  # todo use for debugging to see the cropped image
    return get_vector(home_dir() + "serialized/" + 'temp.png')

# Helper functions use these below
# These are the center of the k-means clusters for each class
white_vector = read_obj("whites.center")
blue_vector = read_obj("blues.center")
ref_center0 = read_obj("ref.center0")
ref_center1 = read_obj("ref.center1")
ref_center2 = read_obj("ref.center2")
ref_center3 = read_obj("ref.center3")

def get_ref_val(vector):
    '''Sq dist to referee k-means center. I used 4 because i found that
    many good clusters and just decided to take the avg dist'''
    return np.mean([
        sq_dist_between_two_vectors(vector, ref_center0),
        sq_dist_between_two_vectors(vector, ref_center1),
        sq_dist_between_two_vectors(vector, ref_center2),
        sq_dist_between_two_vectors(vector, ref_center3)
                    ])

def get_blue_val(vector):
    '''sq dist to center of blue players cluster'''
    return sq_dist_between_two_vectors(vector, blue_vector)

def get_white_val(vector):
    '''sq dist to center of white players cluster'''
    return sq_dist_between_two_vectors(vector, white_vector)

def get_likelihoods_of_person_type(id, arr):
    '''Gives the sq dist to the center of the white, blue, and referee clusters.
    Also gives the percent of the box that has a certain color range. These are
    all used in determining the class for that object id'''
    vector = get_vector_from_array(arr)

    white_val = get_white_val(vector)
    blue_val = get_blue_val(vector)
    ref_val = get_ref_val(vector)
    white_percent = get_mask_percent(arr, "white")
    blue_percent = get_mask_percent(arr, "blue")
    floor_percent = get_mask_percent(arr, "floor")

    likelihoods = {
                    "white": white_val,
                   "blue": blue_val,
                    "ref": ref_val,
                   "white_percent": white_percent,
                   "blue_percent": blue_percent,
                   "floor_percent": floor_percent,
                   "id": id
                   }

    # Ease of reading when debugging
    for key in likelihoods:
        likelihoods[key] = round(likelihoods[key], 2)
    return likelihoods

def get_predicted_type_for_each_id(seen, id_to_likelihood):
    '''Takes in a seen dictionary of object ids already seen and classified,
    as well as id_to_likelihood dict with the id of each object in the current frame and
    relevant likeliehoods that help me determine which class they're in.
    Returns predictions of the form {id1: "blue", id2: "white ...etc}'''
    # These thresholds were set through manual inspection

    # For distance to center of K-means cluster
    BLUE_DIST_THRESHOLD = 3750
    WHITE_DIST_THRESHOLD = 3750

    # For minimum percentage of this color
    BLUE_PERCENT_THRESHOLD = 5
    WHITE_PERCENT_THRESHOLD = 15
    FLOOR_PERCENT_THRESHOLD = 7.5

    # No more than these many players for that class
    MAX_BLUE_PLAYERS = 5
    MAX_WHITE_PLAYERS = 5

    predictions = {"id": "type"}

    # If we know the type from previous frame
    for detection in id_to_likelihood:
        id = detection["id"]
        if id in seen:
            predictions[id] = seen[id]
            continue

    # Check for blue players
    # Sort by dist to blue players k-means center
    id_to_likelihood.sort(key = lambda x: x["blue"])

    count = 0
    for detection in id_to_likelihood:
        if count == MAX_BLUE_PLAYERS:
            break

        id = detection["id"]
        if id not in predictions:
            if detection["blue"] < detection["ref"] and detection["blue"] < detection["white"]:
                if detection["floor_percent"] > FLOOR_PERCENT_THRESHOLD and detection["blue"] < BLUE_DIST_THRESHOLD:
                    if detection["blue_percent"] > BLUE_PERCENT_THRESHOLD:
                        predictions[id] = "blue"
                        count += 1

    # Check for white players
    # Sort by dist to white players k-means center
    id_to_likelihood.sort(key=lambda x: x["white"])

    count = 0
    for detection in id_to_likelihood:
        if count == MAX_WHITE_PLAYERS:
            break

        id = detection["id"]
        if id not in predictions:
            if detection["white"] < detection["ref"] and detection["white"] < detection["blue"]:
                if detection["floor_percent"] > FLOOR_PERCENT_THRESHOLD and detection["white"] < WHITE_DIST_THRESHOLD:
                    if detection["white_percent"] > WHITE_PERCENT_THRESHOLD:
                        predictions[id] = "white"
                        count += 1

    # Check for referees
    for detection in id_to_likelihood:
        id = detection["id"]
        if id not in predictions:
            if detection["ref"] < detection["blue"] and detection["ref"] < detection["white"]:
                if detection["ref"] < 3000:
                    predictions[id] = "ref"

    # Assign remaining persons to "other"
    for detection in id_to_likelihood:
        other_id = detection["id"]
        if other_id not in predictions:
            predictions[other_id] = "other"

    return predictions

def get_output_video():
    ''' get output video with labelled boxes for submission'''
    img_array = []
    for filename in glob(home_dir() + "/frames_for_submission_video/*.png"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
def get_color_ranges():
    '''Color ranges for determining masks'''
    # this shit is in BGR not RGB
    boundaries = {
        "white" : [[160, 160, 160], [255, 255, 255]],
        "black":  [[0, 0, 0], [85, 85, 85]],
        "blue": [[100, 70, 30], [255, 150, 100]],
        "floor-light": [[135, 165, 180], [195, 215, 235]],
        "floor-dark": [[60, 120, 150], [90, 155, 180]]
    }

    return boundaries

def color_list():
    '''List of colors used in masks'''
    return list(get_color_ranges().keys())

color_list = color_list()
boundaries = get_color_ranges()

def get_submission_type(id_type):
    '''Because I used different naming conventions than what the submission wanted'''
    if id_type == "ref":
        return "referee"
    elif id_type == "blue":
        return "player_light_blue"
    elif id_type == "white":
        return "player_white"
    elif id_type == "other":
        return "other"

def get_mask_percent(box, color):
    '''Returns what percentage of the box has pixels in the color range for that specified color'''
    if color == "floor":
        return get_mask_percent(box, "floor-dark") + get_mask_percent(box, "floor-light")

    lower = np.array(boundaries[color][0])
    upper = np.array(boundaries[color][1])
    mask = cv2.inRange(box, lower, upper)
    masked = cv2.bitwise_and(box, box, mask=mask)

    no_pixel_array = np.array([0, 0, 0])
    count, total = 0, 0
    for i in range(len(masked)):
        for j in range(len(masked[i])):
            total += 1
            curr_pixel_array = masked[i][j]
            if not np.array_equal(curr_pixel_array, no_pixel_array):
                count += 1
    return 100 * count / total

def get_region_of_interest(frame):
    '''Returns a subset of the frame called a region of interest.
    I use this for ease when testing because it keeps the bottom part
    of the frame which is just the court (and don't want to see the whole
    audience)'''
    return frame[200: -1, 0: -1]  # defines a region of interest (i chose the lower half of the frame because it has what we want)


def show_masked(image_original, target_color):
    '''Shows a given image masked for a target color'''
    image = image_original.copy() # not sure if inRange or bitwise_and mutate the image
    color = target_color
    lower = np.array(boundaries[color][0])
    upper = np.array(boundaries[color][1])
    mask = cv2.inRange(image, lower, upper)
    masked = cv2.bitwise_and(image, image, mask = mask)

    cv2.imshow("masked with " + str(color),  masked)
    return

def crop_to_n_frames(n=10):
    ''' crop video to n frames for testing purposes'''
    cap = cv2.VideoCapture(get_video_filename())

    frame_num = 0
    while frame_num < 15:
        ret, frame = cap.read()
        cv2.imwrite(home_dir() + "video_frames/" + "frame" + str(frame_num) + ".png", frame)
        frame_num += 1

    img_array = []
    for filename in glob(home_dir() + "video_frames/*.png"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('cropped_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def write_to_json(python_dict, file_name="all_objects.json"):
    '''Writes python object to a json file'''
    with open(file_name, "w") as outfile:
        json.dump(python_dict, outfile)
    return
