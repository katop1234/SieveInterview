# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model
import torch

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from helpers import *

# Set home directory
PATH = home_dir()
os.chdir(PATH)

# USE THIS TO GENERATE RANDOM FRAMES IN THE FRAMES FOLDER
# parse_through_video_for_cropped_objects(video_filename=get_video_filename(), num_frames_to_keep=1000)

# this list holds all the image filename
PERSONS = get_n_random_boxes(1000)
print("USING", len(PERSONS), "BOXES")

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # todo i think i should uncomment this for the embedding

# todo idk if vgg is good, switch to yolo actually
def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

data = {}
output_folder = PATH + "output/"

# loop through each image in the dataset
for count, person in enumerate(PERSONS):
    # try to extract the features and update the dictionary
    feat = extract_features(person, model) # todo uncomment
    data[person] = feat
    if count % 50 == 0:
        print(str(100 * count / len(PERSONS)) + "% done with feature extraction")
    continue
    try:
        feat = extract_features(person, model)
        data[person] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(output_folder, 'wb') as file:
            pickle.dump(data, file)

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are n samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# get the unique labels
labels = [i for i in range(20)] # keeping 20 for now then manually sorting the labels within subclasses
unique_labels = list(set(labels))

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=20, random_state=22)
pca.fit(feat)
pca_predictive_powers = pca.explained_variance_ratio_
print("PCA PREDICTS THIS MUCH VARIATION", sum(pca_predictive_powers), pca_predictive_powers)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=len(unique_labels), random_state=22)
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups:
        groups[cluster] = []
    groups[cluster].append(file)

# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25));
    # gets the list of filenames for a cluster
    files = groups[cluster]

    if len(files) > 100:
        files = files[:100]
        print("Limited images shown to 100 in view_cluster function")
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# # this is just incase you want to see which value for k might be the best
# sse = []
# list_k = list(range(3, 50))
#
# for k in list_k:
#     km = KMeans(n_clusters=k, random_state=22)
#     km.fit(x)
#     sse.append(km.inertia_)

clear_folder(output_folder)
print("GROUPS ARE", groups)
kmeans_centers = kmeans.cluster_centers_
print("K_means centers are", kmeans_centers)
write_obj(kmeans_centers, "kmeans.centers")

for group in groups:
    random.shuffle(groups[group]) # makes sure each subcluster has a uniform distribution
    group_folder = output_folder + str(group)
    os.mkdir(group_folder)
    for count, frame in enumerate(groups[group]):
        shutil.copyfile(frame, group_folder + "/" + str(count))

get_output_folder(groups)

play_sound()
plt.show()

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance');


