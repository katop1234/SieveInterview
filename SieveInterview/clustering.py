from helpers import *

'''
Samples 2000 boxes randomly from yolo across the whole video
and creates subfolders for each kmeans cluster with the box
image files corresponding to that cluster
'''

# Set home directory
PATH = home_dir()
os.chdir(PATH)

# USE THIS TO GENERATE RANDOM FRAMES IN THE FRAMES FOLDER
# parse_through_video_for_cropped_objects(video_filename=get_video_filename(), num_frames_to_keep=1000)

# this list holds all the image filename
PERSONS = get_n_random_boxes(2000)
print("USING", len(PERSONS), "BOXES")

# CNN model that outputs the embedding for an image
# https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
cnn_model = get_cnn_model()
data = {}
output_folder = PATH + "kmeans_clusters/"

# loop through each image in the dataset and assign the embedding to the filename in data
for count, person in enumerate(PERSONS):
    feat = extract_features(person, cnn_model)
    data[person] = feat

    if count % 50 == 0:
        print(str(100 * count / len(PERSONS)) + "% done with feature extraction")

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

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)

# serialize this because we'll use it everytime we get an embedding for a box
write_obj(pca, "fit.pca")

# I reduced to 100 dimensions, kinda overkill but can't hurt.
# 100 explains 84% of the variance which is good enough for me.
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

# Stores serialized objects for groups and kmeans_centers
clear_folder(output_folder)
print("GROUPS ARE", groups)
write_obj(groups, "kmeans.groups")
kmeans_centers = kmeans.cluster_centers_
print("K_means centers are", kmeans_centers)
write_obj(kmeans_centers, "kmeans.centers")

# In output_folder, see the images in each group
for group in groups:
    random.shuffle(groups[group])
    group_folder = output_folder + str(group)
    os.mkdir(group_folder)
    for count, frame in enumerate(groups[group]):
        shutil.copyfile(frame, group_folder + "/" + str(count))

get_kmeans_clusters_output(groups)

play_sound()
plt.show()

# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse)
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance');


