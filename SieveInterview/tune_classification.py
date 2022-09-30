from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from helpers import read_obj, sq_dist_between_two_vectors,\
    get_vector, write_obj, get_ref_val, get_blue_val, get_white_val
from sklearn.svm import SVC
import os, time, numpy as np


# masks!
color_list=['white','black','blue']
boundaries = [
    [[215, 215, 215], [255, 255, 255]], # white
    [[0, 0, 0], [85, 85, 85]], # black
    [[3,169,240], [187,222,251]] # blue
    ]

for i in range(len(color_list)):
    color = color_list[i]
    lower = boundaries[i][0]
    upper = boundaries[i][1]
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    crop_img = frame[ymin:ymax, xmin:xmax]


exit(0)
# SVM to classify each person
# todo lol this doesnt really work im gonna try masks instead ;(
true_blue_files = [file for file in os.listdir("output/8/")]
true_white_files = [file for file in os.listdir("output/12/")]
# true_ref_files = [file for file in os.listdir("output/15/")]

true_blues = [get_vector("output/8/" + file) for file in true_blue_files]
print("GOT TRUE_BLUES")
true_whites = [get_vector("output/12/" + file) for file in true_white_files]
print("GOT TRUE_WHITS")
true_refs = [get_vector("output/15/" + file) for file in os.listdir("output/15/")]
true_refs += [get_vector("output/18/" + file) for file in os.listdir("output/18/")]
true_refs += [get_vector("output/19/" + file) for file in os.listdir("output/19/")]

print("GOT true refs")
X_train = np.array(true_blues + true_whites + true_refs)
y = np.array(["blue" for _ in true_blues] + ["white" for _ in true_whites] + ["ref" for _ in true_refs])

clf = make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced"))
clf.fit(X_train, y)

X_unknown = np.array([get_vector("output/6/" + file) for file in os.listdir("output/6/")])
print("GOT UNKNOWN VECTORS")
predictions = clf.predict(X_unknown)

probs = clf.predict_proba(X_unknown)
for i in range(len(predictions)):
    print(i, predictions[i], probs[i])
exit(0)
# todo the below was too crude and probably not wise. shifting to using SVM instead
whites_vector = read_obj("whites.center")
blues_vector = read_obj("blues.center")
kmeans_centers = read_obj("kmeans.centers")

print("---FOR DECIDING REF THRESHOLD---")
refs_paths = ["19/21", "18/36", "18/23", "6/39",
              "6/69", "8/10", "19/2", "19/40", "19/41", "19/42"]
whites_paths = ["4/55", "4/5", "6/21", "8/20", "12/56", "12/57", "12/25", "12/13", "12/22", "12/79"]
blues_paths = ["0/4", "0/35", "0/74", "1/5", "4/21", "4/57", "5/23", "5/79", "8/38", "8/39", "8/40", "8/80", "8/81", "8/82"]

refs_dists = []
# dist to refs
for path in refs_paths:
    ref_vector = get_vector("output/" + path)
    dist = get_ref_val(ref_vector)
    print("FOR REF", path, dist)
    refs_dists.append(dist)

blues_dists = []
# dist to blues
for path in refs_paths:
    ref_vector = get_vector("output/" + path)
    dist = get_blue_val(ref_vector)
    print("FOR BLUES", path, dist)
    blues_dists.append(dist)

whites_dist = []
# dist to whites
for path in refs_paths:
    ref_vector = get_vector("output/" + path)
    dist = get_white_val(ref_vector)
    print("FOR WHITES", path, dist)
    whites_dist.append(dist)

print("REFS DISTS", sorted(refs_dists))
print("WHITES DISTS", sorted(whites_dist))
print("BLUES DISTS", sorted(blues_dists))

print("---FOR DECIDING REF THRESHOLD---")
print("--- ---")
print("---FOR DECIDING ALL THRESHOLDS---")

# FOR REFS
for path in refs_paths:
    test_vector = get_vector("output/" + path)
    ref_dist = get_ref_val(test_vector)
    blue_dist = get_blue_val(test_vector)
    white_dist = get_white_val(test_vector)
    print("REF", path, "refdist", ref_dist, "bluedist", blue_dist, "whitedist", white_dist)

# FOR WHITES
for path in whites_paths:
    test_vector = get_vector("output/" + path)
    ref_dist = get_ref_val(test_vector)
    blue_dist = get_blue_val(test_vector)
    white_dist = get_white_val(test_vector)
    print("WHITE", path, "refdist", ref_dist, "bluedist", blue_dist, "whitedist", white_dist)

# FOR BLUES
for path in blues_paths:
    test_vector = get_vector("output/" + path)
    ref_dist = get_ref_val(test_vector)
    blue_dist = get_blue_val(test_vector)
    white_dist = get_white_val(test_vector)
    print("BLUE", path, "refdist", ref_dist, "bluedist", blue_dist, "whitedist", white_dist)

print("---FOR DECIDING ALL THRESHOLDS---")

print("ref cener to white cemter", get_ref_val(new_whites_vector))
print("ref center to blue center", get_ref_val(new_blues_vector))

