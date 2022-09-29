# model to detect all person frames
from helpers import *

video_filename = get_test_video_filename() # todo change this to get_video_filename() after
cap = cv2.VideoCapture(video_filename)

# RUN IF YOU WANT TO GET THE FRAME OBJECTS
# parse_through_video_for_cropped_objects(cap)

# creates a mask?

# resources
# https://gist.github.com/Gabe-flomo/83783b2e37bb6ccbf8b752c2560682b5#file-image_clustering-py
# https://gist.github.com/Gabe-flomo/83783b2e37bb6ccbf8b752c2560682b5#file-image_clustering-py
# https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
# https://www.analyticsvidhya.com/blog/2020/12/an-approach-towards-neural-network-based-image-clustering/
# https://datascience.stackexchange.com/questions/45282/generating-image-embedding-using-cnn
# https://rom1504.medium.com/image-embeddings-ed1b194d113e
# https://www.activeloop.ai/resources/generate-image-embeddings-using-a-pre-trained-cnn-and-store-them-in-hub/




