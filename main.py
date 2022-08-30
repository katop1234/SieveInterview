# model to detect all person frames
import cv2, os
from helpers import *

video_filename = get_video_filename()
cap = cv2.VideoCapture(video_filename)

while True:
   ret, frame = cap.read()
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False:
       cap.release()
       cv2.destroyAllWindows()
       break
   cv2.imshow('frame',frame)

# have a helper function that can pull up the person object's frame for a target
# frame to make manual classification easier.

# manually parse through some of the frames for each desired category and get
# their embedding. store these somewhere as reference

# go through each person objects recognized and calculate the sum euclidean
# distance to each of the desired categories, and pick the top n of each to
# classify as being the that type. also have a way to store the objects in a
# json for each frame, should be given to you in the setup for the models.2
