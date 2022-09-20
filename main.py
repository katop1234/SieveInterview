# model to detect all person frames
from helpers import *

video_filename = get_test_video_filename() # todo change this to get_video_filename() after
cap = cv2.VideoCapture(video_filename)

# RUN IF YOU WANT TO GET THE FRAME OBJECTS
# parse_through_video_for_cropped_objects(cap)


