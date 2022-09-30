import time

import cv2
import keyboard
from tracker import *
from helpers import *

# Create tracker object
tracker = EuclideanDistTracker()

# CODE TO GET BOUNDING BOXES
# for (x, y, w, h) in faces:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,   0), 3)
# # displaying image with bounding box
# cv2.imshow('face_detect', img)

ALL_OBJECTS = []
'''
ALL_OBJECTS is a list with elements of the form:
{
	frame_number: (the frame number),
	objects: [
		{
			"object_id": (unique object_id given to an object tracked over frames),
			"person_type": (one of: player_white, player_light_blue, referee, and other),
			"box_coordinates": (x, y, w, h)
		},
		...
	]
}
to be converted into a json file later
'''

# initialize
all_boxes_ids = []
cap = cv2.VideoCapture(get_video_filename())
model = get_yolo_model()
seen = {"id": "object_type"}

# Masks!
color_list = ['white', 'black', 'blue']
boundaries = [
    [[215, 215, 215], [255, 255, 255]],  # white
    [[0, 0, 0], [85, 85, 85]],           # black
    [[3, 169, 240], [187, 222, 251]]     # blue
]

for i in range(len(color_list)):
    color = color_list[i]
    lower = boundaries[i][0]
    upper = boundaries[i][1]
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    crop_img = frame[ymin:ymax, xmin:xmax]

FRAME_NUM = 0
while True:
    ret, frame = cap.read()

    frame_info = {}
    frame_info["frame number"] = FRAME_NUM
    frame_info["objects"] = []

    # Extract Region of interest
    roi = frame[150: 1280,
          0: -1]  # defines a region of interest (i chose the lower half of the frame because it has what we want)
    height, width, _ = roi.shape

    # 1. Object Detection
    results = model(roi)
    boxes_all = results.xyxyn[0]

    detections = get_boxes_with_persons(boxes_all)  # keeps only boxes with people
    detections = [person.tolist() for person in detections]  # converts from tensor to float bc it's nicer
    detections = [person[0:4] for person in detections]
    detections = get_pixel_values_for_detections(detections, height, width)  # cleans up detections

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    all_boxes_ids.append(boxes_ids)

    # Gets likelihood of belonging to each class for each id
    id_and_likelihoods = []
    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        cropped_image = frame[y1:y2, x1:x2]
        likelihoods = get_likelihoods_of_person_type(id, cropped_image)  # TODO make this function
        id_and_likelihoods.append(likelihoods)

    # Make predictions based on likelihoods and show on the frame
    predictions = get_predicted_type_for_each_id(seen, id_and_likelihoods)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        id_type = predictions[id]
        cv2.putText(roi, str(id) + " " + id_type, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        object_info = {}
        object_info["object_id"] = id
        object_info["person_type"] = id_type
        object_info["box_coordinates"] = (x, y, w, h)
        frame_info["objects"].append(object_info)

        seen[id] = id_type

    # Show the frame
    cv2.imshow("roi", roi)
    # cv2.imshow("Frame", frame)

    # exit the video
    # todo i changed this to pause, but delete the continue to EXIT
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        print("press q to unpuase")
        while not (cv2.waitKey(1) & 0xFF == ord('q')):
            time.sleep(5)
        continue

        cap.release()
        cv2.destroyAllWindows()
        print("Hit q to escape")
        break

    ALL_OBJECTS.append(frame_info)
    FRAME_NUM += 1

cap.release()
cv2.destroyAllWindows()


# todo
def write_to_json():
    return


write_to_json(ALL_OBJECTS)
