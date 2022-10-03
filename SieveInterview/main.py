from tracker import *
from helpers import *

'''
How my code works:

1. Run clustering.py to get ~20 clusters of boxes across the entire video. 
I hope that I can find clusters corresponding to each of the 3 target classes among
these 20 and use them as baselines.
- I use yolov5 to identify each box, then get the second-to-last layer from 
a VGG model run for each box to get that box's embedding. 
- Then I run PCA and reduce it to 100 dimensions to get a unique 100-dimensional
vector for each box. 
- Finally, I use kmeans clustering to group them into ~20 clusters
and manually look through them to see which ones might have clusterings for 
the target classes. 
- Once I find clusters that correspond to referee, player_light_blue, and
player_white, I get the center of each of those clusters and store them. 
I will use the distance to these centers to figure out the likelihood of
 a box belonging to that cluster.

2. Run main.py to do the rest
- The code iterates over each frame, and uses yolo to identify
a person object's box. 
- For each box, I get the VGG embedding from it and then reduce it to 100
dimensions and compare its distance to one of the three previously determined
center vectors for each class. 
- Since the above wasn't giving me sufficient accuracy, I also add "masking"
code that checks what percent of pixels in each box correspond to white, blue, 
or floor and used that as an additional heuristic. I know this would require 
more manual intervention if the lighting or team color changes, but I found
the code wasn't that hard to write and improved accuracy a bit.
- I added logic for tracking over multiple frames from code I found online.
- The results are all stored in the home folder.

'''

# Create tracker object
tracker = EuclideanDistTracker()

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
to be converted into a json file at the end
'''

# Initialize
all_boxes_ids = []
cap = cv2.VideoCapture(get_video_filename())
model = get_yolo_model()
seen = {"id": "object_type"}
clear_folder(home_dir() + "detections/")

FRAME_NUM = 0
while True:
    ret, frame = cap.read()

    frame_info = {}
    frame_info["frame_number"] = FRAME_NUM
    frame_info["objects"] = []

    # Extract Region of interest
    # roi = get_region_of_interest(frame) # use this for debugging because it focuses on the court only not audience
    roi = frame
    height, width, _ = roi.shape

    # 1. Object Detection
    results = model(roi)
    boxes_all = results.xyxyn[0]
    detections = get_boxes_with_persons(boxes_all)
    detections = [person.tolist() for person in detections]  # converts from tensor to float bc it's nicer
    detections = [person[0:4] for person in detections] # person[4] just tells us object type
    detections = get_pixel_values_for_detections(detections, height, width)  # cleans up detections

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    all_boxes_ids.append(boxes_ids)

    # Gets likelihood of belonging to each class for each id
    os.chdir("detections/")
    frame_dir = "frame" + str(FRAME_NUM)
    if not os.path.exists(frame_dir): os.mkdir(frame_dir)
    os.chdir(frame_dir)

    id_and_likelihoods = []
    print("------------------------------FRAME", FRAME_NUM, "------------------------------")
    # Sort boxes by id
    boxes_ids.sort(key=lambda x:x[4])

    for box_id in boxes_ids:
        x1, y1, x2, y2, id = box_id
        cropped_image = roi[y1:y2, x1:x2]

        # Get mask percents
        white_percent = get_mask_percent(cropped_image, "white")
        blue_percent = get_mask_percent(cropped_image, "blue")
        floor_percent = get_mask_percent(cropped_image, "floor")
        # print("id frame", id, FRAME_NUM, "white%", white_percent, "blue%", blue_percent, "floor%", floor_percent)

        # Get likelihoods of belonging to a class and add to list of current id's
        likelihoods = get_likelihoods_of_person_type(id, cropped_image)
        id_and_likelihoods.append(likelihoods)

    # Make predictions based on likelihoods and show on the frame
    predictions = get_predicted_type_for_each_id(seen, id_and_likelihoods)

    # Store the box in detections/frame_number/
    for box_id in boxes_ids:
        x1, y1, x2, y2, id = box_id
        id_type = predictions[id]
        cropped_image = roi[y1:y2, x1:x2]
        cv2.imwrite(str(id) + "_" + id_type + ".png", cropped_image)

    # Show boxes on screen and update the all_objects dict
    for box_id in boxes_ids:
        x1, y1, x2, y2, id = box_id
        id_type = predictions[id]

        # Show bounding boxes
        cv2.putText(roi, str(id) + " " + id_type, (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Update objects infos
        object_info = {}
        object_info["object_id"] = id

        object_info["person_type"] = get_submission_type(id_type)
        object_info["box_coordinates"] = (x1, y1, x2 - x1, y2 - y1)
        frame_info["objects"].append(object_info)

        seen[id] = id_type

    # Stores png of current frame
    cv2.imwrite(home_dir() + "/frames_for_submission_video/" + "frame" + str(FRAME_NUM) + ".png", roi)

    # Show the frame
    # cv2.imshow("roi", roi)
    # show_masked(roi, "white")

    # Pause the video
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        print("press q to unpause")
        while not (cv2.waitKey(1) & 0xFF == ord('q')):
            time.sleep(5)
        continue

    # Saves likelihoods and relevant data in each frame folder for debugging
    text_file = open("id_and_likelihood.txt", "w")
    text_file.write(str(id_and_likelihoods))
    text_file.close()
    text_file = open("seen.txt", "w")
    text_file.write(str(seen))
    text_file.close()

    os.chdir(home_dir())

    ALL_OBJECTS.append(frame_info)
    FRAME_NUM += 1

cap.release()
cv2.destroyAllWindows()

get_output_video()
write_to_json(ALL_OBJECTS)
