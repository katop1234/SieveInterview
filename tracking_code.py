import cv2
from tracker import *
from helpers import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture(get_video_filename())

# Object detection from Stable camera
model = get_yolo_model()

while True:
    ret, frame = cap.read()

    # Extract Region of interest
    roi = frame[200: 1280, 0: -1] # defines a region of interest (i chose the lower half of the frame because it has what we want)
    height, width, _ = roi.shape

    # 1. Object Detection
    results = model(roi)
    boxes_all = results.xyxyn[0]

    detections = get_boxes_with_persons(boxes_all) # keeps only boxes with people
    detections = [person.tolist() for person in detections] # converts from tensor to float bc it's nicer
    detections = [person[0:4] for person in detections]
    detections = get_pixel_values_for_detections(detections, height, width) # cleans up input

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        cap.release()
        cv2.destroyAllWindows()
        print("Hit q to escape")
        break

cap.release()
cv2.destroyAllWindows()
