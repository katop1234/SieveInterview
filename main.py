# model to detect all person frames
from helpers import *

video_filename = get_video_filename()
cap = cv2.VideoCapture(video_filename)

model = get_yolo_model()

while 0:
   ret, frame = cap.read()
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False:
       cap.release()
       cv2.destroyAllWindows()
       break
    # Inference
   results = model(frame)

   # Results
   results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

ret, frame = cap.read()
results = model(frame)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()

# Get boxes
print("FIRST", results.xyxyn)

boxes = results.xyxyn[0]
print(len(boxes), len(boxes[0]), len(boxes[0][0]))

# Get unique cropped image
height = frame.shape[0]
width = frame.shape[1]

x1 = int((boxes[0][0]) * width)
y1 = int((boxes[0][1]) * height)
x2 = int((boxes[0][2]) * width)
y2 = int((boxes[0][3]) * height)

cropped_image = frame[y1:y2, x1:x2]
plt.imshow(cropped_image)
cv2.imwrite('test1.png', cropped_image)


