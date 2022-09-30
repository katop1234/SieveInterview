import torch
def get_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    return model

model = get_yolo_model()
print(type(model), dir(model))
