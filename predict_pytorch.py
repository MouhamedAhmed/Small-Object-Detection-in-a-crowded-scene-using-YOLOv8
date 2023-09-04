from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/yolov8m4/weights/last.pt')

# Run inference on 'bus.jpg' with arguments
out = model.predict('cereal_dataset/test_set/test_466.jpeg', save=True, imgsz=640, conf=0.3)

print(out)