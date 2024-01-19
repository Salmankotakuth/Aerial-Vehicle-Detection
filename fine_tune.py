from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.pt')  # load a pre-trained model (recommended for training)

# Train the model
results = model.train(data='VisDrone.yaml', epochs=50, imgsz=640)
