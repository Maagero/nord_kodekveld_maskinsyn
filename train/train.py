from ultralytics import YOLO

# Load a model
model = YOLO("./runs/detect/train/weights/last.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="model/rdd.yaml", epochs=10, imgsz=640, device="mps")