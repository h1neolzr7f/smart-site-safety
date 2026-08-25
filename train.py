import torch
from ultralytics import YOLO


device = 0 if torch.cuda.is_available() else "cpu"
workers = 8 if device != "cpu" else 0
cache = "ram" if device != "cpu" else False

# 本地若无 yolo26s.pt，可改成 yolov8s.pt
model = YOLO("yolo26s.pt")

model.train(
    data="safety_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16 if device != "cpu" else 4,
    device=device,
    workers=workers,
    patience=20,
    amp=device != "cpu",
    cache=cache,
    verbose=True,
    save=True,
    project="runs/detect",
    name="yolo26s",
)
