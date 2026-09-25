from ultralytics import YOLO

# Load the best trained model
model = YOLO(
    r"runs\detect\runs\helmet_detection\weights\best.pt"
)

# Run detection on an unseen test image
results = model.predict(
    source=r"dataset\images\test\BikesHelmets107_png_jpg.rf.726f8df3e4af954012964df0da215860.jpg",
    conf=0.25,
    save=True
)

print("Prediction completed!")