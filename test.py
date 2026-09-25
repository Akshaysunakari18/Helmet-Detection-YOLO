from ultralytics import YOLO

# Load the best trained model
model = YOLO(
    r"runs\detect\runs\helmet_detection\weights\best.pt"
)

# Evaluate on the unseen test dataset
results = model.val(
    data="data.yaml",
    split="test",
    imgsz=640
)

print("Test evaluation completed!")