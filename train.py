from ultralytics import YOLO

# Load the last training checkpoint
model = YOLO(
    r"runs\detect\runs\helmet_detection\weights\last.pt"
)

# Resume training from the checkpoint
model.train(resume=True)

print("Training resumed and completed!")