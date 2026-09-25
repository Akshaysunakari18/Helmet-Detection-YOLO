from pathlib import Path
import random
import shutil

# Original dataset folders
source_dirs = [
    ("train", "train/images", "train/labels"),
    ("valid", "valid/images", "valid/labels"),
    ("test", "test/images", "test/labels"),
]

# Output dataset
output_dir = Path("dataset")

# 70 / 20 / 10 split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Reproducible random split
random.seed(42)

# Collect image-label pairs
pairs = []

for split_name, image_dir, label_dir in source_dirs:
    image_path = Path(image_dir)
    label_path = Path(label_dir)

    for image_file in image_path.glob("*.jpg"):
        label_file = label_path / f"{image_file.stem}.txt"

        if label_file.exists():
            pairs.append((image_file, label_file))

print(f"Total image-label pairs found: {len(pairs)}")

# Shuffle
random.shuffle(pairs)

# Calculate split sizes
total = len(pairs)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_data = pairs[:train_end]
val_data = pairs[train_end:val_end]
test_data = pairs[val_end:]

print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")

# Create directories
for split in ["train", "val", "test"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_data(data, split):
    for image_file, label_file in data:

        destination_image = output_dir / "images" / split / image_file.name
        destination_label = output_dir / "labels" / split / label_file.name

        shutil.copy2(image_file, destination_image)
        shutil.copy2(label_file, destination_label)


# Copy files
copy_data(train_data, "train")
copy_data(val_data, "val")
copy_data(test_data, "test")

print("\nDataset split completed successfully!")
print(f"Dataset location: {output_dir.resolve()}")