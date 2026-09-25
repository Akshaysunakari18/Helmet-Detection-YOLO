import random
import shutil
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).parent

SOURCE_DIRS = [
    PROJECT_DIR / "train",
    PROJECT_DIR / "valid",
    PROJECT_DIR / "test"
]

OUTPUT_DIR = PROJECT_DIR / "dataset"

# Reproducible random split
random.seed(42)

# ============================================================
# FIND ALL IMAGE + LABEL PAIRS
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

pairs = []

for source_dir in SOURCE_DIRS:

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists():
        print(f"WARNING: Images folder not found: {images_dir}")
        continue

    if not labels_dir.exists():
        print(f"WARNING: Labels folder not found: {labels_dir}")
        continue

    for image_path in images_dir.iterdir():

        if image_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            print(f"WARNING: Label missing for {image_path.name}")

print(f"\nTotal valid image-label pairs found: {len(pairs)}")

if len(pairs) < 500:
    raise ValueError(
        f"Only {len(pairs)} valid pairs found. "
        "The assignment requires at least 500 images."
    )

# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(pairs)

total = len(pairs)

train_count = int(total * 0.70)
val_count = int(total * 0.20)

train_pairs = pairs[:train_count]
val_pairs = pairs[train_count:train_count + val_count]
test_pairs = pairs[train_count + val_count:]

print("\nNew split:")
print(f"Train      : {len(train_pairs)}")
print(f"Validation : {len(val_pairs)}")
print(f"Test       : {len(test_pairs)}")
print(f"Total      : {len(pairs)}")

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# COPY FILES
# ============================================================

def copy_pairs(pair_list, split_name):

    image_output = OUTPUT_DIR / "images" / split_name
    label_output = OUTPUT_DIR / "labels" / split_name

    for image_path, label_path in pair_list:

        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        shutil.copy2(
            label_path,
            label_output / label_path.name
        )


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")

# ============================================================
# FINISHED
# ============================================================

print("\nDataset preparation completed successfully!")

print("\nCreated:")
print(OUTPUT_DIR / "images" / "train")
print(OUTPUT_DIR / "images" / "val")
print(OUTPUT_DIR / "images" / "test")
print(OUTPUT_DIR / "labels" / "train")
print(OUTPUT_DIR / "labels" / "val")
print(OUTPUT_DIR / "labels" / "test")