import os
import random
import shutil

# CHANGE THIS PATH to where Kaggle images are extracted
RAW_DATA_DIR = r"C:\Users\mdsha\Downloads\train\train"

BASE_DIR = "data"
SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

cats = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith("cat")]
dogs = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith("dog")]

def split_and_copy(files, label):
    random.shuffle(files)
    total = len(files)

    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)

    split_map = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, filenames in split_map.items():
        for fname in filenames:
            src = os.path.join(RAW_DATA_DIR, fname)
            dst = os.path.join(BASE_DIR, split, label, fname)
            shutil.copy(src, dst)

split_and_copy(cats, "cats")
split_and_copy(dogs, "dogs")

print("✅ Dataset successfully split into train / val / test")
