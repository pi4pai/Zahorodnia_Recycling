import os
import shutil
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = BASE_DIR / "data" / "original"
OUTPUT_DIR = BASE_DIR / "data"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def clear_folder(folder_path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def copy_files(files, destination_folder):
    destination_folder.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, destination_folder / file_path.name)


def split_dataset():
    random.seed(RANDOM_SEED)

    if not ORIGINAL_DIR.exists():
        print(f"Папка не знайдена: {ORIGINAL_DIR}")
        return

    clear_folder(TRAIN_DIR)
    clear_folder(VAL_DIR)
    clear_folder(TEST_DIR)

    class_folders = [folder for folder in ORIGINAL_DIR.iterdir() if folder.is_dir()]

    if not class_folders:
        print("У data/original немає папок з класами.")
        return

    total_train = 0
    total_val = 0
    total_test = 0

    for class_folder in class_folders:
        class_name = class_folder.name

        images = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total_count = len(images)
        train_count = int(total_count * TRAIN_RATIO)
        val_count = int(total_count * VAL_RATIO)

        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        copy_files(train_files, TRAIN_DIR / class_name)
        copy_files(val_files, VAL_DIR / class_name)
        copy_files(test_files, TEST_DIR / class_name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(f"\nКлас: {class_name}")
        print(f"Усього: {total_count}")
        print(f"Train: {len(train_files)}")
        print(f"Val: {len(val_files)}")
        print(f"Test: {len(test_files)}")

    print("\nГотово. Датасет розділено.")
    print(f"Загалом train: {total_train}")
    print(f"Загалом val: {total_val}")
    print(f"Загалом test: {total_test}")


if __name__ == "__main__":
    split_dataset()