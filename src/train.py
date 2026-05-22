import os
import json
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "textile_model.pth")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.csv")
ACCURACY_LOSS_GRAPH_PATH = os.path.join(RESULTS_DIR, "accuracy_loss_graph.png")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


BATCH_SIZE = 16
NUM_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")


data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

    "val": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

    "test": transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


image_datasets = {
    "train": datasets.ImageFolder(TRAIN_DIR, transform=data_transforms["train"]),
    "val": datasets.ImageFolder(VAL_DIR, transform=data_transforms["val"]),
    "test": datasets.ImageFolder(TEST_DIR, transform=data_transforms["test"])
}


dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
    "val": DataLoader(image_datasets["val"], batch_size=BATCH_SIZE, shuffle=False),
    "test": DataLoader(image_datasets["test"], batch_size=BATCH_SIZE, shuffle=False)
}


dataset_sizes = {
    "train": len(image_datasets["train"]),
    "val": len(image_datasets["val"]),
    "test": len(image_datasets["test"])
}


class_names = image_datasets["train"].classes
num_classes = len(class_names)

print("Класи:", class_names)
print("Кількість класів:", num_classes)
print("Розмір train:", dataset_sizes["train"])
print("Розмір val:", dataset_sizes["val"])
print("Розмір test:", dataset_sizes["test"])


with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)


model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_model(model, criterion, optimizer, num_epochs):
    start_time = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_without_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        print(f"\nЕпоха {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["epoch"].append(epoch + 1)
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

                # 🔥 тут логіка Early Stopping
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

        # 🔥 перевірка після кожної епохи
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: точність не покращувалась {PATIENCE} епох.")
            break

    total_time = time.time() - start_time
    print(f"\nНавчання завершено за {total_time // 60:.0f} хв {total_time % 60:.0f} сек")
    print(f"Найкраща val accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_weights)

    return model, history


def save_training_graph(history):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Results")
    plt.legend()
    plt.grid(True)
    plt.savefig(ACCURACY_LOSS_GRAPH_PATH)
    plt.close()

    print(f"Графік навчання збережено: {ACCURACY_LOSS_GRAPH_PATH}")


def evaluate_model(model):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nClassification report:")
    print(report)

    print(f"Confusion matrix збережено: {CONFUSION_MATRIX_PATH}")
    print(f"Classification report збережено: {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    trained_model, history = train_model(
        model,
        criterion,
        optimizer,
        NUM_EPOCHS
    )

    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"\nМодель збережено: {MODEL_PATH}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(TRAINING_HISTORY_PATH, index=False)
    print(f"Історію навчання збережено: {TRAINING_HISTORY_PATH}")

    save_training_graph(history)

    evaluate_model(trained_model)

    print("\nГотово. Модель навчена і збережена.")