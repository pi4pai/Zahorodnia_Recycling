import os
import json
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from database import save_prediction


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "textile_model.pth")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(num_classes):
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_recycling_recommendation(class_name):
    recommendations = {
        "30-50": {
            "method": "Механічна переробка або комбіноване сортування",
            "description": "Матеріал має помірний вміст синтетичних волокон, тому може бути придатним для механічного подрібнення, повторного використання волокон або попереднього сортування."
        },
        "50-70": {
            "method": "Комбінована або хімічна переробка",
            "description": "Матеріал має підвищений вміст синтетики, тому доцільно застосовувати методи, які враховують склад сумішевих тканин."
        },
        "70-100": {
            "method": "Хімічна переробка синтетичних волокон",
            "description": "Матеріал має високий вміст синтетики, тому найбільш доцільними є методи переробки полімерних волокон або спеціалізована утилізація."
        }
    }

    return recommendations.get(class_name, {
        "method": "Метод не визначено",
        "description": "Для цього класу немає заданої рекомендації."
    })


def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не знайдено: {image_path}")

    class_names = load_class_names()
    model = load_model(num_classes=len(class_names))
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    all_probabilities = {
        class_names[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(class_names))
    }

    recycling = get_recycling_recommendation(predicted_class)

    result = {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": all_probabilities,
        "recycling_method": recycling["method"],
        "recycling_description": recycling["description"]
    }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Вкажи шлях до зображення.")
        print("Приклад:")
        print(r'python src/predict.py "C:\Users\КВ\Desktop\test_image.jpg"')
        sys.exit()

    image_path = sys.argv[1]
    result = predict_image(image_path)
    save_prediction(result)

    print("\nРезультат класифікації:")
    print(f"Зображення: {result['image_path']}")
    print(f"Клас: {result['predicted_class']}")
    print(f"Впевненість: {result['confidence']}%")

    print("\nЙмовірності по класах:")
    for class_name, prob in result["probabilities"].items():
        print(f"{class_name}: {prob}%")

    print("\nРекомендований спосіб вторинної переробки:")
    print(result["recycling_method"])
    print(result["recycling_description"])