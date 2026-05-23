
import sys
import json
import argparse
import traceback

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = 6

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def print_json(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = CNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        classes = checkpoint.get("classes", [])
        config = checkpoint.get("config", {})
    else:
        model.load_state_dict(checkpoint)
        classes = []
        config = {}

    model.eval()

    input_size = config.get("input_size", 128)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    return model, classes, config, device, transform


def predict_image(model, classes, device, transform, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

    if classes and predicted_index < len(classes):
        predicted_class = classes[predicted_index]
    else:
        predicted_class = f"classe_indice_{predicted_index}_sem_nome_no_checkpoint"

    top_k = min(3, probabilities.shape[0])

    top_values, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions = []

    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        index = int(index)

        if classes and index < len(classes):
            class_name = classes[index]
        else:
            class_name = f"classe_indice_{index}_sem_nome_no_checkpoint"

        top_predictions.append({
            "class": class_name,
            "index": index,
            "confidence": float(value)
        })

    return {
        "predictedClass": predicted_class,
        "predictedIndex": predicted_index,
        "confidence": confidence,
        "topPredictions": top_predictions
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        model, classes, config, device, transform = load_model(args.model)

        print_json({
            "status": "ready",
            "message": "Modelo carregado com sucesso.",
            "device": device,
            "classes": classes,
            "config": config
        })

    except Exception as e:
        print_json({
            "status": "startup_error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })
        sys.exit(1)

    for line in sys.stdin:
        try:
            request = json.loads(line)

            request_id = request.get("id")
            image_path = request.get("imagePath")

            if not image_path:
                raise ValueError("Campo imagePath não foi informado.")

            result = predict_image(
                model=model,
                classes=classes,
                device=device,
                transform=transform,
                image_path=image_path
            )

            print_json({
                "id": request_id,
                "ok": True,
                "result": result
            })

        except Exception as e:
            print_json({
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })


if __name__ == "__main__":
    main()
