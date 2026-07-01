import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model


def load_model(ckpt_path: str, arch: str):
    model = build_model(arch, 2, pretrained=False)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    return model


def preprocess_image(image_path: str, image_size: int = 224):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)

    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = load_model(args.ckpt, args.arch)
    x = preprocess_image(args.image, args.image_size)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    normal_prob = float(probs[0])
    pneumonia_prob = float(probs[1])
    prediction = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"

    print(f"Prediction: {prediction}")
    print(f"NORMAL probability: {normal_prob:.4f}")
    print(f"PNEUMONIA probability: {pneumonia_prob:.4f}")


if __name__ == "__main__":
    main()