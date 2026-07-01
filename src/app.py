import argparse
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from .model import build_model


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)


def load_model(ckpt_path: str, arch: str):
    model = build_model(arch, 2, pretrained=False)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    return model


def make_predict_fn(model):
    def predict(img: Image.Image):
        img = img.convert("RGB")
        x = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()

        return {
            "NORMAL": float(probs[0]),
            "PNEUMONIA": float(probs[1]),
        }

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/best.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        help="Model architecture used during training",
    )
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = load_model(args.ckpt, args.arch)
    predict_fn = make_predict_fn(model)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil"),
        outputs=gr.JSON(),
        title="Chest X-ray Pneumonia Classifier",
        description=(
            "Upload a chest X-ray image to get predicted probabilities for "
            "NORMAL and PNEUMONIA. This demo is for educational use only."
        ),
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()