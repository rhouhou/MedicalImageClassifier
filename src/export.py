import argparse
from pathlib import Path

import torch

from .model import build_model


def export_onnx(
    ckpt_path: str,
    out_path: str = "outputs/checkpoints/model.onnx",
    arch: str = "resnet18",
    image_size: int = 224,
):
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = torch.load(ckpt_path, map_location="cpu")

    model = build_model(arch, 2, pretrained=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX model to: {out_path}")
    return out_path.as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="outputs/checkpoints/best.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--out",
        default="outputs/checkpoints/model.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        help="Model architecture used during training",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )

    args = parser.parse_args()

    export_onnx(
        ckpt_path=args.ckpt,
        out_path=args.out,
        arch=args.arch,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()