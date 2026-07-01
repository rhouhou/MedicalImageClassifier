import os
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from .data import build_dataloaders
from .model import build_model
from .gradcam import GradCAM


def tensor_to_display_image(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img


def save_gradcam_figure(img_tensor, cam_tensor, true_label, pred_label, pred_prob, save_path):
    img = tensor_to_display_image(img_tensor)
    cam = cam_tensor.detach().cpu().numpy().squeeze()

    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = 0.6 * img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nTrue: {true_label} | Pred: {pred_label}\nP(PNEUMONIA)={pred_prob:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = cfg["paths"]

    os.makedirs(paths["figures_dir"], exist_ok=True)

    _, _, test_loader = build_dataloaders(
        paths["data_dir"],
        cfg["training"]["batch_size"],
        cfg["training"]["num_workers"],
        cfg["augment"]["image_size"],
    )

    model = build_model(
        cfg["model"]["arch"],
        cfg["model"]["num_classes"],
        cfg["model"]["pretrained"],
    ).to(device)

    ckpt_path = Path(paths["ckpt_dir"]) / "best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    gradcam = GradCAM(model, target_layer_name="layer4")

    needed = {
        "true_positive": None,
        "true_negative": None,
        "false_positive": None,
        "false_negative": None,
    }

    label_names = {0: "NORMAL", 1: "PNEUMONIA"}

    with torch.no_grad():
        pass

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        cams, probs = gradcam(images)
        preds = (probs >= 0.5).long()

        for i in range(images.size(0)):
            true_y = int(labels[i].item())
            pred_y = int(preds[i].item())
            prob_y = float(probs[i].item())

            case_name = None
            if true_y == 1 and pred_y == 1 and needed["true_positive"] is None:
                case_name = "true_positive"
            elif true_y == 0 and pred_y == 0 and needed["true_negative"] is None:
                case_name = "true_negative"
            elif true_y == 0 and pred_y == 1 and needed["false_positive"] is None:
                case_name = "false_positive"
            elif true_y == 1 and pred_y == 0 and needed["false_negative"] is None:
                case_name = "false_negative"

            if case_name is not None:
                needed[case_name] = {
                    "image": images[i].detach().cpu(),
                    "cam": cams[i].detach().cpu(),
                    "true_label": label_names[true_y],
                    "pred_label": label_names[pred_y],
                    "pred_prob": prob_y,
                }

        if all(v is not None for v in needed.values()):
            break

    gradcam.close()

    for case_name, item in needed.items():
        if item is None:
            print(f"Could not find example for: {case_name}")
            continue

        save_path = Path(paths["figures_dir"]) / f"gradcam_{case_name}.png"
        save_gradcam_figure(
            item["image"],
            item["cam"],
            item["true_label"],
            item["pred_label"],
            item["pred_prob"],
            save_path,
        )
        print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)