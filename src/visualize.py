from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model import build_model
from gradcam import GradCAM


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35):
    heatmap = plt.cm.jet(cam.squeeze())[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def run(image_path: str, ckpt_path: str, arch: str = 'resnet50', out_path: str = 'outputs/figures/gradcam_example.png'):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil).astype('float32') / 255.0
    x = preprocess(img_pil).unsqueeze(0)

    model = build_model(arch, 2, pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    cam_engine = GradCAM(model, target_layer_name='layer4')

    cam, prob_pos = cam_engine(x)
    overlay = overlay_cam(img_np / max(1.0, img_np.max()), cam.squeeze().numpy())

    plt.figure(); plt.imshow(overlay); plt.axis('off')
    plt.title(f'Grad-CAM | P(pneumonia)={prob_pos.item():.3f}')
    plt.savefig(out_path, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--out', default='outputs/figures/gradcam_example.png')
    args = parser.parse_args()
    run(args.image, args.ckpt, args.arch, args.out)