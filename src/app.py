import argparse
import torch
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import transforms
from model import build_model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])


def load_model(ckpt, arch):
    m = build_model(arch, 2, pretrained=False)
    state = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(state['model_state']); m.eval()
    return m


def predict(img: Image.Image, ckpt: str, arch: str):
    model = load_model(ckpt, arch)
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].numpy()
    return {"NORMAL": float(p[0]), "PNEUMONIA": float(p[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    parser.add_argument('--arch', default='resnet50')
    args = parser.parse_args()

    demo = gr.Interface(
        fn=lambda img: predict(img, args.ckpt, args.arch),
        inputs=gr.Image(type='pil'),
        outputs=gr.Label(num_top_classes=2),
        title='Chest X-ray Pneumonia Classifier',
        description='Upload a chest X-ray image to get probabilities.'
    )
    demo.launch()


if __name__ == '__main__':
    main()