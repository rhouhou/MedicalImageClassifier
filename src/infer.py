import torch
from model import build_model

@torch.inference_mode()
def predict(image_tensor, ckpt_path: str, arch: str = 'resnet50'):
    model = build_model(arch, 2, pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model_state']); model.eval()
    logits = model(image_tensor)
    return torch.softmax(logits, dim=1)[:, 1]
