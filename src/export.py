import torch
from model import build_model


def export_onnx(ckpt_path: str, out_path: str = 'outputs/checkpoints/model.onnx', arch: str = 'resnet50'):
    state = torch.load(ckpt_path, map_location='cpu')
    model = build_model(arch, 2, pretrained=False)
    model.load_state_dict(state['model_state']); model.eval()
    dummy = torch.randn(1,3,224,224)
    torch.onnx.export(
        model, dummy, out_path, opset_version=17,
        input_names=['input'], output_names=['probs'],
        dynamic_axes={'input':{0:'batch'}, 'probs':{0:'batch'}}
    )
    return out_path