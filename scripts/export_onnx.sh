#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-outputs/checkpoints/best.pt}
OUT=${2:-outputs/checkpoints/model.onnx}
ARCH=${3:-resnet50}
python - <<PY
import os, torch
from src.model import build_model
ckpt=os.environ.get('CKPT') or '${CKPT}'
out=os.environ.get('OUT') or '${OUT}'
arch=os.environ.get('ARCH') or '${ARCH}'
state=torch.load(ckpt, map_location='cpu')
model=build_model(arch, 2, pretrained=False)
model.load_state_dict(state['model_state']); model.eval()
dummy=torch.randn(1,3,224,224)
torch.onnx.export(model, dummy, out, opset_version=17, input_names=['input'], output_names=['probs'], dynamic_axes={'input':{0:'batch'}, 'probs':{0:'batch'}})
print('Exported ONNX to', out)
PY