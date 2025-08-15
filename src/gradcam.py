from typing import List
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer_name: str = 'layer4'):
        self.model = model
        self.model.eval()
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.activations = None
        self.gradients = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def __call__(self, images: torch.Tensor, target_class: int = None):
        images.requires_grad = True
        logits = self.model(images)
        if target_class is None:
            target_class = logits.argmax(dim=1)
        loss = logits[range(logits.shape[0]), target_class]
        self.model.zero_grad()
        loss.backward(torch.ones_like(loss))
        cam = self._generate()
        return cam, logits.softmax(dim=1)[:, 1]

    def close(self):
        for h in self.hook_handles:
            h.remove()