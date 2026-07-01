import torch
from torchvision import transforms


def tta_predictions(model, image, n=4):
    """Simple test-time augmentation: horizontal flips."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(n):
            x = image.clone()
            if i % 2 == 1:
                x = torch.flip(x, dims=[-1])
            p = torch.softmax(model(x), dim=1)[:, 1]
            preds.append(p)
    return torch.stack(preds, dim=0).mean(dim=0)