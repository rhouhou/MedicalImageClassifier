import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.softmax(logits, dim=1)[range(len(targets)), targets]
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


def build_loss(cfg, train_loader):
    if cfg["training"].get("use_focal_loss", False):
        return FocalLoss()
    if cfg["training"].get("class_weighted_loss", False):
        counts = torch.zeros(2)
        for _, y in train_loader:
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
        weights = (counts.sum() / (2 * counts)).to(torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()