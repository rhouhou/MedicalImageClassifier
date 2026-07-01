import yaml
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import argparse
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from .data import build_dataloaders
from .model import build_model
from .metrics import compute_metrics, plot_roc, plot_pr
from .losses import build_loss


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train(); running_loss = 0.0
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                outputs = model(images); loss = criterion(outputs, targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outputs = model(images); loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); running_loss = 0.0; probs, labels = [], []
    for images, targets in tqdm(loader, desc='Eval', leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images); loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        p = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        probs.append(p); labels.append(targets.cpu().numpy())
    return running_loss / len(loader.dataset), np.concatenate(probs), np.concatenate(labels)

def main(cfg_path: str = 'configs/default.yaml'):
    with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = cfg['paths']
    os.makedirs(paths['metrics_dir'], exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders(
        paths['data_dir'], cfg['training']['batch_size'], cfg['training']['num_workers'], cfg['augment']['image_size']
    )
    model = build_model(cfg['model']['arch'], cfg['model']['num_classes'], cfg['model']['pretrained']).to(device)
    criterion = build_loss(cfg, train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg['training']['epochs'] - cfg['training']['warmup_epochs'])) if cfg['training']['scheduler']=='cosine' else None
    scaler = GradScaler(enabled=bool(cfg['training']['mixed_precision']))
    writer = SummaryWriter(log_dir=paths['logs_dir'])

    best_val = float('inf'); patience = cfg['training']['early_stopping_patience']; bad_epochs = 0
    ckpt_dir = Path(paths['ckpt_dir']); ckpt_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(paths['figures_dir']); figures_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg['training']['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_prob, val_true = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(val_true, val_prob)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/auc', metrics['auc'], epoch)
        writer.add_scalar('Metrics/pr_auc', metrics['pr_auc'], epoch)
        writer.add_scalar('Metrics/brier', metrics['brier'], epoch)
        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f} acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}")
        if scheduler: scheduler.step()

        if val_loss < best_val:
            best_val = val_loss; bad_epochs = 0
            torch.save({'model_state': model.state_dict(), 'epoch': epoch+1}, ckpt_dir / 'best.pt')
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print('Early stopping!'); break

    state = torch.load(ckpt_dir / 'best.pt', map_location=device)
    model.load_state_dict(state['model_state'])

    test_loss, test_prob, test_true = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_true, test_prob)

    print('TEST:', test_metrics)

    plot_roc(test_true, test_prob, figures_dir.as_posix(), 'roc_test.png')
    plot_pr(test_true, test_prob, figures_dir.as_posix(), 'pr_test.png')

    metrics_path = Path(paths['metrics_dir']) / 'test_metrics.json'

    clean_metrics = {}

    for key, value in test_metrics.items():
        if isinstance(value, np.ndarray):
            clean_metrics[key] = value.tolist()
        else:
            clean_metrics[key] = float(value)

    clean_metrics['test_loss'] = float(test_loss)
    clean_metrics['model'] = cfg['model']['arch']
    clean_metrics['threshold'] = 0.5

    with open(metrics_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    
    print(f"Saved test metrics to {metrics_path}")

    test_prob_array = np.array(test_prob)
    test_true_array = np.array(test_true)

    test_pred = (test_prob_array >= 0.5).astype(int)
    cm = confusion_matrix(test_true_array, test_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['NORMAL', 'PNEUMONIA']
    )
    disp.plot(values_format='d')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrix.png', dpi=200)
    plt.close()

    print(f"Saved confusion matrix to {figures_dir / 'confusion_matrix.png'}")

    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml', 
        help='Path to YAML config file',
    )
    args = parser.parse_args()

    main(args.config)