import optuna, yaml, torch
from data import build_dataloaders
from model import build_model
from losses import build_loss
from metrics import compute_metrics
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def objective(trial, cfg):
    # Sample hyperparams
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    arch = trial.suggest_categorical('arch', ['resnet18', 'resnet50'])
    bs = trial.suggest_categorical('batch_size', [16, 32, 64])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg['paths']['data_dir'], bs, cfg['training']['num_workers'], cfg['augment']['image_size'])

    # Model + loss + opt
    model = build_model(arch, cfg['model']['num_classes'], cfg['model']['pretrained']).to(device)
    cfg_local = {**cfg, 'training': {**cfg['training'], 'lr': lr, 'weight_decay': wd}}
    criterion = build_loss(cfg_local, train_loader)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=bool(cfg['training']['mixed_precision']))

    # Quick training (few epochs)
    best_auc = 0.0
    for epoch in range(4):
        model.train()
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg['training']['mixed_precision'])):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # Val
        model.eval(); probs=[]; labels=[]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = torch.softmax(model(x), dim=1)[:,1].cpu()
                probs.append(p); labels.append(y.cpu())
        import numpy as np
        prob = torch.cat(probs).numpy(); lab = torch.cat(labels).numpy()
        m = compute_metrics(lab, prob)
        best_auc = max(best_auc, m['auc'])
        trial.report(1.0 - m['auc'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return 1.0 - best_auc  # minimize error


def main(cfg_path='configs/default.yaml'):
    with open(cfg_path,'r') as f: cfg = yaml.safe_load(f)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, cfg), n_trials=10)
    print('Best params:', study.best_trial.params)


if __name__ == '__main__':
    main()