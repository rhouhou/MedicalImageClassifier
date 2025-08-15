from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, brier_score_loss
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(np.trapz(pr_p, pr_r))
    brier = brier_score_loss(y_true, y_prob)
    return {"accuracy": acc, "auc": auc, "pr_auc": pr_auc, "brier": brier, "confusion_matrix": cm}


def _savefig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_roc(y_true, y_prob, out_dir: str, name: str = 'roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(); plt.plot(fpr, tpr, label='ROC'); plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend()
    out_path = Path(out_dir) / name
    _savefig(fig, out_path)
    return str(out_path)


def plot_pr(y_true, y_prob, out_dir: str, name: str = 'pr_curve.png'):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure(); plt.plot(r, p, label='PR')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend()
    out_path = Path(out_dir) / name
    _savefig(fig, out_path)
    return str(out_path)