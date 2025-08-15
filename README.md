# Chest X-ray Pneumonia Classifier (ResNet50/timm + Grad-CAM + PR-AUC + Docker)

A complete, hiring-manager-friendly medical imaging project:

- Transfer learning with ResNet/timm backbones
- Robust training loop with early stopping, mixed precision, cosine LR
- Explainability (Grad-CAM + Integrated Gradients)
- Imbalance-aware loss (class weights or Focal Loss)
- Metrics: Accuracy, ROC-AUC, PR-AUC, Brier score + curves
- Optuna hyperparameter tuning
- ONNX export, Dockerfile, Makefile, CI, unit tests

## Dataset

Place the dataset under `data/chest_xray/` with the following structure:
data/chest_xray/
train/
NORMAL/
PNEUMONIA/
val/
NORMAL/
PNEUMONIA/
test/
NORMAL/
PNEUMONIA/

## Install & Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py  # trains and evaluates; saves ROC/PR curves

Grad-CAM on a sample image:
python src/visualize.py --image path/to/test_image.jpg --ckpt outputs/checkpoints/best.pt

Launch the demo app (Gradio):
python src/app.py --ckpt outputs/checkpoints/best.pt

Results


Artifacts in outputs/:

outputs/checkpoints/best.pt – best checkpoint

outputs/figures/roc_test.png, pr_test.png – curves

outputs/figures/gradcam_example.png – explainability

Advanced Features

Optuna tuning: ./scripts/run_tune.sh (or python src/tune.py)

Export ONNX: ./scripts/export_onnx.sh

Docker: docker build -t cxr:latest . && docker run --gpus all cxr:latest

CI: Lint + tests on push/PR via GitHub Actions

Ethical Notes

This is not a medical device. See MODEL_CARD.md for limitations and considerations.
```
