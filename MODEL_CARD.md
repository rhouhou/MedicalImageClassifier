# Model Card: Chest X-ray Pneumonia Classifier

## Model Summary

This model is a deep learning image classifier for binary chest X-ray classification.

It predicts one of two classes:

* `NORMAL`
* `PNEUMONIA`

The model is part of the `MedicalImageClassifier` repository, which demonstrates an end-to-end medical image classification workflow using PyTorch, transfer learning, evaluation metrics, Grad-CAM explainability, and deployment-oriented tooling.

> **Important:** This model is for educational and research demonstration only. It is not intended for clinical diagnosis, treatment decisions, patient triage, or deployment in real healthcare settings.

---

## Model Details

| Field                  | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| Model type             | Binary image classifier                                               |
| Task                   | Chest X-ray pneumonia classification                                  |
| Classes                | `NORMAL`, `PNEUMONIA`                                                 |
| Framework              | PyTorch                                                               |
| Backbone               | ResNet18 for CPU smoke test; configurable for ResNet50/timm backbones |
| Input type             | Chest X-ray image                                                     |
| Input size             | Configurable; smoke-test configuration uses 224 × 224                 |
| Output                 | Class probabilities                                                   |
| Training configuration | YAML-based configuration                                              |
| Explainability         | Grad-CAM; Integrated Gradients support can be added or extended       |
| Deployment support     | ONNX export, Docker, Gradio app                                       |

---

## Intended Use

This model is intended for:

* educational demonstration of medical image classification
* portfolio demonstration of computer vision and medical AI engineering skills
* experimentation with PyTorch transfer learning
* evaluation of image classification metrics
* explainability demonstrations using Grad-CAM
* reproducible ML pipeline development

---

## Not Intended For

This model is not intended for:

* clinical diagnosis
* clinical decision support
* patient triage
* medical treatment planning
* deployment in hospitals or healthcare systems
* use as a medical device
* replacing radiologists or medical professionals

The model has not been clinically validated and should not be used in real medical workflows.

---

## Dataset

This project uses the public [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) dataset available on Kaggle.

The dataset contains chest X-ray images categorized into two classes:

* `NORMAL`
* `PNEUMONIA`

The dataset is organized into training, validation, and test folders. It is not included in this repository. Users must download it separately from Kaggle and follow the dataset’s licensing and usage terms.

Expected local structure:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## Training Procedure

The model is trained using a configurable PyTorch pipeline.

The pipeline includes:

* image loading from folder structure
* image resizing and augmentation
* transfer learning with pretrained convolutional backbones
* class-imbalance-aware loss
* AdamW optimization
* cosine learning-rate scheduling
* early stopping
* validation monitoring
* best-checkpoint saving
* final test-set evaluation

A CPU-compatible smoke-test configuration is provided:

```bash
python -m src.train --config configs/smoke_cpu.yaml
```

A fuller training configuration can be run with:

```bash
python -m src.train --config configs/default.yaml
```

---

## Evaluation

The model is evaluated on the test set using:

| Metric           | Purpose                                                            |
| ---------------- | ------------------------------------------------------------------ |
| Accuracy         | Overall classification correctness                                 |
| ROC-AUC          | Ranking ability across thresholds                                  |
| PR-AUC           | Precision-recall performance, useful for imbalanced classification |
| Brier Score      | Probability calibration quality                                    |
| Confusion Matrix | Error analysis for false positives and false negatives             |

---

## Current Smoke-Test Results

The following results are from a local CPU smoke-test run using `ResNet18`.

| Metric      | Value |
| ----------- | ----: |
| Accuracy    | 0.840 |
| ROC-AUC     | 0.941 |
| PR-AUC      | 0.962 |
| Brier Score | 0.124 |

Confusion matrix:

```text
[[145,  89],
 [ 11, 379]]
```

These results are useful for verifying that the full pipeline works, but they should not be interpreted as final model performance.

A stronger final evaluation should include:

* full GPU training
* larger number of epochs
* repeated runs with multiple random seeds
* confidence intervals
* external validation data
* subgroup analysis if metadata is available
* calibration analysis

---

## Explainability

The repository includes Grad-CAM visualizations for model inspection.

Generated examples include:

* true positive
* true negative
* false positive
* false negative

Grad-CAM can help identify whether the model focuses on plausible image regions, but it has important limitations. A heatmap should not be interpreted as proof that the model has learned clinically valid reasoning.

Explainability outputs are intended for:

* debugging
* error analysis
* model inspection
* portfolio demonstration

They are not sufficient for clinical validation.

---

## Limitations

This model has several important limitations:

1. **No clinical validation**
   The model has not been evaluated in a clinical environment and has not been reviewed for medical use.

2. **Dataset bias**
   Public chest X-ray datasets may not represent all populations, hospitals, imaging devices, disease severities, or acquisition protocols.

3. **Domain shift**
   Performance may decrease when applied to images from different hospitals, scanners, preprocessing pipelines, or patient populations.

4. **Label noise**
   Public datasets may contain imperfect or inconsistent labels.

5. **Shortcut learning**
   The model may learn dataset-specific artifacts, image markers, borders, or acquisition patterns instead of disease-relevant features.

6. **Calibration uncertainty**
   Probability outputs may not be well calibrated without additional calibration methods.

7. **Limited evaluation**
   The current reported result is from a smoke-test run and should not be treated as a final benchmark.

8. **Explainability limitations**
   Grad-CAM visualizations are approximate and should not be treated as clinical evidence.

---

## Ethical Considerations

Medical AI systems can cause harm if used without proper validation. Potential risks include:

* false reassurance from false negatives
* unnecessary concern or intervention from false positives
* degraded performance on underrepresented populations
* overreliance on automated predictions
* misinterpretation of explainability heatmaps
* deployment outside the intended context

Any real clinical system would require:

* expert clinical review
* external validation
* prospective evaluation
* fairness and bias assessment
* uncertainty estimation
* regulatory review
* monitoring after deployment

---

## Responsible Use Statement

This model should only be used for educational, research, and portfolio demonstration purposes.

It must not be used to diagnose pneumonia, recommend treatment, triage patients, or make healthcare decisions.

---

## Recommended Future Work

To improve the project, future work should include:

* full training on GPU with a stronger backbone such as ResNet50 or EfficientNet
* comparison across multiple backbones
* repeated experiments with multiple random seeds
* threshold tuning for sensitivity and specificity
* calibration plots and calibration correction
* external validation on a different chest X-ray dataset
* stronger error analysis
* Integrated Gradients examples
* model uncertainty estimation
* subgroup analysis if demographic or acquisition metadata is available
* automated data-quality checks
* lightweight multi-agent ML audit workflow

---

## Reproducibility

The repository includes:

* YAML configuration files
* deterministic seed setting where possible
* saved metrics
* saved figures
* unit tests
* Docker support
* ONNX export support
* model documentation

The CPU smoke-test configuration is intended to let users verify that the pipeline runs locally without requiring a GPU.

---

## Contact and Repository

Repository: `MedicalImageClassifier`

Author: Dr. Rola Houhou
