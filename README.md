# Hazelnut Anomaly Detection

Detecting defects in hazelnuts for industrial quality control, using the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. The model learns only from normal (defect-free) images, then flags anything that looks different. It also shows where the defect is, not just whether one exists.

## Approach

- Used a pretrained ResNet18 as a feature extractor (no training on defects).
- **Baseline:** each image becomes one feature vector, scored by distance to the nearest normal example.
- **Localization:** kept ResNet's spatial grid so each region gets its own anomaly score, turned into a heatmap showing the defect location.
- **Improvement:** fused features from two ResNet layers for sharper, more accurate detection.

## Results

| Method | Image-level AUC |
|---|---|
| Baseline (whole-image) | 0.96 |
| Multi-layer fusion | 0.997 |

At the chosen threshold the fused model reaches about 97% accuracy. The heatmaps correctly localize defects (cracks, holes, cuts, print errors) while staying quiet on clean nuts.

## Run

```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
jupyter notebook hazelnut_anomaly_detection.ipynb
```

Dataset: download the `hazelnut` category from MVTec AD and place it in the project folder.
