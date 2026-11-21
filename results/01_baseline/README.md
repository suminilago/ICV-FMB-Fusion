# FMB Validation Analysis

## 1. Overview

This directory contains validation-set analyses for the
RGB-only and RGB+IR Sum Fusion baselines.

All analyses use the fixed FMB training/validation split:
- Train: 1,000 images
- Validation: 220 images
- Classes: 14
- Ignore label: 255

## 2. Overall Results

| Model | Validation mIoU |
|---|---:|
| RGB Only | 61.53% |
| RGB+IR Sum | 65.48% |
| Sum + Class 12 Oversampling (10%) | 67.97% |

Sum Fusion improved mIoU by 3.95 percentage points
over RGB Only.

Oversampling improved mIoU by a further 2.49 percentage
points over Sum Fusion.

## 3. Class-wise IoU Analysis

Source: `class_iou_comparison.csv`

The RGB vs Sum comparison shows:
- 12 classes improved.
- Class 7 had the largest improvement (+19.26 pp).
- Class 12 remained at 0% IoU.
- Class 4 decreased by 0.39 pp.

These results indicate that multimodal fusion improves
overall segmentation performance but does not improve
every class.

## 4. Rare-Class Distribution

Source: `train_class_distribution.csv`

Class 12 is severely underrepresented:
- Present in 10 of 1,000 training images.
- Training pixel ratio: 0.00173%.
- Present in 2 of 220 validation images.
- Total Class 12 validation pixels: 994.

This motivates investigating class-aware oversampling.

## 5. Confusion Analysis

Source: `class_7_12_confusion_analysis.csv`

For ground-truth Class 12:
- RGB Only predicts Class 8 for 43.46% of its pixels.
- Sum Fusion predicts Class 8 for 68.41% of its pixels.
- Both models achieve 0% Class 12 IoU.

The results suggest that improved overall fusion performance
does not necessarily translate into rare-class recognition.

## 6. Oversampling Experiment

The 10% Class 12 oversampling experiment achieved:
- Validation mIoU: 67.97%.
- Class 12 IoU: 43.40%.

The improvement is promising, but Class 12 appears in only
two validation images. Generalization beyond these examples
is not established.

A full class-wise Sum vs Oversampling comparison remains
pending and will be added after evaluation.

## 7. Analysis Files

| File | Description |
|---|---|
| class_iou_comparison.csv | RGB vs Sum per-class IoU |
| class_iou_improvement.png | Per-class IoU visualization |
| confusion_matrix_comparison.png | RGB vs Sum confusion matrices |
| class_7_12_confusion_analysis.csv | Main errors for Classes 7 and 12 |
| train_class_distribution.csv | Training-set class distribution |
| class_12_val_images.csv | Validation images containing Class 12 |
| class_12_gt_visualization.png | Class 12 ground-truth visualization |
| class_12_prediction_comparison.png | Class 12 prediction comparison |

## 8. Limitations and Next Steps

- Evaluate per-class IoU for Sum vs Oversampling.
- Analyze improvements and regressions in other classes.
- Inspect qualitative failure cases.
- Consider sampling-ratio ablations if compute is available.

No test labels were used for this validation analysis.
