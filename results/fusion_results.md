# FMB Concat and Adaptive Fusion Results

## 1. Experiment Overview

This document summarizes two RGB–infrared fusion experiments for ICV Track A. Concat joins RGB and infrared features along the channel dimension and projects them with a 1 × 1 convolution. Adaptive learns channel-wise weights for combining the two modalities.

## 2. Dataset

FMB has 14 semantic classes. The provided split uses 1,000 training images, 220 validation images, and 280 leaderboard test images. The ignore label is 255. Checkpoint selection used validation mIoU.

## 3. Experimental Settings

| Parameter | Setting |
|---|---|
| Backbone | MiT-B0 |
| Input resolution (H × W) | 384 × 512 |
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 0.00006 |
| Weight decay | 0.01 |
| Warmup epochs | 2 |
| Random seed | 0 |
| Oversampling | Disabled for both runs |

## 4. Validation Results

| Model | Best validation mIoU (%) | Difference from Sum (pp) |
|---|---:|---:|
| RGB Only | 61.53 | −3.95 |
| RGB + IR (Sum) | 65.48 | 0.00 |
| RGB + IR (Concat) | 62.47 | −3.01 |
| RGB + IR (Adaptive) | **65.35** | −0.13 |

RGB Only and Sum scores come from [the team's baseline results](baseline_results.md).

## 5. Leaderboard Test Results

| Model | Test mIoU (%) | Pixel accuracy (%) | Missing files |
|---|---:|---:|---:|
| RGB + IR (Adaptive) | 55.84 | 91.95 | 0 |
| RGB + IR (Concat) | Not submitted | — | — |

Adaptive used the checkpoint with the highest validation mIoU to generate 280 test prediction masks for Codabench (submission 1058499).

## 6. Analysis

Adaptive improved on Concat by **2.88 percentage points** in validation mIoU. Sum achieved the highest validation score in this comparison, with a 0.13-point lead over Adaptive. These scores describe the recorded runs at seed 0.

## 7. Checkpoints and Logs

The run configurations and per-epoch logs are available for [Concat](../runs/fmb_concat/) and [Adaptive](../runs/fmb_adaptive/). The corresponding `best.pth` and `last.pth` checkpoints are stored in Google Drive under `ICV_FMB/runs/`.

## 8. Next Steps

Compute class-wise IoU on the same validation split and inspect classes 7 and 12. Use the run logs to plot the validation mIoU curves for both fusion methods.
