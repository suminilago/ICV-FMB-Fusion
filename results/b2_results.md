# FMB Backbone Scaling: MiT-B0 → MiT-B2

## 1. Objective

Evaluate whether a larger encoder improves RGB–infrared fusion segmentation,
keeping the fusion method (Sum) and all other settings fixed.

## 2. Experimental Settings

| Parameter | Setting |
|---|---|
| Fusion | Sum |
| Backbone | MiT-B2 (both RGB and IR encoders, ImageNet-pretrained `nvidia/mit-b2`) |
| Parameters | 51.55M (rule: ≤ ~52M) |
| Input Resolution | 384 × 512 |
| Epochs | 30 |
| Batch Size | 8 |
| Learning Rate | 0.00006 |
| Weight Decay | 0.01 |
| Warmup Epochs | 2 |
| Random Seed | 0 |
| Environment | Google Colab, Tesla T4 (~2.0 min/epoch, 61.6 min total) |

Only the backbone differs from the B0 Sum baseline.

## 3. Results

| Model | Params | Best Epoch | Val mIoU (%) | Test mIoU (%) |
|---|---:|---:|---:|---:|
| Sum, MiT-B0 | 7.04M | 30 | 65.48 | — |
| Adaptive, MiT-B0 | 7.23M | 28 | 65.35 | 55.84 |
| **Sum, MiT-B2** | **51.55M** | **28** | **72.74** | **63.90** |

- Val mIoU: +7.26 pp over Sum (MiT-B0).
- Test mIoU (Codabench, submission 1060429): +8.06 pp over Adaptive (MiT-B0),
  +7.60 pp over the instructor Sum baseline (56.30).
- Best epoch is reported 1-indexed (log.jsonl epoch 27 → 28).
- No TTA was used for this submission.

## 4. Analysis

Scaling the encoder gave a much larger gain than changing the fusion module
(Concat/Adaptive vs Sum at B0: −3.01 / −0.13 pp). The val–test gap
(~9 pp) is similar to the B0 runs, so validation mIoU remains a consistent
model-selection signal.

## 5. Reproduction

```bash
python train.py --data_root data/FMB_prepared --fusion sum \
  --backbone nvidia/mit-b2 --epochs 30 --seed 0 --out runs/fmb_b2_sum
python predict.py --data_root data/FMB_prepared \
  --ckpt runs/fmb_b2_sum/best.pth --split test --out submission_b2_sum
```

The data split was verified to be identical to the original
`tools/prepare_fmb.py` output (train 1,000 / val 220 / test 280 file names match).

## 6. Checkpoints and Logs

- Config and per-epoch log: [`runs/fmb_b2_sum/`](../runs/fmb_b2_sum/)
- `best.pth` / `last.pth` are stored in Google Drive (`ICV_FMB/runs/fmb_b2_sum/`).
