# 학습 로그 기반 검증 분석

`runs/*/log.jsonl`과 클래스별 IoU CSV에서 얻은 관찰만 모은 문서다. 해석과 판정은 [실험 설계 문서](../docs/experiment_design.md)에 있다.
각 절의 제목에 어느 사이클을 검증하는 분석인지 적었다. epoch는 1부터 센 값이고, 표의 수치는 `python tools/analyze_runs.py`로 다시 만들 수 있다.

## 1. 수렴 (전 사이클)

고정한 30 epoch가 수렴점인지 본다. 11개 run 전체 기준이다. [run_convergence.csv](00_overview/run_convergence.csv)

| run | best epoch | epoch 20 | 마지막 epoch | 상승폭 |
|---|---:|---:|---:|---:|
| `fmb_rgb` | 29 | 59.93 | 61.47 | +1.54 |
| `fmb_sum` | 30 | 64.19 | 65.48 | +1.29 |
| `fmb_sum_oversample10` | 29 | 66.10 | 67.91 | +1.81 |
| `fmb_concat` | 28 | 60.24 | 62.33 | +2.09 |
| `fmb_adaptive` | 29 | 63.22 | 65.30 | +2.08 |
| `fmb_concat_res` | 27 | 63.65 | 64.89 | +1.24 |
| `fmb_early` | 30 | 59.90 | 62.19 | +2.29 |
| `fmb_b2_sum` | 29 | 68.51 | 72.38 | +3.87 |
| `fmb_b2_sum_480` | 30 | 70.69 | 73.65 | +2.95 |
| `fmb_b2_rgb_480` | 30 | 67.33 | 70.99 | +3.66 |
| `fmb_b2_sum_480_oversample10` | 30 | 72.72 | 74.46 | +1.74 |

- 11개 run 모두 epoch 20에서 마지막까지 +1.2pp 이상 올랐다 (최소 `fmb_concat_res` +1.24).
- best epoch는 27~30에 몰려 있고, `fmb_sum`, `fmb_early`, `fmb_b2_sum_480`, `fmb_b2_rgb_480`, `fmb_b2_sum_480_oversample10`은 마지막 epoch가 best다.

## 2. 학습 초반 격차 (사이클 2, 가설 2-B)

(Concat_res − Concat)과 (Adaptive − Concat)의 epoch별 val mIoU 격차(pp)다.
[run_gap_concat_res_vs_concat.csv](02_fusion/run_gap_concat_res_vs_concat.csv), [run_gap_adaptive_vs_concat.csv](02_fusion/run_gap_adaptive_vs_concat.csv)

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Concat_res − Concat | +1.46 | +6.31 | +6.51 | **+9.74** | +6.68 | +6.27 | +3.14 | +3.41 | +2.56 |
| Adaptive − Concat | +1.59 | +5.16 | +7.08 | **+9.30** | +6.29 | +6.24 | +4.73 | +2.98 | +2.97 |

- 두 격차 모두 ep4에서 최대이고, 30 epoch 내내 부호가 바뀌지 않는다 (최소 +1.46, +1.59).
- ep10 이후 격차가 3pp 안팎으로 줄지만 단조 감소는 아니다 (Concat_res − Concat은 ep10 +3.14, ep20 +3.41).
- 그림: [curve_gap_concat_res_vs_concat.png](02_fusion/curve_gap_concat_res_vs_concat.png), [curve_gap_adaptive_vs_concat.png](02_fusion/curve_gap_adaptive_vs_concat.png)

## 3. Bus 교차 패턴 (사이클 3, 4)

Bus(10)의 IoU 변화가 바꾼 요소에 따라 방향이 갈린다.

| 바꾼 요소 | 사이클 | Bus 변화 (pp) | 파일 |
|---|---|---:|---|
| 용량 확대 (B0 → B2) | 3-A | **+7.65** | [classiou_b0_vs_b2.csv](03_capacity/classiou_b0_vs_b2.csv) |
| IR 추가 (RGB 단독 → Sum) | 4 | **−2.60** | [classiou_rgb_vs_sum.csv](04_ir_contribution/classiou_rgb_vs_sum.csv) |
| 해상도 상승 (384×512 → 480×640) | 3-B | **−3.33** | [classiou_384_vs_480.csv](03_capacity/classiou_384_vs_480.csv) |

- 같은 Bus가 용량 확대에서는 가장 크게 오른 축에 들고, IR 추가와 해상도 상승에서는 내린다.
- 원인은 확인하지 못했다. 단일 seed라 변동일 가능성이 있다.

## 4. IR 기여의 epoch별 추이 (사이클 4)

(Sum − RGB 단독)의 epoch별 val mIoU 격차(pp), B2 480×640. [run_gap_b2_sum_480_vs_b2_rgb_480.csv](04_ir_contribution/run_gap_b2_sum_480_vs_b2_rgb_480.csv)

| epoch | 1 | 5 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| Sum − RGB 단독 | −3.63 | +2.07 | +2.41 | **+3.36** | +2.66 |

- 격차가 음수인 epoch는 ep1(−3.63)과 ep6(−2.33)뿐이고 이후는 양수다. 최대는 ep20(+3.36)이다.
- 그림: [curve_gap_b2_sum_480_vs_b2_rgb_480.png](04_ir_contribution/curve_gap_b2_sum_480_vs_b2_rgb_480.png)

## 5. 해상도 효과의 epoch별 추이 (사이클 3, 가설 3-B)

(480×640 − 384×512)의 epoch별 val mIoU 격차(pp), B2 Sum. [run_gap_b2_sum_480_vs_b2_sum.csv](03_capacity/run_gap_b2_sum_480_vs_b2_sum.csv)

| epoch | 1 | 5 | 10 | 20 | 24 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| 480×640 − 384×512 | −0.86 | +1.83 | +0.67 | +2.18 | **+2.75** | +1.27 |

- 음수인 epoch는 ep1, 2, 6이고 최대는 ep24(+2.75)다.
- best epoch 기준 차이(+0.91)는 마지막 epoch 기준(+1.27)과 다르다. #8은 ep29, #9는 ep30이 best다.
- 그림: [curve_gap_b2_sum_480_vs_b2_sum.png](03_capacity/curve_gap_b2_sum_480_vs_b2_sum.png)
