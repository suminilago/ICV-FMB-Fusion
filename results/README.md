# 결과 폴더 안내

트랙 A(FMB, RGB + IR, 14 클래스)의 결과 표와 그림이다. 파일은 **그 실험의 폴더 안**에 표와 그림을 함께 둔다.
한 실험의 근거가 두 폴더로 갈라지지 않도록 종류별(`tables/`, `figures/`)로 나누지 않았다.

| 문서 | 내용 |
|---|---|
| [../docs/experiment_design.md](../docs/experiment_design.md) | 가설의 근거, 검증 지표, 판정. 이 프로젝트의 논리 뼈대 |
| [report.md](report.md) | 최종 보고서. 11개 실험 표, 주요 분석, 리더보드 제출, 재현 방법 |
| [design_notes.md](design_notes.md) | 학습 로그에서만 얻은 검증 분석 |

## 폴더 지도

폴더 번호와 사이클 번호가 같다. `00`과 `06`은 사이클이 아니다.

| 폴더 | 내용 | 설계 문서 |
|---|---|---|
| [00_overview](00_overview/README.md) | 전체 run 요약, 전체 학습 곡선 | |
| [01_baseline](01_baseline/README.md) | 사이클 1. 기준선 재현, 희귀 클래스(Bicycle) | [가설 1-A, 1-B](../docs/experiment_design.md#사이클-1-기준선과-희귀-클래스) |
| [02_fusion](02_fusion/README.md) | 사이클 2. 융합 구조와 초기화 | [가설 2-A, 2-B, 2-C](../docs/experiment_design.md#사이클-2-융합-구조와-초기화) |
| [03_capacity](03_capacity/README.md) | 사이클 3. 백본 확대와 해상도 | [가설 3-A, 3-B](../docs/experiment_design.md#사이클-3-용량과-해상도) |
| [04_ir_contribution](04_ir_contribution/README.md) | 사이클 4. 최종 설정에서 IR 기여 분리 | [가설 4](../docs/experiment_design.md#사이클-4-최종-설정에서-ir-기여-분리) |
| [05_oversampling_final](05_oversampling_final/README.md) | 사이클 5. 최종 설정 샘플링, val과 test의 역전 | [가설 5](../docs/experiment_design.md#사이클-5-최종-설정에서-샘플링) |
| [06_final_model](06_final_model/README.md) | 최종 모델의 혼동, 실패 사례, TTA | |

## 파일 이름 규칙

`<측정 대상>_<비교>` 형태다. 폴더가 실험을 가리키므로 백본, 해상도 같은 접미사는 뺐다.

| 접두어 | 측정 대상 |
|---|---|
| `run_` | 실험 전체 메타 (설정, best, 수렴, epoch별 격차) |
| `classiou_` | 클래스별 IoU 비교 |
| `confusion_` | 혼동행렬, 주요 오답 |
| `failure_` | 실패 사례 |
| `miou_` | 전체 mIoU 집계 (밝기 그룹, TTA) |
| `curve_` | 학습 곡선. `curve_gap_`은 epoch별 격차 패널이 붙은 것 |
| `dist_` | 데이터 분포 |
| `qualitative_` | 예측 결과 시각 비교 |

`run_gap_<a>_vs_<b>.csv`와 `curve_gap_<a>_vs_<b>.png`의 격차는 (a − b)이고 이름은 run 폴더명에서 `fmb_`를 뺀 것이다.

## 파일 색인

| 폴더 | 파일 |
|---|---|
| 00_overview | `run_summary.csv`, `run_convergence.csv`, `curve_all_runs.png` |
| 01_baseline | `classiou_rgb_vs_sum.csv`, `classiou_rgb_vs_sum.png`, `confusion_rgb_vs_sum.png`, `confusion_class7_12_rgb_vs_sum.csv`, `curve_rgb_vs_sum.png`, `dist_train_class.csv`, `dist_class12_val_images.csv`, `qualitative_class12_gt.png`, `qualitative_class12_rgb_vs_sum.png` |
| 02_fusion | `classiou_concat_vs_adaptive.csv`, `curve_concat_vs_adaptive.png`, `run_gap_concat_res_vs_concat.csv`, `curve_gap_concat_res_vs_concat.png`, `run_gap_adaptive_vs_concat.csv`, `curve_gap_adaptive_vs_concat.png` |
| 03_capacity | `classiou_b0_vs_b2.csv`, `classiou_384_vs_480.csv`, `run_gap_b2_sum_480_vs_b2_sum.csv`, `curve_gap_b2_sum_480_vs_b2_sum.png` |
| 04_ir_contribution | `classiou_rgb_vs_sum.csv`, `classiou_dark_rgb_vs_sum.csv`, `miou_brightness_rgb_vs_sum.csv`, `classiou_ir_gain_brightness.png`, `qualitative_dark_rgb_vs_sum.png`, `run_gap_b2_sum_480_vs_b2_rgb_480.csv`, `curve_gap_b2_sum_480_vs_b2_rgb_480.png` |
| 05_oversampling_final | (README만) |
| 06_final_model | `confusion_final.png`, `confusion_final_top.csv`, `failure_final.csv`, `failure_final.png`, `miou_tta.csv` |

## 재생성

학습 로그(`runs/*/log.jsonl`, `args.json`)만으로 만들 수 있는 표와 그림은 스크립트가 다시 만든다. 데이터, 체크포인트, GPU가 필요 없다.

```bash
pip install matplotlib
python tools/analyze_runs.py              # results/ 에 출력
python tools/analyze_runs.py --out /tmp/x # 레포를 건드리지 않고 확인
```

만드는 파일: `run_summary.csv`, `run_convergence.csv`, `curve_all_runs.png`, 모든 `run_gap_*.csv`와 `curve_gap_*.png`.
로그가 있는 run 쌍만 만들고, 로그가 없는 run은 건너뛴다고 출력한다.
나머지 `classiou_`, `confusion_`, `failure_`, `miou_`, `dist_`, `qualitative_` 파일은 체크포인트로 val을 예측해서 만든 것이라 로그만으로는 다시 만들 수 없다. `02_fusion` ~ `06_final_model`의 파일을 만든 셀은 [notebooks/trackA_FMB_test.ipynb](../notebooks/trackA_FMB_test.ipynb)의 "3. 분석" 절에 있다.

클래스 번호는 FMB 순서다: 0 Road, 1 Sidewalk, 2 Building, 3 TrafficLight, 4 TrafficSign, 5 Vegetation, 6 Sky, 7 Person, 8 Car, 9 Truck, 10 Bus, 11 Motorcycle, 12 Bicycle, 13 Pole. 클래스 IoU 단위는 %다.
