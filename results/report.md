# Track A: FMB RGB–IR Fusion Segmentation

- Dataset: FMB RGB+IR, 14 classes
- Leaderboard account: `reduce_l`
- Final setup: RGB·IR MiT-B2 encoders, stage-wise Sum fusion, 480×640 input, 51.55M parameters
- Final performance: val mIoU 73.65 / test mIoU 66.13, Codabench submission 1061666
- Experiment rationale: [experiment design](../docs/experiment_design.md)

## 1. 실험 조건

- Fixed settings: seed 0, batch 8, AdamW lr 6e-5, decoder lr ×10, weight decay 0.01, 2-epoch warm-up + poly schedule, 30 epochs, identical augmentation
- Control: one changed factor per comparison
- Model selection: val mIoU
- Metric: mean IoU over 14 classes

## 2. 전체 실험 결과

| # | Experiment | Backbone | Input | Fusion | Params | Best epoch | Val mIoU | Test mIoU | Logs |
|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | RGB 단독 | B0 | 384×512 | — | 3.7M | 29 | 61.53 | | `runs/fmb_rgb` |
| 2 | Sum | B0 | 384×512 | 단계별 합 | 7.04M | 30 | 65.48 | | `runs/fmb_sum` |
| 3 | Sum + Oversampling | B0 | 384×512 | 단계별 합 | 7.04M | 29 | 67.97 | | `runs/fmb_sum_oversample10` |
| 4 | Concat | B0 | 384×512 | concat → 1×1 conv | 7.23M | 28 | 62.47 | | `runs/fmb_concat` |
| 5 | Adaptive | B0 | 384×512 | 채널별 게이트 | 7.23M | 29 | 65.35 | 55.84 | `runs/fmb_adaptive` |
| 6 | Concat_res | B0 | 384×512 | sum + concat 잔차, 0 초기화 | 7.33M | 27 | 64.95 | | `runs/fmb_concat_res` |
| 7 | Early | B0 | 384×512 | 입력 단계 합, 인코더 1개 | 3.72M | 30 | 62.19 | | `runs/fmb_early` |
| 8 | Sum | B2 | 384×512 | 단계별 합 | 51.55M | 29 | 72.74 | 63.90 | `runs/fmb_b2_sum` |
| 9 | Sum (최종) | B2 | 480×640 | 단계별 합 | 51.55M | 30 | 73.65 | 66.13 | `runs/fmb_b2_sum_480` |
| 10 | RGB 단독 | B2 | 480×640 | — | 27.36M | 30 | 70.99 | | `runs/fmb_b2_rgb_480` |
| 11 | Sum + Oversampling | B2 | 480×640 | 단계별 합 | 51.55M | 30 | 74.46 | 65.64 | `runs/fmb_b2_sum_480_oversample10` |
| Ref. | Instructor baseline RGB / Sum | B0 | 384×512 | | | | | 52.77 / 56.30 | |

- Best epoch 표기: `log.jsonl`의 epoch + 1
- 전체 11개 run: 30 epoch 기록 완료
- #6, #9: Colab 연결 종료 후 동일 명령으로 `last.pth` 재개
- 학습 곡선과 run 요약: [00_overview](00_overview/README.md)

## 3. 주요 분석

### 3.1 IR 기여 — 사이클 4, B2 480×640, #10 vs #9

밝기 그룹: val 220장을 RGB 평균 밝기 기준 73 / 73 / 74장으로 분할.

| 밝기 그룹 (0–255) | 이미지 | RGB 단독 | RGB + IR | IR 기여 |
|---|---:|---:|---:|---:|
| 어두움 (39–98) | 73 | 61.91 | 66.55 | +4.64 |
| 중간 (98–149) | 73 | 73.06 | 75.86 | +2.80 |
| 밝음 (149–162) | 74 | 73.94 | 74.83 | +0.89 |

- 전체 IR 기여: 70.99 → 73.65, +2.66pp. 어두운 그룹에서 가장 큰 차이.
- 클래스별: Person +18.79, Bicycle +7.81, Motorcycle +3.04, Pole +2.66, Car +2.27. TrafficSign −0.75, Bus −2.60.
- 어두운 장면 Person: 42.32 → 65.82, +23.50pp
- epoch별 IR 기여: ep1 −3.63, ep5 +2.07, ep10 +2.41, ep20 +3.36, ep30 +2.66
- B0 결과: RGB 61.53 → Sum 65.48, +3.95pp. 동일 방향.
- Files: [04_ir_contribution](04_ir_contribution/README.md)

### 3.2 융합 위치·구조와 초기화 — 사이클 2, B0

- Sum 대비: Adaptive −0.13, Concat_res −0.53, Concat −3.01, Early −3.29. Sum을 넘은 학습형 융합 없음.
- Early의 RGB 단독 대비 변화: +0.66pp
- Concat_res − Concat: ep1 +1.46, ep4 +9.74 최대, ep10 +3.14, ep20 +3.41, ep30 +2.56
- Adaptive − Concat: ep4 +9.30 최대
- 클래스별 결과: Bicycle 제외 13개 클래스에서 Adaptive가 Concat보다 높은 IoU
- 해석 범위: 0.5pp 이내 차이의 우열 판단 제외, B0에서만 융합 비교
- Files: [02_fusion](02_fusion/README.md)

### 3.3 백본과 해상도 — 사이클 3, #2 → #8 → #9

- B0 → B2: +7.26pp, 전체 실험 중 최대. 14개 클래스 전체 상승. Bicycle +39.83, TrafficLight +10.2, Motorcycle +9.4, Sky와 Road +1.5pp 이하.
- 384×512 → 480×640: val +0.91pp, test +2.23pp. Bicycle +6.77, Pole +3.43, Person +2.02, TrafficLight +1.32, Bus −3.33.
- 입력 축소율: 384×512는 원본의 0.64배, 480×640은 0.8배
- 공통 조건: 3:4 비율, 32의 배수, Colab T4, batch 8
- Bus 변화: 용량 확대 +7.65, IR 추가 −2.60, 해상도 상승 −3.33. [design_notes.md](design_notes.md#3-bus-cross-cycle-pattern-cycles-3-and-4)
- Files: [03_capacity](03_capacity/README.md)

### 3.4 Oversampling — 사이클 1-B와 5

- B0, #2 → #3: Bicycle IoU 0 → 43.40, 전체 65.48 → 67.97. Bicycle 포함 val 2장.
- B2 480×640, #9 → #11: val 73.65 → 74.46, +0.81pp. test 66.13 → 65.64, −0.49pp.
- val 상승 분해: Bicycle +1.14pp, 나머지 13개 클래스 합계 −0.33pp
- Bicycle 변화: +15.98pp. 하락 클래스 8개. TrafficSign −2.10, Motorcycle −1.69, Truck −1.38 등.
- Bicycle 제외 13개 클래스 mIoU: 75.73 → 75.37, −0.36pp. test와 같은 방향.
- val−test 격차: 7.52 → 8.82pp
- Files: [01_baseline](01_baseline/README.md), [05_oversampling_final](05_oversampling_final/README.md) ([classiou_sum_vs_oversample.csv](05_oversampling_final/classiou_sum_vs_oversample.csv))

### 3.5 실패 사례와 TTA — 최종 모델

- 최저 픽셀 정확도 val 4장: 00108, 00107, 00136, 00394. 384×512 모델과 동일.
- 주요 오류: Sidewalk ↔ Road, Vegetation → Road 등 노면 재질 혼동
- 좌우 반전 TTA: val 73.65 → 73.63. 제출 미사용.
- Figures and tables: [06_final_model](06_final_model/README.md)

## 4. 리더보드 제출

| Submission ID | Model | Val mIoU | Test mIoU | Pixel Acc |
|---|---|---:|---:|---:|
| 1061666 | #9 B2 Sum 480×640, 최종 | 73.65 | 66.13 | 93.38 |
| 1060429 | #8 B2 Sum 384×512 | 72.74 | 63.90 | |
| 1058499 | #5 B0 Adaptive | 65.35 | 55.84 | 91.95 |
| 1062714 | #11 B2 Sum 480×640 + Oversampling | 74.46 | 65.64 | |

- 제출 전 점검: #5, #8, #9, #11 모두 `check_submission.py` 통과
- 점검 항목: PNG 280장, 원본 크기, 클래스 0–13, 누락 0
- 모델 선택 기준: val mIoU만 사용, test 점수 기반 조정 없음
- 제출 선택 근거: [설계 문서](../docs/experiment_design.md#리더보드-제출)

## 5. 재현 방법

```bash
pip install -r requirements.txt
python tools/prepare_fmb.py --raw data/FMB_raw --out data/FMB_prepared

# 최종 모델
python train.py --data_root data/FMB_prepared --fusion sum --backbone nvidia/mit-b2 \
  --height 480 --width 640 --epochs 30 --seed 0 --out runs/fmb_b2_sum_480
python predict.py --data_root data/FMB_prepared --ckpt runs/fmb_b2_sum_480/best.pth \
  --split test --out submission_b2_sum_480
python check_submission.py --zip submission_b2_sum_480.zip --data_root data/FMB_prepared
```

`check_submission.py`는 교수가 공유 폴더(26-2 project)에 올린 점검 스크립트다.

| Experiment | CLI option |
|---|---|
| RGB 단독 / Early | `--fusion rgb` / `--fusion early` |
| Concat / Adaptive / Concat_res | `--fusion concat` / `adaptive` / `concat_res` ([`fusion_seg/fusion.py`](../fusion_seg/fusion.py)) |
| Oversampling | `python train_oversample.py ... --target_fraction 0.10` |
| MiT-B2 | `--backbone nvidia/mit-b2` |
| 480×640 | `--height 480 --width 640` |

- 각 run의 설정과 epoch별 로그: `runs/<이름>/args.json`, `runs/<이름>/log.jsonl`. `best.pth` / `last.pth`는 용량 때문에 팀 Google Drive(`ICV_FMB/runs/`)에 보관하며 요청 시 제출한다.
- 로그에서 표와 그림을 다시 만들려면 `python tools/analyze_runs.py` ([results/README.md](README.md#재생성)).
- 실험 노트북: [`notebooks/trackA_FMB_results.ipynb`](../notebooks/trackA_FMB_results.ipynb) (B0 융합 비교는 [`trackA_FMB_fusion.ipynb`](../notebooks/trackA_FMB_fusion.ipynb), B0 RGB/Sum/Oversampling은 [`trackA_FMB_baseline.ipynb`](../notebooks/trackA_FMB_baseline.ipynb)).
- 데이터: 저자 Google Drive 다운로드 쿼터 초과로, 원본 `train.zip` / `test.zip`을 팀 Drive에 두고 Colab에서 압축 해제 후 `prepare_fmb.py`로 준비했다. 분할(train 1,000 / val 220 / test 280)은 원본 스크립트 결과와 파일 목록이 동일함을 확인했다. test 라벨은 읽지 않았다.
- 환경: Google Colab, Tesla T4 1장, `transformers==4.46.3` (`requirements.txt`).

## 6. 규칙 준수

- 학습·튜닝·모델 선택은 `index_train.json` / `index_val.json`만 사용, test는 최종 예측에만 사용.
- 사전학습은 ImageNet MiT(`nvidia/mit-b0`, `nvidia/mit-b2`)만 사용, 외부 데이터 없음.
- 최종 모델: 두 인코더 모두 MiT-B2, 총 51.55M (약 52M 이하).
- 리더보드는 팀 계정 하나(`reduce_l`)로만 제출했다.
- 측정 한계(단일 seed, IR 기여의 파라미터 교란, 밝기 그룹의 의미 등)는 [설계 문서](../docs/experiment_design.md#측정-한계)에 모았다.

## 7. 폴더 구성

| Path | Description |
|---|---|
| [`../docs/experiment_design.md`](../docs/experiment_design.md) | 실험 순서, 선택 이유, 비교 기준, 결과 해석 |
| [`../fusion_seg/fusion.py`](../fusion_seg/fusion.py) | 융합 모듈: `sum`, `concat`, `adaptive`, `concat_res` (각 클래스에 출처 주석) |
| `../fusion_seg/model.py`, `data.py`, `metrics.py` | 제공 코드 |
| `../train.py`, `../predict.py` | 제공 코드 |
| [`../train_oversample.py`](../train_oversample.py) | Bicycle(12) oversampling 학습 (`WeightedRandomSampler`) |
| [`../tools/analyze_runs.py`](../tools/analyze_runs.py) | 로그에서 표·그림 재생성 |
| `../runs/` | 실험별 `args.json`, `log.jsonl` |
| [`design_notes.md`](design_notes.md) | 학습 로그 기반 검증 분석 |
| `00_overview` ~ `06_final_model` | 사이클별 결과 ([폴더 지도](README.md)) |
| `../notebooks/` | 실행 노트북 |

## 8. 출처

- FMB: J. Liu et al., "Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation," ICCV 2023. https://github.com/JinyuanLiu-CV/SegMiF
- SegFormer / MiT: E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021. Hugging Face `nvidia/mit-b0`, `nvidia/mit-b2`.
- Adaptive 융합(채널 게이트): J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," CVPR 2018; ICV 07 강의 SE 블록.
- Concat / Concat_res: ICV 07 강의 (concat + 1×1 conv). 잔차 0 초기화: P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017.
- 제공 코드(`vip_trackA_FMB`)와 ICV 06·07 강의 자료.
