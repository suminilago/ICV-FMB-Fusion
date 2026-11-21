# 트랙 A: FMB RGB–열화상(IR) 융합 세그멘테이션

컴퓨터비전 기말 프로젝트 트랙 A (FMB, RGB + IR, 14 클래스). 리더보드 팀 계정: `reduce_l`.

**최종 모델**: SegFormer, RGB·IR 인코더 모두 MiT-B2, Stage별 Sum 융합, 입력 480×640
→ **val mIoU 73.65 / test mIoU 66.13** (Codabench 제출 1061666), params 51.55M

가설을 세운 근거와 판정은 [실험 설계 문서](../docs/experiment_design.md)에 있고, 이 보고서는 수치와 결과를 다룬다.
각 절의 제목에 어느 사이클을 검증하는지 적었다.

## 1. 실험 조건

모든 실험은 seed 0, batch 8, AdamW lr 6e-5(디코더 10배), weight decay 0.01, warm-up 2 epoch + poly, 30 epoch, 같은 증강으로 고정하고 **한 번에 한 요소만** 바꿨다.
모델 선택은 val mIoU로만 했다. 평가는 14개 클래스 평균 mIoU다.

## 2. 전체 실험 결과

| # | 실험 | 백본 | 입력 | 융합 | Params | Best epoch | Val mIoU | Test mIoU | 로그 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | RGB 단독 | B0 | 384×512 | — | 3.7M | 29 | 61.53 | | `runs/fmb_rgb` |
| 2 | Sum | B0 | 384×512 | 단계별 합 | 7.04M | 30 | 65.48 | | `runs/fmb_sum` |
| 3 | Sum + Oversampling | B0 | 384×512 | 단계별 합 | 7.04M | 29 | 67.97 | | `runs/fmb_sum_oversample10` |
| 4 | Concat | B0 | 384×512 | concat → 1×1 conv | 7.23M | 28 | 62.47 | | `runs/fmb_concat` |
| 5 | Adaptive | B0 | 384×512 | 채널별 게이트 | 7.23M | 29 | 65.35 | 55.84 | `runs/fmb_adaptive` |
| 6 | Concat_res | B0 | 384×512 | sum + concat 잔차(0 초기화) | 7.33M | 27 | 64.95 | | `runs/fmb_concat_res` |
| 7 | Early | B0 | 384×512 | 입력 단계 합(인코더 1개) | 3.72M | 30 | 62.19 | | `runs/fmb_early` |
| 8 | Sum | B2 | 384×512 | 단계별 합 | 51.55M | 29 | 72.74 | 63.90 | `runs/fmb_b2_sum` |
| **9** | **Sum (최종)** | **B2** | **480×640** | 단계별 합 | **51.55M** | 30 | **73.65** | **66.13** | `runs/fmb_b2_sum_480` |
| 10 | RGB 단독 | B2 | 480×640 | — | 27.36M | 30 | 70.99 | | `runs/fmb_b2_rgb_480` |
| 11 | Sum + Oversampling | B2 | 480×640 | 단계별 합 | 51.55M | 30 | 74.46 | 65.64 | `runs/fmb_b2_sum_480_oversample10` |
| 참고 | 교수 베이스라인 RGB / Sum | B0 | 384×512 | | | | | 52.77 / 56.30 | |

- Best epoch는 1부터 센 값이다(`log.jsonl`의 epoch + 1). 11개 run 모두 30 epoch가 빠짐없이 기록되어 있다.
- #6, #9는 Colab 연결 끊김으로 같은 명령을 다시 실행해 `last.pth`에서 재개했다.
- 학습 곡선과 run 요약: [00_overview](00_overview/README.md)

## 3. 주요 분석

### 3.1 IR 기여 (사이클 4: 최종 설정 B2, 480×640, #10 vs #9)

밝기 그룹은 val 220장을 RGB 평균 밝기로 3등분했다.

| 밝기 그룹 (0–255) | 이미지 | RGB 단독 | RGB + IR | IR 기여 |
|---|---:|---:|---:|---:|
| 어두움 (39–98) | 73 | 61.91 | 66.55 | **+4.64** |
| 중간 (98–149) | 73 | 73.06 | 75.86 | +2.80 |
| 밝음 (149–162) | 74 | 73.94 | 74.83 | +0.89 |

- 전체 IR 기여는 +2.66 (70.99 → 73.65)이고, 어두울수록 커진다.
- 클래스별로 Person **+18.79** (어두운 장면 42.32 → 65.82, +23.50), Bicycle +7.81, Motorcycle +3.04, Pole +2.66, Car +2.27이다. TrafficSign(−0.75)과 Bus(−2.60)는 내렸다.
- epoch별 IR 기여는 ep1 −3.63, ep5 +2.07, ep10 +2.41, ep20 +3.36, ep30 +2.66이다.
- B0에서도 RGB 단독 61.53 → Sum 65.48 (+3.95)로 같은 방향이었다 (사이클 1).
- 파일: [04_ir_contribution](04_ir_contribution/README.md)

### 3.2 융합 위치·구조와 초기화 (사이클 2, B0)

- Sum 대비: Adaptive −0.13, Concat_res −0.53, Concat −3.01, Early −3.29. 학습형 융합 중 Sum을 넘은 것은 없다.
- Early는 RGB 단독 대비 +0.66이다.
- (Concat_res − Concat)의 epoch별 격차는 ep1 +1.46, ep4 **+9.74**(최대), ep10 +3.14, ep20 +3.41, ep30 +2.56이다. (Adaptive − Concat)도 ep4에서 +9.30으로 최대다.
- 같은 val에서 Adaptive는 Bicycle을 뺀 13개 클래스 모두 Concat보다 높다.
- 0.5pp 이내 차이(Sum, Adaptive, Concat_res)는 순위를 정하지 않았다. 융합 비교는 B0에서만 했다.
- 파일: [02_fusion](02_fusion/README.md)

### 3.3 백본과 해상도 (사이클 3, #2 → #8 → #9)

- B0 → B2 (+7.26, 전체 실험 중 최대): 14개 클래스가 모두 올랐다. Bicycle 0 → 39.83(oversampling 없이), TrafficLight +10.2, Motorcycle +9.4, 큰 클래스(Sky, Road)는 +1.5 이하다.
- 384×512 → 480×640 (val +0.91, test +2.23): 작은 물체가 주로 올랐다. Bicycle +6.77, Pole +3.43, Person +2.02, TrafficLight +1.32. Bus는 −3.33.
- 입력 384×512는 원본 600×800의 0.64배(베이스라인 기본값), 480×640은 0.8배다. 둘 다 3:4 비율과 32의 배수를 유지한다. 모든 실험이 Colab T4에서 batch 8로 돌았다.
- Bus는 용량 확대에서 +7.65, IR 추가에서 −2.60, 해상도 상승에서 −3.33으로 세 사이클에서 방향이 갈렸다 ([design_notes.md](design_notes.md#3-bus-교차-패턴-사이클-3-4)).
- 파일: [03_capacity](03_capacity/README.md)

### 3.4 Oversampling (사이클 1-B와 5)

- B0 (#2 → #3): Bicycle IoU 0 → 43.40, 전체 65.48 → 67.97. val에서 Bicycle이 나오는 이미지는 2장뿐이다.
- B2, 480×640 (#9 → #11): val +0.81 (73.65 → 74.46), test **−0.49** (66.13 → 65.64).
  val 상승 +0.81 중 Bicycle 기여가 +1.14, 나머지 13개 클래스 합계가 −0.33이다. Bicycle은 +15.98이고 8개 클래스가 내렸다(TrafficSign −2.10, Motorcycle −1.69, Truck −1.38 등).
  Bicycle을 뺀 val 13개 클래스 mIoU는 75.73 → 75.37 (−0.36)로 test와 방향이 같다. val−test 격차는 7.52에서 8.82로 벌어졌다.
- 파일: [01_baseline](01_baseline/README.md), [05_oversampling_final](05_oversampling_final/README.md) ([classiou_sum_vs_oversample.csv](05_oversampling_final/classiou_sum_vs_oversample.csv))

### 3.5 실패 사례와 TTA (최종 모델)

- 픽셀 정확도가 가장 낮은 val 4장(00108, 00107, 00136, 00394)은 384×512 모델과 같았고, 오류는 대부분 노면 재질 혼동(Sidewalk ↔ Road, Vegetation → Road)이다.
- 좌우 반전 TTA는 val 73.65 → 73.63으로 효과가 없어 제출에 쓰지 않았다.
- 그림과 표: [06_final_model](06_final_model/README.md)

## 4. 리더보드 제출

| 제출 ID | 모델 | Val mIoU | Test mIoU | Pixel Acc |
|---|---|---:|---:|---:|
| 1061666 | B2 Sum 480×640 (#9, 최종) | 73.65 | **66.13** | 93.38 |
| 1060429 | B2 Sum 384×512 (#8) | 72.74 | 63.90 | |
| 1058499 | B0 Adaptive (#5) | 65.35 | 55.84 | 91.95 |
| 1062714 | B2 Sum 480×640 + Oversampling (#11) | 74.46 | 65.64 | |

#5, #8, #9는 `check_submission.py`로 점검했다(280장, 원본 크기, 클래스 0–13, 누락 0).
모델과 설정은 val mIoU로만 골랐고 test 점수에 맞춘 조정은 없다. 제출 선택의 기준은 [설계 문서](../docs/experiment_design.md#리더보드-제출)에 있다.

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

| 실험 | 실행 옵션 |
|---|---|
| RGB 단독 / Early | `--fusion rgb` / `--fusion early` |
| Concat / Adaptive / Concat_res | `--fusion concat` / `adaptive` / `concat_res` ([`fusion_seg/fusion.py`](../fusion_seg/fusion.py)) |
| Oversampling | `python train_oversample.py ... --target_fraction 0.10` |
| MiT-B2 | `--backbone nvidia/mit-b2` |
| 480×640 | `--height 480 --width 640` |

- 각 run의 설정과 epoch별 로그: `runs/<이름>/args.json`, `runs/<이름>/log.jsonl`. `best.pth` / `last.pth`는 용량 때문에 팀 Google Drive(`ICV_FMB/runs/`)에 보관하며 요청 시 제출한다.
- 로그에서 표와 그림을 다시 만들려면 `python tools/analyze_runs.py` ([results/README.md](README.md#재생성)).
- 실험 노트북: [`notebooks/trackA_FMB_test.ipynb`](../notebooks/trackA_FMB_test.ipynb) (B0 융합 비교는 [`trackA_FMB_fusion.ipynb`](../notebooks/trackA_FMB_fusion.ipynb), B0 RGB/Sum/Oversampling은 [`trackA_FMB_baseline.ipynb`](../notebooks/trackA_FMB_baseline.ipynb)).
- 데이터: 저자 Google Drive 다운로드 쿼터 초과로, 원본 `train.zip` / `test.zip`을 팀 Drive에 두고 Colab에서 압축 해제 후 `prepare_fmb.py`로 준비했다. 분할(train 1,000 / val 220 / test 280)은 원본 스크립트 결과와 파일 목록이 동일함을 확인했다. test 라벨은 읽지 않았다.
- 환경: Google Colab, Tesla T4 1장, `transformers==4.46.3` (`requirements.txt`).

## 6. 규칙 준수

- 학습·튜닝·모델 선택은 `index_train.json` / `index_val.json`만 사용, test는 최종 예측에만 사용.
- 사전학습은 ImageNet MiT(`nvidia/mit-b0`, `nvidia/mit-b2`)만 사용, 외부 데이터 없음.
- 최종 모델: 두 인코더 모두 MiT-B2, 총 51.55M (약 52M 이하).
- 리더보드는 팀 계정 하나(`reduce_l`)로만 제출했다.
- 측정 한계(단일 seed, IR 기여의 파라미터 교란, 밝기 그룹의 의미 등)는 [설계 문서](../docs/experiment_design.md#측정-한계)에 모았다.

## 7. 폴더 구성

| 경로 | 내용 |
|---|---|
| [`../docs/experiment_design.md`](../docs/experiment_design.md) | 설계와 가설 검증 (근거, 검증 지표, 판정) |
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
