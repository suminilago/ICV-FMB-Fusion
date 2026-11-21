# 트랙 A: FMB RGB–열화상(IR) 융합 세그멘테이션

컴퓨터비전 기말 프로젝트 트랙 A (FMB, RGB + IR, 14 클래스). 리더보드 팀 계정: `reduce_l`.

**최종 모델**: SegFormer, RGB·IR 인코더 모두 MiT-B2, Stage별 Sum 융합, 입력 480×640
→ **val mIoU 73.65 / test mIoU 66.13** (Codabench 제출 1061666), params 51.55M

## 1. 한눈에 보기

| 단계 | 질문 (가설) | 비교 | val mIoU | 결론 |
|---|---|---|---|---|
| 1 | IR이 도움이 되는가? | RGB 단독 → RGB+IR Sum (B0) | 61.53 → 65.48 | **+3.95**, IR 효과 확인 |
| 2 | 학습형 융합이 Sum보다 나은가? | Concat / Adaptive (B0) | 62.47 / 65.35 | Sum보다 확실히 나은 구조 없음 |
| 3 | Concat이 낮은 이유는 초기화인가? | Concat → Concat_res (0 초기화) | 62.47 → 64.95 | **+2.48**, 사전학습 특징 보존이 핵심 |
| 4 | 합치는 위치가 중요한가? | Early(입력 융합) vs Sum(단계별) | 62.19 vs 65.48 | IR 전용 인코더가 필요 |
| 5 | 희귀 클래스(Bicycle)를 살릴 수 있나? | Sum → Sum + Oversampling | 65.48 → 67.97 | Bicycle IoU 0 → 43.4 (val 2장 한계) |
| 6 | 병목이 모델 용량인가? | B0 → B2 (Sum, 384×512) | 65.48 → 72.74 | **+7.26**, 가장 큰 개선 |
| 7 | 작은 물체가 축소로 사라지나? | 384×512 → 480×640 (B2) | 72.74 → 73.65 | +0.91 (test **+2.23**), **최종 모델** |
| 8 | 강한 백본에서도 IR이 필요한가? | B2 480 RGB 단독 → Sum | 70.99 → 73.65 | **+2.66**, 어두울수록 효과 ↑ |
| 9 | 강한 백본에서도 oversampling이 도움이 되나? | B2 480 Sum → Sum + Oversampling | 73.65 → 74.46 | Bicycle +15.98, 나머지 13개 평균 기여 −0.33, test **−0.49** → 최종 미채택 |

모든 실험은 seed 0, batch 8, AdamW lr 6e-5(디코더 10배), weight decay 0.01, warm-up 2 epoch + poly, 30 epoch, 같은 증강으로 고정하고 **한 번에 한 요소만** 바꿨다. 모델 선택은 val mIoU로만 했다.

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
- 학습 곡선: [`results/analysis/learning_curves.png`](results/analysis/learning_curves.png), 요약: [`results/analysis/run_summary.csv`](results/analysis/run_summary.csv)

## 3. 주요 분석

### 3.1 IR 기여 (최종 설정: B2, 480×640)

밝기 그룹은 val 220장을 RGB 평균 밝기로 3등분했다.

| 밝기 그룹 (0–255) | 이미지 | RGB 단독 | RGB + IR | IR 기여 |
|---|---:|---:|---:|---:|
| 어두움 (39–98) | 73 | 61.91 | 66.55 | **+4.64** |
| 중간 (98–149) | 73 | 73.06 | 75.86 | +2.80 |
| 밝음 (149–162) | 74 | 73.94 | 74.83 | +0.89 |

- 어두울수록 IR 기여가 커진다. 파라미터 증가 효과라면 밝기와 무관하게 비슷해야 하므로, 조명이 약할 때 IR이 정보를 보완한다는 가설과 일치한다.
- 클래스별로는 열을 내는 물체에서 크다: Person **+18.79** (어두운 장면 42.32 → 65.82, +23.50), Bicycle +7.81, Motorcycle +3.04, Pole +2.66, Car +2.27.
- 무늬로 구분하는 TrafficSign(−0.75)과 Bus(−2.60)는 소폭 하락했다.
- 그림: [`ir_gain_class_brightness.png`](results/analysis/ir_gain_class_brightness.png), [`dark_rgb_vs_sum.png`](results/analysis/dark_rgb_vs_sum.png)

### 3.2 융합 위치·구조와 초기화 (B0)

- Early는 RGB 단독 대비 +0.66에 그쳤다. ImageNet RGB로 사전학습된 인코더 하나로는 IR 고유의 특징을 뽑지 못한다.
- Concat(랜덤 1×1 conv)은 Sum보다 −3.01 낮았다. 융합 출력이 다음 RGB Stage의 입력이 되므로, 랜덤 초기화가 학습 초반 사전학습 특징을 깨뜨린다고 보았다.
- 구조는 그대로 두고 처음에 `x + a`가 되도록 0 초기화한 Concat_res는 +2.48 회복했다. Adaptive(게이트 0 초기화)도 Sum과 비슷했다(−0.13).
- 단일 seed이므로 약 0.5pp 이내 차이(Sum vs Adaptive, Concat_res)는 순위를 단정하지 않았다.

### 3.3 백본과 해상도

- B0 → B2: 14개 클래스 모두 상승. Bicycle 0 → 39.83(oversampling 없이), TrafficLight +10.2, Motorcycle +9.4, 큰 클래스(Sky, Road)는 +1.5 이하. B0의 한계는 데이터 부족뿐 아니라 모델 용량이었다.
- 384×512 → 480×640: 작은 물체가 주로 올랐다 (Bicycle +6.77, Pole +3.43, Person +2.02, TrafficLight +1.32). Bus는 −3.33.
- 입력 384×512는 원본 600×800의 0.64배(베이스라인 기본값, Colab T4 비용 고려), 480×640은 0.8배다. 둘 다 3:4 비율과 32의 배수를 유지한다. 모든 실험이 batch 8로 메모리 문제 없이 돌았다.

### 3.4 실패 사례 (최종 모델)

픽셀 정확도가 가장 낮은 val 4장은 384×512 모델과 같았고, 오류는 대부분 노면 재질 혼동이었다.

| 이미지 | 장면 | 픽셀 정확도 | 주된 오류 (정답 → 예측) | 오류 중 비율 |
|---|---|---:|---|---:|
| 00108 | 비에 젖은 광장, 주간 | 58.9% | Sidewalk → Road | 93.0% |
| 00107 | 비에 젖은 광장, 주간 | 59.1% | Sidewalk → Road | 94.7% |
| 00136 | 야간, 가로등 | 67.9% | Road → Sidewalk | 72.6% |
| 00394 | 야간, 전조등 눈부심 | 73.6% | Vegetation → Road | 78.9% |

젖은 노면 반사와 야간 조명으로 RGB가 흔들리고, IR에서는 온도가 비슷한 보도·차도·잔디가 하나의 면으로 보인다. 센서 융합과 모델 확장으로도 풀리지 않은 오류다.
그림: [`final_failure_cases_b2_480.png`](results/analysis/final_failure_cases_b2_480.png), [`final_confusion_matrix_b2_480.png`](results/analysis/final_confusion_matrix_b2_480.png)

### 3.5 TTA

최종 모델에 좌우 반전 TTA를 val로 비교했다: 73.65 → 73.63. 효과가 없어 제출에 쓰지 않았다.

### 3.6 B2에서의 Oversampling (#9 → #11)

B0에서 Bicycle을 살린 oversampling(클래스 12 포함 이미지 비율 10%)을 최종 설정(B2, 480×640)에 같은 조건으로 적용했다.

| 클래스 | Sum (#9) | Sum + OS (#11) | 차이 |
|---|---:|---:|---:|
| Bicycle | 46.60 | 62.59 | **+15.98** |
| Sidewalk / Bus / Road | 57.04 / 73.72 / 90.29 | 57.61 / 74.25 / 90.70 | +0.57 / +0.53 / +0.40 |
| Person | 67.16 | 66.58 | −0.58 |
| Truck | 65.58 | 64.20 | −1.38 |
| Motorcycle | 64.15 | 62.46 | −1.69 |
| TrafficSign | 76.24 | 74.15 | −2.10 |
| **mIoU (val)** | 73.65 | 74.46 | +0.81 |
| **mIoU (test)** | 66.13 | 65.64 | −0.49 |

- val 상승 +0.81 중 Bicycle 기여가 +1.14, 나머지 13개 클래스 기여는 −0.33이다. B0(#2 → #3)와 같은 패턴으로, 희귀 클래스를 자주 보여주면 그 클래스는 오르지만 다른 클래스는 조금씩 내려간다.
- val에서 Bicycle이 나오는 이미지가 매우 적어 Bicycle IoU의 변동이 크고, test에서는 오히려 −0.49로 상승이 이어지지 않았다. 단일 seed에서 0.5pp 안팎의 차이라 노이즈 범위로 본다.
- 결론: oversampling은 **희귀 클래스 보완 방법**으로는 확인됐지만, 전체 성능 개선은 확인되지 않았다(가설 부분 확인). 최종 모델은 #9로 유지했다.
- 클래스별 결과: [`class_iou_b2_480_os_vs_sum.csv`](class_iou_b2_480_os_vs_sum.csv)

## 4. 리더보드 제출

| 제출 ID | 모델 | Val mIoU | Test mIoU | Pixel Acc |
|---|---|---:|---:|---:|
| 1062714 | B2 Sum 480×640 + Oversampling (#11) | 74.46 | 65.64 | |
| 1061666 | B2 Sum 480×640 (#9, 최종) | 73.65 | **66.13** | 93.38 |
| 1060429 | B2 Sum 384×512 (#8) | 72.74 | 63.90 | |
| 1058499 | B0 Adaptive (#5) | 65.35 | 55.84 | 91.95 |

모든 제출은 `check_submission.py`로 점검했다(280장, 원본 크기, 클래스 0–13, 누락 0). 모델과 설정은 val mIoU로만 골랐고 test 점수에 맞춘 조정은 없다. #11은 최종 모델(#9) 제출 후 진행한 추가 실험이다. val은 #9보다 높지만 상승분이 val 이미지가 적은 Bicycle 한 클래스에서 나와(3.6절) 최종 모델을 바꾸지 않았고, test 결과(65.64)도 함께 기록했다.

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

| 실험 | 실행 옵션 |
|---|---|
| RGB 단독 / Early | `--fusion rgb` / `--fusion early` |
| Concat / Adaptive / Concat_res | `--fusion concat` / `adaptive` / `concat_res` ([`fusion_seg/fusion.py`](fusion_seg/fusion.py)) |
| Oversampling | `python train_oversample.py ... --target_fraction 0.10` |
| MiT-B2 | `--backbone nvidia/mit-b2` |
| 480×640 | `--height 480 --width 640` |

- 각 run의 설정과 epoch별 로그: `runs/<이름>/args.json`, `runs/<이름>/log.jsonl`. `best.pth` / `last.pth`는 용량 때문에 팀 Google Drive(`ICV_FMB/runs/`)에 보관하며 요청 시 제출한다.
- 실험 노트북: [`notebooks/trackA_FMB_test.ipynb`](notebooks/trackA_FMB_test.ipynb) (B0 RGB/Sum/Oversampling은 [`trackA_FMB_baseline.ipynb`](notebooks/trackA_FMB_baseline.ipynb)).
- 데이터: 저자 Google Drive 다운로드 쿼터 초과로, 원본 `train.zip` / `test.zip`을 팀 Drive에 두고 Colab에서 압축 해제 후 `prepare_fmb.py`로 준비했다. 분할(train 1,000 / val 220 / test 280)은 원본 스크립트 결과와 파일 목록이 동일함을 확인했다. test 라벨은 읽지 않았다.
- 환경: Google Colab, Tesla T4 1장, `transformers==4.46.3` (`requirements.txt`).

## 6. 규칙 준수

- 학습·튜닝·모델 선택은 `index_train.json` / `index_val.json`만 사용, test는 최종 예측에만 사용.
- 사전학습은 ImageNet MiT(`nvidia/mit-b0`, `nvidia/mit-b2`)만 사용, 외부 데이터 없음.
- 최종 모델: 두 인코더 모두 MiT-B2, 총 51.55M (약 52M 이하).
- 한계: 모든 결과가 단일 seed. Sum의 IR 기여에는 인코더 2개에 따른 파라미터 증가가 섞여 있을 수 있다(밝기·클래스별 분석이 IR 정보 효과를 뒷받침). 밝기 그룹은 날씨 라벨이 아니다.

## 7. 폴더 구성

| 경로 | 내용 |
|---|---|
| `fusion_seg/fusion.py` | 융합 모듈: `sum`, `concat`, `adaptive`, `concat_res` (각 클래스에 출처 주석) |
| `fusion_seg/model.py`, `data.py`, `metrics.py` | 제공 코드 (수정 없음) |
| `train.py`, `predict.py` | 제공 코드 (수정 없음) |
| `train_oversample.py` | 클래스 12 oversampling 학습 (`WeightedRandomSampler`) |
| `runs/` | 실험별 `args.json`, `log.jsonl` |
| `results/` | 단계별 결과 문서(`baseline_results.md`, `fusion_results.md`, `b2_results.md`)와 B0 분석 |
| `results/analysis/` | 최종 분석: 학습 곡선, IR 기여(클래스·밝기), 최종 모델 혼동·실패 사례, TTA |
| `notebooks/` | 실행 노트북 |

## 8. 출처

- FMB: J. Liu et al., "Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation," ICCV 2023. https://github.com/JinyuanLiu-CV/SegMiF
- SegFormer / MiT: E. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021. Hugging Face `nvidia/mit-b0`, `nvidia/mit-b2`.
- Adaptive 융합(채널 게이트): J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," CVPR 2018; ICV 07 강의 SE 블록.
- Concat / Concat_res: ICV 07 강의 (concat + 1×1 conv). 잔차 0 초기화: P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," 2017.
- 제공 코드(`vip_trackA_FMB`)와 ICV 06·07 강의 자료.
