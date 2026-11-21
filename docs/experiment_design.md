# 실험 설계와 가설 검증

트랙 A(FMB, RGB + 열화상, 14 클래스)에서 11개 실험을 어떤 순서로, 무엇을 근거로 돌렸는지 적은 문서다.
수치 전체는 [results/report.md](../results/report.md)에 있고, 여기서는 각 가설의 **근거**와 **판정**을 다룬다.
근거로 쓴 수치는 모두 출처 파일에 링크했다.

베이스라인과 데이터를 먼저 관찰하고, 그 관찰을 근거로 가설을 세우고, 한 번에 한 요소만 바꿔 검증했다.
검증 결과가 다음 사이클의 근거가 되어 이 과정을 다섯 번 돌았다. 사이클마다 아래 순서로 쓴다.

> 근거 → 가설 → 근거에서 가설이 나온 이유 → 설계 → 검증 지표 → 결과 → 판정 → 다음으로

epoch는 모두 1부터 센 값이다. 클래스 번호는 FMB 순서를 따른다:
0 Road, 1 Sidewalk, 2 Building, 3 TrafficLight, 4 TrafficSign, 5 Vegetation, 6 Sky, 7 Person, 8 Car,
9 Truck, 10 Bus, 11 Motorcycle, 12 Bicycle, 13 Pole.

## 출발점

### 과제가 고정한 조건

| 항목 | 값 | 설계에 준 제약 |
|---|---|---|
| 분할 | 공식 train 1,220 → train 1,000 / val 220 (고정 시드), 공식 test 280 | 모델 선택은 val로만 한다. test는 리더보드 점수로만 본다 |
| 평가 | mIoU = 클래스 평균 | 한 클래스가 빈도와 무관하게 1/14 가중치를 가진다. 희귀 클래스가 점수에 크게 작용한다 |
| 모델 | 두 인코더 모두 MiT-B2 이하, 융합 모듈 포함 총 52M 이하 | 쓸 수 있는 최대 구성은 B2 + B2 (51.55M) |
| 환경 | Colab T4 1장 | 입력 해상도와 학습 길이의 상한 |
| 필수 | RGB 단독 대비 IR 기여 ablation 표 | 사이클 1과 4 |
| 재현 | `runs/<이름>/args.json`, `log.jsonl` 제출, 교수가 재실행 | run별 설정과 로그를 레포에서 추적한다 |

과제 안내문 원문은 [루트 README](../README.md) 아래쪽에 그대로 있다.

### 우리가 고정한 조건

모든 run이 같다: seed 0, batch 8, AdamW lr 6e-5(디코더 10배), weight decay 0.01,
warm-up 2 epoch + poly(0.9), 30 epoch, 증강 동일(scale 0.75~1.5, 랜덤 크롭, 좌우 반전 0.5).
한 번에 한 요소만 바꿨고, run별 설정은 `runs/<이름>/args.json`에 있다.
Colab 연결이 끊긴 #6, #9는 같은 명령을 다시 실행해 `last.pth`에서 재개했다.

### 실험 전에 확인한 데이터 성질

가설을 세우기 전에 이미 알 수 있던 것들이다.

| 항목 | 값 | 확인한 곳 |
|---|---|---|
| 원본 해상도 | 600×800 | val 220장의 라벨 픽셀 합 103,781,695 ÷ 220 = 471,735, 즉 480,000의 98.3% ([classiou_b0_vs_b2.csv](../results/03_capacity/classiou_b0_vs_b2.csv)의 `gt_pixels` 합). 라벨 크기도 600×800 ([dist_class12_val_images.csv](../results/01_baseline/dist_class12_val_images.csv)) |
| Bicycle(12) train 노출 | 10장, 전체 픽셀의 0.0017%. 14개 클래스 중 최소 | [dist_train_class.csv](../results/01_baseline/dist_train_class.csv) |
| Bicycle val 노출 | 220장 중 2장, 합계 994픽셀 (183, 811) | [dist_class12_val_images.csv](../results/01_baseline/dist_class12_val_images.csv) |
| 입력 축소율 | 384×512는 원본의 0.64배, 480×640은 0.8배 | 600×800 기준 계산 |

### 제공 베이스라인

교수 제공 베이스라인(MiT-B0, 384×512)의 test mIoU는 RGB 단독 52.77, Sum 56.30이다
([report.md 실험 표의 참고 행](../results/report.md#2-전체-실험-결과)).
같은 설정으로 재현한 val mIoU는 RGB 단독 61.53, Sum 65.48이고, 교수 제공 기준값(61.50, 65.40)과 거의 같다
([01_baseline](../results/01_baseline/README.md)).

## 사이클 1. 기준선과 희귀 클래스

### 가설 1-A. IR은 RGB를 보완한다

**근거**
1. 제공 베이스라인에서 IR 융합의 이득이 test +3.53이다 (52.77 → 56.30). [report.md](../results/report.md#2-전체-실험-결과)
2. FMB에는 안개, 폭우, 저조도 장면이 들어 있다. [루트 README](../README.md)의 "데이터" 항목
3. 과제가 RGB 단독 대비 IR 기여를 ablation 표로 요구한다. [루트 README](../README.md)의 "규칙"

**가설** 단계별 Sum 융합이 RGB 단독보다 mIoU가 높다.

**근거 → 가설** 제공 베이스라인이 이미 같은 방향의 이득을 보였고, 데이터에 RGB가 약해지는 조건이 있으므로 조명과 무관하게 열을 재는 IR이 정보를 보탤 것이라 보았다.
먼저 우리 환경에서 같은 결과가 재현되는지를 본다.

**설계** B0, 384×512에서 `--fusion rgb`와 `--fusion sum`만 다르게 학습한다. (#1, #2)

**검증 지표** 사전에 정한 세부 지표는 없고, Sum이 RGB 단독보다 높은지로 판정한다.

**결과** RGB 단독 61.53 → Sum 65.48 (+3.95). 클래스별로는 12개가 오르고 Person이 +19.26으로 가장 크다.
TrafficSign은 −0.39, Bicycle은 두 모델 모두 0이다. [classiou_rgb_vs_sum.csv](../results/01_baseline/classiou_rgb_vs_sum.csv)

**판정** 지지

**다음으로** IR 기여가 클래스마다 −0.39 ~ +19.26으로 크게 갈린다는 관찰이 사이클 2의 근거가 되었다.
Bicycle이 IoU 0인 것은 가설 1-B의 근거가 되었다.

### 가설 1-B. 희귀 클래스는 노출을 늘리면 학습된다

**근거**
1. Bicycle IoU가 RGB 단독과 Sum 모두 0이다. [classiou_rgb_vs_sum.csv](../results/01_baseline/classiou_rgb_vs_sum.csv)
2. train에서 Bicycle은 10장, 전체 픽셀의 0.0017%로 14개 중 최소이고, Car는 928장이다. [dist_train_class.csv](../results/01_baseline/dist_train_class.csv)
3. 정답 Bicycle 픽셀의 68.41%가 Car로 예측된다 (RGB 단독은 43.46%). [confusion_class7_12_rgb_vs_sum.csv](../results/01_baseline/confusion_class7_12_rgb_vs_sum.csv)
4. mIoU가 클래스 평균이라 Bicycle 한 클래스가 전체의 1/14을 차지한다. [과제의 평가 조건](#과제가-고정한-조건)

**가설** Bicycle이 들어 있는 이미지를 더 자주 보여 주면 Bicycle IoU가 오르고 전체 mIoU도 오른다.

**근거 → 가설** 융합 방식과 무관하게 0이므로(근거 1) IR이 해결해 줄 문제가 아니라 클래스 자체의 문제다.
오답이 train에서 훨씬 흔한 Car로 쏠리는 것(근거 2, 3)은 노출이 적은 클래스가 다수 클래스에 흡수되는 전형적인 모양이다.
이 클래스가 점수의 1/14을 차지하므로(근거 4) 풀 가치가 있다.

**설계** B0 Sum에서 `train_oversample.py`로 Bicycle이 들어 있는 train 10장의 샘플링 가중치만 11로 올려 한 epoch에서 뽑히는 비율을 약 10%로 맞췄다
(10w / (990 + 10w) = 0.10, w = 11. 복원 추출). 모델과 손실은 그대로이고 가중치는 train 라벨로만 만든다. (#3)

**검증 지표** 사전 지표는 없고, 대상 클래스인 Bicycle IoU가 0에서 벗어나는지와 전체 mIoU가 떨어지지 않는지로 판정한다.

**결과** Bicycle 0 → 43.40, 전체 65.48 → 67.97 (+2.49).
한계: val에서 Bicycle이 나오는 이미지가 2장뿐이라 43.40의 변동 폭이 크다. [dist_class12_val_images.csv](../results/01_baseline/dist_class12_val_images.csv)
다른 클래스에 미친 영향은 클래스별로 비교하지 않았다.

**판정** 지지 (val 2장의 한계 포함)

**다음으로** B0에서 오른 이 방식이 test에서도 통하는지, 최종 설정에서도 통하는지는 확인하지 못한 채 남았다. 사이클 5의 근거가 되었다.

## 사이클 2. 융합 구조와 초기화

### 가설 2-A. 학습형 융합이 고정 Sum보다 낫다

**근거**
1. IR 기여가 클래스마다 −0.39(TrafficSign) ~ +19.26(Person)으로 크게 다르다. [classiou_rgb_vs_sum.csv](../results/01_baseline/classiou_rgb_vs_sum.csv)
2. Sum은 두 특징을 모든 클래스, 모든 위치에 1:1로 더한다. [fusion.py](../fusion_seg/fusion.py)의 `SumFusion`

**가설** 두 모달리티의 비중을 학습하는 융합(Concat, Adaptive)이 Sum보다 mIoU가 높다.

**근거 → 가설** IR이 도움이 되는 정도가 클래스마다 크게 다르다면(근거 1) 모든 곳에 같은 비율을 쓰는 Sum(근거 2)은 그 차이를 반영하지 못한다.
비중을 학습하면 이 편차를 활용할 여지가 있다고 보았다.

**설계** B0, 384×512에서 융합만 바꿨다. Concat은 두 특징을 이어 붙인 뒤 1×1 conv로 채널을 줄이고, Adaptive는 채널별 게이트로 두 특징을 섞는다.
Adaptive 게이트는 0으로 초기화해 시작 출력이 Sum과 같다. (#4, #5)

**검증 지표** 사전 지표는 없고, val mIoU를 Sum과 비교한다. 단일 seed이므로 0.5pp 이내 차이는 순위를 말하지 않는다.

**결과** Concat 62.47 (Sum 대비 −3.01), Adaptive 65.35 (−0.13). 둘 다 Sum 65.48을 넘지 못했다.
Adaptive는 Concat보다 모든 클래스에서 같거나 높다. [classiou_concat_vs_adaptive.csv](../results/02_fusion/classiou_concat_vs_adaptive.csv)

**판정** 기각

**다음으로** 학습형 융합이 Sum보다 낫다는 증거가 없으므로 **융합은 Sum으로 고정**하고, 이후 사이클은 모두 Sum 위에서 진행했다.
Concat이 Sum보다 낮은 이유는 가설 2-B로 따로 확인했다.

### 가설 2-B. Concat이 낮은 원인은 랜덤 초기화다

**근거**
1. Concat은 파라미터가 7.23M으로 Sum의 7.04M보다 많은데 val이 3.01 낮다. 용량 부족으로는 설명되지 않는다. [report.md](../results/report.md#2-전체-실험-결과)의 #2, #4
2. 융합 출력이 다음 RGB stage의 입력이다. [model.py 98~102행](../fusion_seg/model.py#L98-L102)
3. Sum의 시작 출력은 `x + a`이고, Concat의 1×1 conv는 랜덤 초기화라 시작 출력이 `x + a`가 아니다. [fusion.py](../fusion_seg/fusion.py)의 `ConcatConvFusion`

**가설** Concat이 낮은 것은 랜덤 초기화된 1×1 conv가 학습 초반에 사전학습된 RGB 특징을 깨뜨리기 때문이다.

**근거 → 가설** 파라미터가 더 많은데도 낮으므로 용량이 원인일 수 없다(근거 1).
융합 출력이 다음 stage의 입력이라(근거 2) 융합 모듈이 처음에 엉뚱한 값을 내면 ImageNet으로 사전학습된 stage가 곧바로 영향을 받고, Concat은 정확히 그렇게 시작한다(근거 3).
이 설명이 맞다면 격차는 초반에 가장 크고 학습이 진행되며 줄어야 한다.

**설계** 시작 출력이 `x + a`가 되도록 `Concat_res`를 만들었다: Sum에 concat 분기를 잔차로 더하고, 분기의 마지막 conv를 0으로 초기화한다. (#6)
Concat 대비 바뀐 것은 시작 출력만이 아니다. 잔차 형태이고 분기 안에 BatchNorm, GELU, 두 번째 conv가 들어 있다.
그래서 초기화 하나의 효과를 분리한 실험은 아니다. 이를 보완하려고 아래 검증 지표를 시간 축으로 잡고, 구조가 다른 Adaptive로 교차 확인했다.

**검증 지표** (Concat_res − Concat)의 epoch별 격차가 학습 **초반에 최대**이고 뒤로 갈수록 줄어야 한다.

**결과** 62.47 → 64.95 (+2.48). 격차는 ep1 +1.46, **ep4 +9.74(최대)**, ep10 +3.14, ep20 +3.41, ep30 +2.56이고 모든 epoch에서 양수다.
단조 감소는 아니다(ep10 +3.14 → ep20 +3.41). 교차 확인으로 구조가 다른 Adaptive − Concat도 ep4에서 +9.30으로 최대다.
[run_gap_concat_res_vs_concat.csv](../results/02_fusion/run_gap_concat_res_vs_concat.csv), [curve_gap_concat_res_vs_concat.png](../results/02_fusion/curve_gap_concat_res_vs_concat.png), [run_gap_adaptive_vs_concat.csv](../results/02_fusion/run_gap_adaptive_vs_concat.csv)

**판정** 지지

**다음으로** 융합을 더 복잡하게 만들어도 Sum 수준이 한계이고, 중요한 것은 사전학습 특징을 보존하는 시작점이라는 점이 확인되었다. 구조는 Sum으로 두고 다른 요소로 넘어갔다.

### 가설 2-C. 입력 단계에서 합쳐도 IR 이득을 얻는다

**근거**
1. 융합 위치(early, sum, 후기) 비교가 과제의 개선 아이디어 4번이다. [루트 README](../README.md)의 "개선 아이디어"
2. 제공 `model.py`의 Early는 입력에서 RGB와 IR을 합치고 인코더를 하나만 쓴다. 파라미터가 3.72M으로 RGB 단독(3.7M)과 같은 수준이다. [model.py](../fusion_seg/model.py)
3. 사이클 1에서 단계별 융합(Sum) 한 가지만 비교한 상태다. [01_baseline](../results/01_baseline/README.md)

**가설** 인코더 하나로 입력 단계에서 합쳐도 Sum에 가까운 이득을 얻는다. 파라미터는 절반이다.

**근거 → 가설** IR의 이득이 정보 자체에서 오는 것이라면 합치는 위치와 상관없이 얻어져야 하고, Early는 비용이 가장 낮은 구조다(근거 2). 위치 비교가 아직 비어 있었다(근거 3).

**설계** B0, 384×512에서 `--fusion early`만 다르게 학습한다. (#7)

**검증 지표** 사전 지표는 없고, Sum과 RGB 단독 대비 이득으로 판정한다.

**결과** 62.19. Sum보다 3.29 낮고 RGB 단독(61.53) 대비로는 +0.66이다.

**판정** 기각

**다음으로** IR 전용 인코더를 두는 구조(Sum의 두 인코더)가 필요하다고 보고 그대로 유지했다.
ImageNet RGB로 사전학습된 인코더 하나로는 IR 고유의 특징을 뽑기 어려웠을 것으로 해석한다(이 해석은 따로 검증하지 않았다).

## 사이클 3. 용량과 해상도

### 가설 3-A. B0의 병목은 모델 용량이다

**근거**
1. 융합 5종(Early, Concat, Concat_res, Adaptive, Sum)을 모두 바꿔도 62.19 ~ 65.48, 폭이 3.3pp에 그친다. [report.md](../results/report.md#2-전체-실험-결과)의 #2, #4, #5, #6, #7
2. 총 52M 상한 중 7.04M만 쓰고 있다. [report.md](../results/report.md#2-전체-실험-결과)의 #2와 [루트 README](../README.md)의 "규칙"

**가설** B0의 한계는 융합 방식이 아니라 모델 용량이다. 인코더를 B2로 키우면 오른다.

**근거 → 가설** 구조를 다섯 가지로 바꿔도 3.3pp 안에 있으면 구조는 큰 레버가 아니다(근거 1).
상한이 52M인데 7.04M만 쓰고 있어(근거 2) 용량은 비어 있는 레버다.

**설계** Sum과 입력 384×512를 고정하고 두 인코더를 B0 → B2로만 바꿨다. 51.55M으로 상한 안이다. (#8)

**검증 지표** 용량이 병목이라면 **전 클래스**가 오른다. 일부 클래스만 오른다면 용량이 아니라 그 클래스의 다른 문제다.

**결과** **14개 클래스 전부 상승**, 65.48 → 72.74 (+7.26, 전체 실험 중 최대). test는 63.90.
[classiou_b0_vs_b2.csv](../results/03_capacity/classiou_b0_vs_b2.csv)

**판정** 지지

**다음으로** 이득이 작은 물체에 몰려 있다는 관찰이 가설 3-B의 근거가 되었다.

### 가설 3-B. 입력 축소가 작은 물체를 지운다

**근거**
1. B2의 이득이 작은 물체에 집중되어 있다: Bicycle +39.83, TrafficLight +10.21, 반면 Sky +0.45, Road +1.31. [classiou_b0_vs_b2.csv](../results/03_capacity/classiou_b0_vs_b2.csv)
2. 384×512는 원본(600×800)의 0.64배다. [데이터 성질](#실험-전에-확인한-데이터-성질)

**가설** 해상도를 올리면 작은 물체 클래스가 오른다.

**근거 → 가설** 작은 물체에서 모델이 가장 약했고(근거 1), 입력이 가로세로 0.64배로 줄어 있어 얇거나 작은 물체는 몇 픽셀로 줄어들 수 있다(근거 2).
해상도는 파라미터 수와 무관하므로 용량 실험(3-A)과 따로 검증할 수 있다.
다만 근거 1만으로는 용량 증가의 효과와 해상도 부족을 구분하지 못하므로 해상도를 따로 바꿔 본다.

**설계** B2 Sum을 고정하고 입력만 384×512 → 480×640으로 올렸다. 원본의 0.8배이고, 3:4 비율과 32의 배수를 유지한다.
Colab T4에서 batch 8로 돌아가는 크기를 골랐고 여러 해상도를 탐색하지는 않았다. 파라미터는 51.55M으로 그대로다. (#9)

**검증 지표** **작은 클래스 위주**로 오르고, 큰 클래스(Road, Sky 등)는 거의 변하지 않는다.

**결과** Bicycle +6.77, Pole +3.43, Person +2.02, TrafficLight +1.32. Road +0.16, Sky +0.07로 큰 클래스는 거의 그대로다.
예외로 Bus −3.33, Truck −0.27이 내렸다. 전체 72.74 → 73.65 (val +0.91), test 63.90 → 66.13 (+2.23).
[classiou_384_vs_480.csv](../results/03_capacity/classiou_384_vs_480.csv)

**판정** 지지 (Bus 하락은 원인을 확인하지 못했다)

**다음으로** 이 구성(B2, 480×640, Sum)을 **최종 모델(#9)**로 정했다. 이 설정에서 IR이 여전히 필요한지가 사이클 4의 근거가 되었다.

## 사이클 4. 최종 설정에서 IR 기여 분리

### 가설 4. 최종 설정에서도 IR이 기여하고, 기여는 어두운 장면에서 크다

**근거**
1. IR의 효과는 B0에서만 확인했다 (+3.95). [classiou_rgb_vs_sum.csv](../results/01_baseline/classiou_rgb_vs_sum.csv)
2. 과제가 최종 설정에서의 RGB 단독 대비 IR 기여 ablation을 요구한다. [루트 README](../README.md)의 "규칙"
3. 데이터에 저조도 장면이 들어 있다. [루트 README](../README.md)의 "데이터" 항목

**가설** B2, 480×640에서도 Sum이 RGB 단독보다 높고, 그 차이는 RGB가 약한 어두운 장면일수록 크다.

**근거 → 가설** 백본이 강해지면 RGB만으로 충분해져 IR의 몫이 줄 수 있으므로 최종 설정에서 다시 재야 한다(근거 1, 2).
다만 Sum 모델은 RGB 단독보다 인코더가 하나 더 있어 파라미터가 27.36M에서 51.55M으로 늘어난다. 전체 mIoU 차이만으로는 IR의 정보 효과와 용량 효과가 섞인다.
이를 가르려고 밝기 구간별로 나눈다. 용량 효과라면 장면 밝기와 무관하게 비슷하게 오르고, IR이 조명 부족을 보완하는 것이라면 어두울수록 커진다(근거 3).

**설계** B2, 480×640, Sum 고정으로 `--fusion rgb`(#10)와 `--fusion sum`(#9)을 비교한다.
val 220장을 RGB 평균 밝기로 정렬해 73 / 73 / 74장 세 그룹으로 나눴다. 낮/밤 라벨이 아니라 이미지 밝기 기준이다.

**검증 지표** 용량 효과면 밝기와 무관, IR 정보 효과면 **어두울수록 격차가 커야** 한다.

**결과** 전체 70.99 → 73.65 (+2.66).
밝기별 IR 기여는 어두움 **+4.64**, 중간 +2.80, 밝음 **+0.89**. [miou_brightness_rgb_vs_sum.csv](../results/04_ir_contribution/miou_brightness_rgb_vs_sum.csv)
클래스별로는 Person +18.79(어두운 장면만 보면 42.32 → 65.82), TrafficSign −0.75, Bus −2.60.
[classiou_rgb_vs_sum.csv](../results/04_ir_contribution/classiou_rgb_vs_sum.csv), [classiou_dark_rgb_vs_sum.csv](../results/04_ir_contribution/classiou_dark_rgb_vs_sum.csv)
epoch별 IR 기여는 ep1 −3.63, ep5 +2.07, ep10 +2.41, ep20 +3.36, ep30 +2.66이다. [run_gap_b2_sum_480_vs_b2_rgb_480.csv](../results/04_ir_contribution/run_gap_b2_sum_480_vs_b2_rgb_480.csv)

**판정** 지지. 남은 교란: 파라미터가 함께 늘어난 효과는 밝기 분해로 뒷받침했을 뿐 직접 분리하지 못했다 ([측정 한계](#측정-한계)).

**다음으로** IR 기여가 확인되어 최종 모델의 구조는 그대로 두었다. 마지막으로 샘플링(가설 1-B)을 최종 설정에서 다시 확인했다.

## 사이클 5. 최종 설정에서 샘플링

### 가설 5. Bicycle oversampling은 최종 설정에서도 val과 test를 함께 올린다

**근거**
1. B0에서 oversampling이 전체 +2.49, Bicycle 0 → 43.40을 만들었다. [01_baseline](../results/01_baseline/README.md)
2. 그 실험(#3)은 test 점수가 없다. [report.md](../results/report.md#4-리더보드-제출)의 제출 표
3. 평가가 클래스 평균이라 Bicycle이 1/14 가중치이고, val에서 Bicycle은 2장(994픽셀)뿐이다. [dist_class12_val_images.csv](../results/01_baseline/dist_class12_val_images.csv)

**가설** 최종 설정(B2, 480×640)에서도 Bicycle oversampling이 val과 test mIoU를 함께 올린다.

**근거 → 가설** B0의 이득은 test로 검증된 적이 없다(근거 2).
Bicycle은 1/14 가중치를 가지면서 val에서 2장으로만 측정되므로(근거 3) val 상승이 실제 개선인지 측정 변동인지 가르려면 test에서 같은 방향인지 봐야 한다.

**설계** #9와 같은 설정에서 1-B와 같은 방식(Bicycle 포함 10장 가중치 11, 목표 비율 10%)만 추가한다. (#11)
test 점수는 모델 선택에 쓰지 않고, val에서 한 판단이 일반화되는지 확인하는 데에만 본다.

**검증 지표** val과 test가 **같은 방향**으로 움직인다.

**결과** val +0.81 (73.65 → 74.46), test **−0.49** (66.13 → 65.64). 방향이 갈렸다.
- 원인 분해(val): 상승 +0.81 중 Bicycle 기여가 +1.14, 나머지 13개 클래스 합계가 −0.33이다. Bicycle은 +15.98이고 8개 클래스가 내렸다(TrafficSign −2.10, Motorcycle −1.69, Truck −1.38 등). [classiou_sum_vs_oversample.csv](../results/05_oversampling_final/classiou_sum_vs_oversample.csv)
- 교차 확인: Bicycle을 뺀 val 13클래스 mIoU가 75.73 → 75.37 (−0.36)이고, test −0.49와 방향과 크기가 비슷하다. val 안에서도 Bicycle 외 클래스는 오르지 않았다.
- val−test 격차는 #9가 7.52, #11이 8.82로 벌어졌다.
- epoch별 val 격차(#11 − #9)는 epoch 3을 뺀 29개 epoch에서 양수이고 epoch 6의 +6.34에서 시작해 ep30 +0.81로 줄었다. [run_gap_b2_sum_480_oversample10_vs_b2_sum_480.csv](../results/05_oversampling_final/run_gap_b2_sum_480_oversample10_vs_b2_sum_480.csv)

상세는 [05_oversampling_final](../results/05_oversampling_final/README.md).

**판정** 기각

**다음으로** val 상승이 val 2장짜리 한 클래스에서만 나왔고 나머지 13클래스는 val에서도 오르지 않았으므로(test는 이를 확인했을 뿐이다) **최종 모델은 #9를 유지**했다.
#11은 val이 가장 높지만 최종 모델로 쓰지 않았다.

## 기각된 가설과 결정

기각이 세 건이고, 각각 최종 모델의 구성을 결정했다.

| 기각 | 결정 |
|---|---|
| 2-A 학습형 융합이 Sum보다 낫다 | 융합은 Sum으로 고정. 이후 모든 사이클이 Sum 위에서 진행되었다 |
| 2-C 입력 단계 융합으로 충분하다 | IR 전용 인코더를 둔 두 인코더 구조를 유지 |
| 5 최종 설정에서 oversampling이 전체를 올린다 | 최종 모델은 #9로 유지, #11은 사용하지 않음 |

## 리더보드 제출

11개 중 4개를 올렸다.

| 제출 ID | 모델 | val | test | 비고 |
|---|---|---:|---:|---|
| 1058499 | #5 B0 Adaptive | 65.35 | 55.84 | 학습형 융합 중 val 최고 (Concat 62.47, Concat_res 64.95) |
| 1060429 | #8 B2 Sum 384×512 | 72.74 | 63.90 | 제출 시점 val 최고 갱신 |
| 1061666 | #9 B2 Sum 480×640 | **73.65** | **66.13** | **최종 모델**, val 최고 갱신 |
| 1062714 | #11 B2 Sum 480×640 + Oversampling | 74.46 | 65.64 | 사이클 5 검증. 최종 모델로 쓰지 않음 |

#8, #9, #11은 올린 시점에 val 최고를 갱신한 모델이다. 모델 선택은 val로만 했고, test 점수에 맞춘 조정은 없다.
#3(val 67.97)은 #5(65.35)보다 val이 높았지만 제출하지 않았고, 그 사유는 미확정이다.

## 측정 한계

- **단일 seed.** 전 실험이 seed 0이다. 0.5pp 이내 차이는 순위를 말하지 않았고(Sum 65.48, Adaptive 65.35, Concat_res 64.95), 주장한 차이는 IR +3.95, B2 +7.26, Concat −3.01처럼 훨씬 큰 것들이다.
- **30 epoch는 수렴점이 아니라 고정한 예산이다.** 로그가 있는 run은 모두 epoch 20 이후 마지막까지 +1.2 이상 오르고 있고 best epoch가 27~30이다. [run_convergence.csv](../results/00_overview/run_convergence.csv)
- **val과 test의 격차.** #9는 val 73.65, test 66.13으로 7.5pp 차이가 난다. 모델 순서는 val과 test에서 같았다 (#5 < #8 < #9).
- **Bicycle 평가.** val에 2장뿐이라 Bicycle IoU는 변동이 크고, 사이클 1-B와 5의 결론은 이 한계 안에서만 읽어야 한다.
- **IR 기여의 용량 교란.** 사이클 4의 Sum은 RGB 단독보다 인코더가 하나 많다. 보조 인코더의 입력만 RGB로 바꾼 같은 구조를 한 번 더 돌리면 분리되지만(run 1개, 약 56분) 아직 돌리지 않았다.
- **융합 비교는 B0에서만.** B2에서도 Sum이 최선인지는 보지 않았다.
- **가설 2-B의 변인.** Concat_res는 초기화 외에 잔차 형태와 분기 구조도 Concat과 다르다. 초반 격차와 Adaptive 교차 확인으로 보완했다.
- **탐색하지 않은 축.** 손실은 전 실험 CE 단독이다 (CE + Lovász 등 미비교). oversampling 비율은 10% 한 가지만 썼다.
- **밝기 그룹은 날씨나 시간대 라벨이 아니다.** RGB 평균 밝기로 나눈 세 구간이다.
- **해결하지 못한 오류.** 젖은 노면과 야간 조명에서 Sidewalk ↔ Road 혼동이 남아 있다. [06_final_model](../results/06_final_model/README.md)
