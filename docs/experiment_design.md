# Experiment Design

- Scope: 트랙 A, FMB RGB+IR, 14개 클래스, 실험 11개
- Purpose: 실험 순서와 선택 근거 정리
- Full results: [results/report.md](../results/report.md)
- Workflow: 제공 베이스라인 재현 → 결과 분석 → 후속 실험 설계
- Control: 한 번에 한 요소만 변경
- Format: Evidence → Hypothesis → Setup → Result → Interpretation → Decision

- Epoch index: 1-based
- Class order: 0 Road, 1 Sidewalk, 2 Building, 3 TrafficLight, 4 TrafficSign, 5 Vegetation, 6 Sky, 7 Person, 8 Car, 9 Truck, 10 Bus, 11 Motorcycle, 12 Bicycle, 13 Pole

## 출발점

### 고정 조건

| Item | Condition | Experimental constraint |
|---|---|---|
| Split | train 1,000 / val 220 / test 280 | 모델 선택은 val 기준, test는 최종 확인용 |
| Metric | 14-class mIoU | 클래스별 동일 가중치 1/14 |
| Model size | 각 인코더 MiT-B2 이하, 총 52M 이하 | 최대 구성 B2 + B2, 51.55M |
| Pretraining | ImageNet pretrained MiT만 허용 | 외부 데이터 및 다른 사전학습 제외 |
| Environment | Colab T4 1장 | 해상도와 batch size 제한 |
| Required ablation | RGB-only vs RGB+IR | IR 기여 분리 필요 |
| Reproducibility | `args.json`, `log.jsonl`, 재실행 검증 | run별 설정과 로그 보존 |

### 공통 학습 설정

| Item | Setting |
|---|---|
| Seed | 0 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | encoder 6e-5 / decoder 6e-4 |
| Weight decay | 0.01 |
| Schedule | warm-up 2 epoch + poly 0.9 |
| Training budget | 30 epoch |
| Augmentation | scale 0.75–1.5, random crop, horizontal flip 0.5 |
| Model selection | best val mIoU |

- 비교당 변경 요소 1개
- 동일 train/val split 유지
- #6, #9의 Colab 종료 후 `last.pth` 기반 재개

### 데이터 확인

| Item | Observation | Design implication |
|---|---|---|
| Original resolution | 600×800 | 384×512와 480×640 비교 필요 |
| Bicycle train exposure | 10장, 전체 픽셀의 0.0017% | 희귀 클래스 노출 부족 가능성 |
| Bicycle val exposure | 2장, 총 994픽셀 | Bicycle IoU의 높은 변동성 |
| Input scale | 384×512는 원본의 0.64배, 480×640은 0.8배 | 작은 물체 손실 가능성 |

### 제공 베이스라인

- Architecture: MiT-B0, 384×512
- Instructor test: RGB-only 52.77 / Sum 56.30
- Reproduced val: RGB-only 61.53 / Sum 65.48
- Instructor val reference: RGB-only 61.50 / Sum 65.40
- Conclusion: 기준선 재현 완료

## 사이클 1. 기준선과 희귀 클래스

### 가설 1-A. IR은 RGB를 보완한다

Evidence

- 제공 베이스라인의 test 이득 +3.53pp
- 안개·폭우·저조도 중심의 FMB 환경
- RGB-only 대비 IR 기여 ablation 요구

Hypothesis

- Stage-wise Sum fusion의 RGB-only 대비 mIoU 상승

Setup

- Backbone: MiT-B0
- Input: 384×512
- Comparison: `--fusion rgb` vs `--fusion sum`
- Runs: #1, #2

Criterion

- 전체 val mIoU 차이
- 클래스별 IoU 변화

Result

- Overall: 61.53 → 65.48, +3.95pp
- Improved classes: 12/14
- Largest gain: Person +19.26pp
- Negative change: TrafficSign −0.39pp
- Bicycle: 두 모델 모두 IoU 0

Interpretation

- IR 보완 효과 확인
- 클래스별 IR 기여 편차 확인
- Bicycle 성능 정체 확인

Decision

- 융합 구조 비교 진행
- Bicycle 문제 별도 분리

### 가설 1-B. 희귀 클래스는 노출을 늘리면 학습된다

Evidence

- RGB-only와 Sum 모두 Bicycle IoU 0
- Bicycle 포함 train 이미지 10장
- Bicycle 비중 전체 train 픽셀의 0.0017%
- Bicycle 정답 픽셀의 Car 오분류 집중
- 클래스당 동일 mIoU 가중치 1/14

Hypothesis

- Bicycle 포함 이미지의 노출 증가에 따른 Bicycle IoU 및 전체 mIoU 상승

Setup

- Base model: B0 Sum
- Sampler: `WeightedRandomSampler`, 복원 추출
- Positive images: 10장
- Positive weight: 11
- Target fraction: 10%
- Run: #3

Criterion

- Bicycle IoU 0 탈출 여부
- 전체 mIoU 유지 여부

Result

- Bicycle: 0 → 43.40
- Overall: 65.48 → 67.97, +2.49pp

Interpretation

- val 기준 가설 지지
- 희귀 클래스 노출 부족 가능성 확인
- Bicycle 포함 val 2장에 따른 높은 불확실성
- 다른 클래스 영향 분석 부족

Decision

- B0 결과의 test 제출 보류
- 최종 설정에서 동일 sampling 재검증

## 사이클 2. 융합 구조와 초기화

### 가설 2-A. 학습형 융합이 고정 Sum보다 낫다

Evidence

- 클래스별 IR 기여 −0.39pp에서 +19.26pp
- 모든 채널과 위치에 동일 비중을 적용하는 Sum
- 모달리티별 중요도 학습 가능성

Hypothesis

- Concat 또는 Adaptive fusion의 Sum 대비 mIoU 상승

Setup

- Backbone: MiT-B0
- Input: 384×512
- Fixed factor: 학습 설정 전체
- Changed factor: fusion module
- Concat: channel concatenation + 1×1 convolution
- Adaptive: channel-wise modality gate, Sum-equivalent initialization
- Runs: #4, #5

Criterion

- Sum 대비 val mIoU
- 단일 seed를 고려한 0.5pp 이내 차이의 우열 판단 제외

Result

- Sum: 65.48
- Concat: 62.47, −3.01pp
- Adaptive: 65.35, −0.13pp
- Adaptive의 Concat 대비 13개 클래스 우위, Bicycle 동률

Interpretation

- 학습형 융합 우위 가설 기각
- Adaptive와 Sum의 차이는 판단 보류 범위
- Concat의 뚜렷한 성능 저하

Decision

- 후속 실험의 Sum fusion 고정
- Concat 저하 원인 별도 확인

### 가설 2-B. Concat이 낮은 원인은 랜덤 초기화다

Evidence

- Concat 7.23M, Sum 7.04M
- 더 많은 파라미터에도 Concat −3.01pp
- 사전학습 특징 뒤에 추가된 랜덤 초기화 1×1 convolution
- Sum과 다른 초기 출력

Hypothesis

- 랜덤 초기화에 따른 학습 초반의 사전학습 특징 훼손
- 초기 격차 최대 후 점진적 감소

Setup

- Concat residual branch 추가
- Initial output: `x + a`
- Last convolution: zero initialization
- Run: #6
- Cross-check: Adaptive와 Concat의 epoch별 격차

Criterion

- Concat_res − Concat의 epoch별 차이
- 학습 초반 최대 격차 여부

Result

- Concat: 62.47
- Concat_res: 64.95, +2.48pp
- Gap: ep1 +1.46 / ep4 +9.74 / ep10 +3.14 / ep20 +3.41 / ep30 +2.56
- Adaptive − Concat 최대 격차: ep4 +9.30pp

Interpretation

- 초기화 영향 가능성 지지
- 학습 초반 최대 격차 확인
- 단조 감소 아님
- 초기화 외 잔차 구조·BatchNorm·GELU 차이 존재
- 단일 원인 분리 불가

Decision

- Concat 계열 추가 확장 중단
- Sum fusion 유지

### 가설 2-C. 입력 단계에서 합쳐도 IR 이득을 얻는다

Evidence

- Early fusion의 단일 인코더 구성
- Early 3.72M, RGB-only 3.70M
- Dual encoder 대비 약 절반의 모델 크기

Hypothesis

- Early fusion의 RGB-only 대비 상승 및 Sum 근접 성능

Setup

- Backbone: MiT-B0
- Input: 384×512
- Fusion: `--fusion early`
- Run: #7

Criterion

- RGB-only 및 Sum 대비 val mIoU

Result

- Early: 62.19
- RGB-only 대비 +0.66pp
- Sum 대비 −3.29pp

Interpretation

- 제한적인 IR 이득
- Sum 근접 가설 기각
- 단일 RGB-pretrained encoder의 IR 특징 추출 한계 가능성
- 원인 검증 없음

Decision

- RGB·IR dual encoder 구조 유지

## 사이클 3. 용량과 해상도

### 가설 3-A. B0의 병목은 모델 용량이다

Evidence

- B0 fusion 5종의 val 범위 62.19–65.48
- Fusion 변경 최대 폭 약 3.3pp
- B0 Sum 7.04M
- 허용 상한 약 52M

Hypothesis

- Fusion 방식보다 backbone capacity의 더 큰 영향
- B0에서 B2 변경 시 전반적 성능 상승

Setup

- Fusion: Sum 고정
- Input: 384×512 고정
- Backbone: MiT-B0 → MiT-B2
- Parameters: 7.04M → 51.55M
- Run: #8

Criterion

- 전체 mIoU 변화
- 다수 클래스의 동일 방향 상승

Result

- Overall: 65.48 → 72.74, +7.26pp
- Test: 63.90
- Class-wise: 14개 클래스 전체 상승
- Largest gain: Bicycle +39.83pp

Interpretation

- 전체 실험 중 최대 향상
- B0 capacity bottleneck 가설 지지
- 희귀·소형 클래스의 큰 상승폭

Decision

- MiT-B2 유지
- 입력 해상도 효과 추가 확인

### 가설 3-B. 입력 축소가 작은 물체를 지운다

Evidence

- Bicycle +39.83, TrafficLight +10.21의 큰 B2 이득
- Sky +0.45, Road +1.31의 제한적 변화
- 384×512 입력의 원본 대비 0.64배 축소

Hypothesis

- 입력 해상도 증가에 따른 작은 물체 클래스 중심의 IoU 상승

Setup

- Backbone: MiT-B2 고정
- Fusion: Sum 고정
- Input: 384×512 → 480×640
- Original scale: 0.64 → 0.8
- Parameters: 51.55M 유지
- Run: #9

Criterion

- 작은 물체 클래스와 큰 영역 클래스의 IoU 변화 비교

Result

- Overall val: 72.74 → 73.65, +0.91pp
- Test: 63.90 → 66.13, +2.23pp
- Gains: Bicycle +6.77, Pole +3.43, Person +2.02, TrafficLight +1.32
- Stable classes: Road +0.16, Sky +0.07
- Negative changes: Bus −3.33, Truck −0.27

Interpretation

- 작은 물체 중심의 상승
- 큰 영역 클래스의 변화 미미
- 가설 대체로 지지
- Bus 하락 원인 미확인

Decision

- B2, Sum, 480×640을 submission candidate로 선정
- 최종 설정에서 IR 기여 재측정

## 사이클 4. 최종 설정에서 IR 기여 분리

### 가설 4. 최종 설정에서도 IR이 기여하고, 기여는 어두운 장면에서 크다

Evidence

- B0에서 확인된 IR 이득 +3.95pp
- 최종 설정의 RGB-only ablation 필요
- 저조도 장면에서 RGB 정보 감소 가능성

Hypothesis

- B2, 480×640에서도 RGB+IR의 RGB-only 대비 상승
- 어두운 장면에서 더 큰 IR 기여

Setup

- Backbone: MiT-B2
- Input: 480×640
- Comparison: RGB-only #10 vs Sum #9
- Brightness split: RGB 평균 밝기 기준 73 / 73 / 74장
- Groups: dark / middle / bright

Criterion

- 전체 mIoU 차이
- 밝기 그룹별 RGB+IR 이득
- 어두운 그룹의 최대 이득 여부

Result

- Overall: 70.99 → 73.65, +2.66pp
- Dark: +4.64pp
- Middle: +2.80pp
- Bright: +0.89pp
- Person: +18.79pp
- Dark-group Person: 42.32 → 65.82, +23.50pp
- Negative changes: TrafficSign −0.75, Bus −2.60
- Epoch gap: ep1 −3.63 / ep5 +2.07 / ep10 +2.41 / ep20 +3.36 / ep30 +2.66

Interpretation

- 최종 설정의 IR 기여 확인
- 어두운 장면일수록 큰 상승폭
- 저조도 보완 가설 지지
- RGB-only 27.36M과 Sum 51.55M의 capacity 차이 존재
- IR 정보 효과와 추가 인코더 효과의 완전한 분리 불가

Decision

- RGB+IR dual encoder 유지
- 최종 설정에서 oversampling 재검증

## 사이클 5. 최종 설정에서 샘플링

### 가설 5. Bicycle oversampling은 최종 설정에서도 val과 test를 함께 올린다

Evidence

- B0 oversampling의 overall +2.49pp
- B0 Bicycle IoU 0 → 43.40
- B0 oversampling의 test 미제출
- Bicycle 포함 val 이미지 2장

Hypothesis

- B2, 480×640에서도 oversampling에 따른 val과 test 동반 상승

Setup

- Base model: #9 B2 Sum 480×640
- Positive images: 10장
- Positive weight: 11
- Target fraction: 10%
- Run: #11

Criterion

- val과 test mIoU의 변화 방향
- Bicycle과 나머지 13개 클래스의 기여 분리

Result

- Val: 73.65 → 74.46, +0.81pp
- Test: 66.13 → 65.64, −0.49pp
- Bicycle: +15.98pp
- Bicycle의 전체 mIoU 기여: +1.14pp
- 나머지 13개 클래스 합계 기여: −0.33pp
- Bicycle 제외 13-class mIoU: 75.73 → 75.37, −0.36pp
- Negative classes: 8개
- Val–test gap: 7.52 → 8.82pp

Interpretation

- val과 test의 반대 방향
- val 상승의 Bicycle 집중
- 나머지 클래스의 평균 하락
- 최종 설정에서 가설 기각
- Bicycle 포함 val 2장에 따른 과대평가 가능성

Final decision

- #11 제외
- #9 유지
- Final model: MiT-B2, Sum, 480×640

## 결과가 예상과 달랐던 실험

| Hypothesis | Result | Decision |
|---|---|---|
| 2-A. 학습형 fusion > Sum | Adaptive −0.13pp, Concat −3.01pp | Sum 고정 |
| 2-C. Early fusion ≈ Sum | Early −3.29pp | Dual encoder 유지 |
| 5. Oversampling의 val·test 동반 상승 | val +0.81pp, test −0.49pp | #11 제외, #9 유지 |

## 리더보드 제출

| Submission ID | Model | Val | Test | Purpose |
|---|---|---:|---:|---|
| 1058499 | #5 B0 Adaptive | 65.35 | 55.84 | 학습형 fusion 대표 제출 |
| 1060429 | #8 B2 Sum 384×512 | 72.74 | 63.90 | Backbone scaling 확인 |
| 1061666 | #9 B2 Sum 480×640 | 73.65 | 66.13 | Final model |
| 1062714 | #11 B2 Sum 480×640 + Oversampling | 74.46 | 65.64 | Sampling 일반화 확인 |

- Model selection: val mIoU 기준
