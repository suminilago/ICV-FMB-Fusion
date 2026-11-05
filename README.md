# 트랙 A: FMB 악천후 RGB–열화상(IR) 융합 세그멘테이션

| 항목 | 내용 |
|---|---|
| 데이터 | FMB (RGB + 열화상, 안개·폭우·저조도 포함) |
| 클래스 | 14개 (255 = 무시) |
| 분할 | 공식 train 1,220장 → train 1,000 / val 220 (고정 시드), 공식 test 280장 → 리더보드용 |
| 모델 | SegFormer MiT-B0 두 갈래(RGB / IR), 단계마다 융합 (베이스라인: 합산) |
| 환경 | Colab T4 (무료) 한 장 |

## 폴더 구성
- `notebooks/trackA_FMB_baseline.ipynb` : 위에서부터 실행하면 데이터 준비 → 학습 → 제출 파일까지 진행
- `fusion_seg/model.py` : 융합 모델 (`--fusion rgb | early | sum`)
- `fusion_seg/fusion.py` : 단계별 융합 모듈 등록부. 여기에 새 아이디어를 추가하고 `FUSIONS`에 등록
- `fusion_seg/data.py` : 데이터셋과 증강
- `train.py` : 학습 (세션이 끊기면 같은 명령을 다시 실행하면 이어서 학습)
- `predict.py` : test 예측 PNG와 `submission.zip` 생성
- `tools/download_fmb.py`, `tools/prepare_fmb.py` : 다운로드, 분할 인덱스 생성

## 실행 순서
1. Colab에서 GPU(T4)를 선택하고 이 폴더를 `vip_trackA_FMB.zip`으로 올린 뒤 노트북을 엽니다.
2. 데이터는 저자 Google Drive에서 직접 내려받습니다 (재배포 없음). 라이선스와 인용은 원 저자 페이지를 따르세요.
3. `prepare_fmb.py` 출력의 `label values`를 확인합니다. (실제 FMB에서 확인: 0 = 미라벨, 1..14 = 클래스이며, 기본값 `--label_shift 1`로 0..13에 대응. 폴더 `color`는 컬러 라벨 시각화이므로 입력으로 쓰지 않고 `Visible`(RGB)과 `Infrared`(IR, 3채널)를 사용)
4. `--fusion rgb`(RGB 단독)와 `--fusion sum`(융합)을 모두 학습해 기록합니다.
5. `predict.py`로 만든 `submission.zip`을 리더보드에 제출합니다.

## 규칙
- 분할 파일(`index_*.json`)이 정의한 train/val만 학습·튜닝·모델 선택에 사용합니다. 원 데이터의 test 폴더는 라벨이 공개되어 있어도 사용하지 않습니다.
- 외부 데이터 금지. 사전학습은 ImageNet 사전학습 MiT만 허용 (기본 `nvidia/mit-b0`).
- 모델 크기: RGB·보조 모달 두 인코더 모두 MiT-B2 이하 (융합 모듈 포함 총 파라미터 약 52M 이하). `train.py`가 출력하는 `params:` 값으로 확인합니다.
- 백본을 바꿔 학습해도 `best.pth`에 구조 설정(`enc_config`)이 함께 저장되어 `predict.py`가 같은 구조로 복원합니다. 제출 PNG의 값은 0 ~ (클래스 수 − 1)이어야 하며, 범위를 벗어난 값이 있으면 채점이 오류로 처리됩니다.
- 상위 제출 팀은 코드·학습 로그·설정 파일(`runs/<이름>/args.json`, `log.jsonl`)을 제출하며, 교수가 재실행해 점수를 검증합니다. 재현되지 않으면 점수를 인정하지 않습니다.
- 보고서에는 RGB 단독 대비 IR의 기여를 ablation 표로 제시합니다.

## 개선 아이디어 (자유롭게 조합)
1. 옵티마이저와 LR 스케줄 (`train.py`)
2. 증강 (`fusion_seg/data.py`): 색 지터는 RGB에만, IR은 별도 정규화
3. 손실 함수: 클래스 가중 CE, OHEM, Dice
4. 융합 위치·방식 비교: `early` / `sum` / 후기 융합
5. SE 블록 기반 채널 가중 융합
6. concat 후 1×1 conv 병목
7. 백본 교체 (`--backbone nvidia/mit-b1` 등)
8. 전이학습 전략: IR 브랜치 초기화·동결 시점
9. skip connection, 멀티스케일 디코더
10. 규제(Dropout, weight decay, early stopping)와 추론 시 플립 TTA (`predict.py --flip`)

## 문제가 생기면
- Google Drive 쿼터 초과: 폴더를 내 Drive에 바로가기로 추가하고 Colab에서 Drive를 마운트한 뒤 `prepare_fmb.py --raw <마운트 경로>`로 실행
- `could not detect the layout`: 출력된 폴더 목록을 캡처해 교수에게 문의
- 사전학습 가중치 로드 경고(`encoder weights were NOT loaded`): `pip install transformers==4.46.3` 후 재시작
- 체크포인트는 학습한 환경과 같은 `transformers` 버전에서만 불러올 수 있습니다. 학습 후 `pip install`로 버전을 바꾸지 말고, 바꿨다면 다시 학습하세요 (`Missing key(s) in state_dict` 오류가 나옵니다).
- 보고서나 코드에 쓴 아이디어의 출처는 반드시 표기

## 출처
- FMB: https://github.com/JinyuanLiu-CV/SegMiF
- SegFormer: Hugging Face `transformers`, `nvidia/mit-b0`

---

## Track A: Class 12 Oversampling Experiment

### Objective

Improve rare-class segmentation performance using
RGB + infrared (IR) multimodal fusion.

### Experimental Results

| Experiment | Best Epoch | Validation mIoU |
|---|---:|---:|
| RGB Only | 29 | 61.53% |
| RGB + IR Sum | 30 | 65.48% |
| Sum + Class 12 Oversampling (10%) | 29 | 67.97% |

The proposed oversampling strategy improved validation
mIoU by **+2.49 percentage points** over RGB + IR Sum.

### Method

- Backbone: NVIDIA MiT-B0
- Fusion: Sum
- Parameters: 7.04M
- Target class: 12
- Sampling method: WeightedRandomSampler
- Target positive-image sampling fraction: 10%
- Training images: 1,000
- Class 12 positive images: 10
- Class 12 negative images: 990
- Positive sampling weight: 11.0
- Sampling with replacement

Only training labels are used to construct sampling weights.
Validation labels are used for evaluation only.

### Training

```bash
python train_oversample.py \
  --data_root data/FMB_prepared \
  --fusion sum \
  --backbone nvidia/mit-b0 \
  --epochs 30 \
  --seed 0 \
  --target_fraction 0.10 \
  --out runs/fmb_sum_oversample10
```

### Inference

```bash
python predict.py \
  --data_root data/FMB_prepared \
  --ckpt runs/fmb_sum_oversample10/best.pth \
  --split test \
  --out submission
```

### Rare-Class Results

| Metric | RGB + IR Sum | Oversampling |
|---|---:|---:|
| Class 12 IoU | 0.00% | 43.40% |
| Overall mIoU | 65.48% | 67.97% |

Class 12 appears in only 2 of the 220 validation images.
The rare-class improvement should therefore be interpreted
with this limited validation coverage in mind.

### Submission Verification

- Test predictions: 280 PNG files
- Class IDs: 0-13
- ZIP integrity: PASS
- Prediction filenames: PASS
- Validation mIoU reproduced with updated instructor code
