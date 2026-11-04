# 기말 프로젝트: 악천후 멀티모달 융합 세그멘테이션

두 개의 트랙 중 하나(또는 둘 다)를 선택해 베이스라인을 개선하고 리더보드에 제출합니다.

| 트랙 | 데이터 | 모달리티 | 클래스 | 난이도 |
|---|---|---|---|---|
| A (기본) | FMB | RGB + 열화상(IR), 안개·폭우·저조도 | 14 | 낮음 (Colab T4) |
| B (고급) | DELIVER 전면 뷰 부분집합 | RGB + LiDAR, 5개 날씨·조명 조건 | 25 | 중간 (다운로드 12.2GB) |

모든 데이터는 학생이 원 저자 사이트에서 직접 내려받습니다(재배포 없음). 데이터 라이선스와 인용 요구는 각 원 저자 페이지를 따르세요.

## 폴더 구성
- `fusion_seg/model.py` : SegFormer MiT-B0 기반 융합 모델 (`rgb`, `early`, `sum`)
- `fusion_seg/fusion.py` : 단계별 융합 모듈 레지스트리 (여기에 아이디어를 추가)
- `fusion_seg/data.py` : 데이터셋(증강 포함)
- `train.py`, `predict.py` : 학습, 제출용 예측 PNG·zip 생성
- `tools/` : 다운로드와 분할 스크립트
- `scoring/scoring.py` : 리더보드 채점기(교수용, 학생 로컬 검증에도 사용 가능)
- `notebooks/` : Colab 노트북 2개

## 학생: 실행 순서 (트랙 A)
1. Colab에서 GPU(T4)를 선택하고 `notebooks/trackA_FMB_baseline.ipynb`를 엽니다.
2. `vip_fusion_project.zip`을 업로드하고 위에서부터 실행합니다.
3. `--fusion rgb`(RGB 단독)와 `--fusion sum`(융합) 결과를 모두 기록합니다.
4. `submission.zip`을 리더보드에 제출합니다.

세션이 끊기면 같은 명령을 다시 실행하면 `runs/<이름>/last.pth`에서 이어서 학습합니다. 체크포인트는 Drive에 저장하는 것을 권장합니다(`--out /content/drive/MyDrive/...`).

## 학생: 규칙
- 분할 파일(`index_*.json`)이 정의한 train/val만 학습에 사용합니다. 원 데이터의 test 폴더는 라벨이 공개되어 있어도 학습·튜닝·모델 선택에 쓰지 않습니다.
- 외부 데이터 금지. 사전학습은 ImageNet 사전학습 MiT만 허용합니다(기본 `nvidia/mit-b0`).
- 제출 가능한 모델은 기본 구조 기준 MiT-B2 이하로 합니다(아이디어 7 백본 교체 포함).
- 상위 제출자는 코드·학습 로그·설정 파일을 제출하고, 교수가 재실행하여 점수를 검증합니다. 재현되지 않으면 리더보드 점수를 인정하지 않습니다.
- 보고서에는 RGB 단독 대비 보조 모달리티(IR 또는 LiDAR)의 기여를 ablation 표로 제시합니다.

## 개선 아이디어 (자유롭게 조합)
1. 옵티마이저와 LR 스케줄(`train.py`)
2. 증강(`fusion_seg/data.py`): 색 지터는 RGB에만, 보조 모달리티는 별도 정규화
3. 손실 함수: 클래스 가중 CE, OHEM, Dice
4. 융합 위치·방식 비교: `early` / `sum` / 후기 융합
5. SE 블록 기반 채널 가중 융합 (ICV 07 슬라이드 64–66)
6. concat 후 1×1 conv 병목 (ICV 07 슬라이드 40–45, 58)
7. 백본 교체: `--backbone nvidia/mit-b1` 등 (ICV 07 슬라이드 51–60, 69–72)
8. 전이학습 전략: 보조 브랜치 초기화·동결 시점 (ICV 07 슬라이드 72)
9. skip connection·멀티스케일 디코더
10. 규제(Dropout, weight decay, early stopping)와 추론 시 플립 TTA(`predict.py --flip`)

## 교수용: 사전 준비 (개강 전 1회)
1. 직접 한 번 완주합니다. 실제 FMB로 확인된 사항: 폴더는 train/test × {color, Infrared, Visible, Label}(train 1,220 / test 280), `color`는 컬러 라벨 시각화(입력 아님), Label 값은 0=미라벨 / 1..14=클래스이므로 `prepare_fmb.py`의 기본값 `--label_shift 1`을 그대로 사용합니다. IR은 3채널 PNG입니다.
   DELIVER도 실제 데이터로 확인했습니다: 라벨 PNG는 RGBA 4채널(클래스 값은 R 채널, 1..25)이고 LiDAR 투영은 8비트 1채널입니다. `prepare_deliver.py`는 R 채널만 저장하며 기본값 `--label_shift 1`을 사용합니다. 원본 tar는 약 13.1GB(13,117,650,134 bytes)입니다.
2. 숨은 테스트 라벨 생성 (교수 PC/Colab):
   - 트랙 A: `python tools/prepare_fmb.py --raw data/FMB_raw --out /tmp/p --export_ref ref_fmb`
   - 트랙 B: `python tools/prepare_deliver.py --out /tmp/d --export_ref ref_deliver`
3. 채점기 점검: `python scoring/scoring.py <input_dir> <output_dir>` (`input_dir/ref`, `input_dir/res`).
4. Codabench 번들(competition.yaml, 채점 프로그램 지정)은 공식 예제 번들을 복제해 만들고 `scoring.py`로 교체합니다. 일일 제출 횟수와 무료 한도는 공식 문서에서 확인하세요 [확인 필요].
   리더보드 열 키: 트랙 A `mIoU, pixAcc, missing` / 트랙 B `mIoU, adverse_mIoU, mIoU_cloud, mIoU_fog, mIoU_night, mIoU_rain, mIoU_sun, pixAcc, missing`.
5. 기준 점수 두 개(RGB 단독, 융합)를 직접 측정해 공지합니다.

## 다운로드 실패 시
- FMB: Google Drive 폴더 링크가 쿼터 초과일 수 있습니다. 폴더를 내 Drive에 바로가기로 추가하고 Colab에서 Drive를 마운트한 뒤 `prepare_fmb.py --raw <마운트 경로>`로 실행하세요.
- DELIVER: Hugging Face에서 `DELIVER.tar.gz` 하나로 제공됩니다(부분 다운로드 불가). 디스크가 부족하면 노트북 런타임을 다시 시작해 캐시를 비우세요.

## 출처 (확인일 2026-11-01)
- FMB 배포: https://github.com/JinyuanLiu-CV/SegMiF
- DELIVER 데이터: https://huggingface.co/datasets/InSAI-Lab/DELIVER , 논문 https://arxiv.org/abs/2303.01480
- SegFormer 구현과 가중치: Hugging Face `transformers`, `nvidia/mit-b0`
