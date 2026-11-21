# 분석 결과 (트랙 A, FMB)

최종 분석 파일 모음이다. 모든 수치는 val 220장 기준이며 test 라벨은 사용하지 않았다.
"B2 480"은 MiT-B2, 입력 480×640을 뜻한다(최종 모델: RGB + IR Sum). 클래스 IoU 단위는 %.

## 1. 전체 실험

| 파일 | 내용 |
|---|---|
| `run_summary.csv` | 10개 실험의 융합 방식, 백본, 입력 크기, epoch, seed, best val mIoU, best epoch(1부터), 기록된 epoch 수 |
| `learning_curves.png` | epoch별 val mIoU 곡선. 왼쪽: B0 실험(384×512), 오른쪽: B2 실험 |

## 2. IR 기여 (B2 480: RGB 단독 vs RGB + IR Sum)

| 파일 | 내용 |
|---|---|
| `class_iou_rgb_vs_sum_480.csv` | RGB 단독과 Sum의 클래스별 IoU, 클래스별 IR 기여 (Person +18.79로 최대) |
| `miou_by_brightness.csv` | val을 RGB 평균 밝기로 3등분한 그룹별 mIoU (RGB 단독 / Sum / Sum + flip)와 IR 기여 (어두움 +4.64, 중간 +2.80, 밝음 +0.89) |
| `class_iou_dark_rgb_vs_sum.csv` | 어두운 그룹만의 클래스별 IoU, RGB 단독 vs Sum (Person 42.32 → 65.82). Bicycle은 이 그룹에 없어서 NaN |
| `ir_gain_class_brightness.png` | 왼쪽: 클래스별 IR 기여. 오른쪽: 밝기 그룹별 mIoU, RGB 단독 vs RGB + IR |
| `dark_rgb_vs_sum.png` | IR 효과가 가장 컸던 어두운 val 장면 3장: RGB, IR, 정답, RGB 단독 예측, RGB + IR 예측 |

## 3. 최종 모델 (B2 Sum 480)

| 파일 | 내용 |
|---|---|
| `final_class_iou_b2_480.csv` | 최종 모델의 클래스별 IoU와 B2 Sum 384×512 대비 차이 (`480_minus_384`) |
| `final_top_confusions_b2_480.csv` | IoU가 가장 낮은 6개 클래스와, 각 클래스가 가장 많이 잘못 예측된 클래스 3개 |
| `final_confusion_matrix_b2_480.png` | 행 정규화 혼동행렬 (각 정답 클래스 픽셀 중 예측 비율 %) |
| `final_failure_cases_b2_480.csv` | 픽셀 정확도가 가장 낮은 val 4장과 각 이미지의 주된 오류 (예: Sidewalk → Road) |
| `final_failure_cases_b2_480.png` | 같은 4장의 RGB, IR, 정답, 예측 |

## 4. 기타

| 파일 | 내용 |
|---|---|
| `b2_class_iou.csv` | B0 RGB / Sum / Concat / Adaptive와 B2 Sum 384×512의 클래스별 IoU, B2 − B0 Sum 차이 |
| `val_overall_tta.csv` | B2 RGB 480, B2 Sum 480, B2 Sum 480 + flip TTA의 val mIoU (73.65 → 73.63, TTA 미사용) |

클래스 번호는 FMB 순서를 따른다: 0 Road, 1 Sidewalk, 2 Building, 3 TrafficLight, 4 TrafficSign, 5 Vegetation, 6 Sky, 7 Person, 8 Car, 9 Truck, 10 Bus, 11 Motorcycle, 12 Bicycle, 13 Pole.
