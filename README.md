# Model_Discriminator_module

2025 한이음 ICT 멘토링 프로젝트에서 Prediction Error Tracking Method(PETM) 기반 딥러닝 검증 모듈을 실험한 저장소입니다.

## 저장소 범위

기본 분류 모델이 오분류한 샘플을 추출하고, 해당 데이터로 A′ 모델을 학습한 뒤 두 모델의 예측 차이를 이용해 입력별 신뢰도를 확인하는 PETM 기본 구조를 포함합니다.

이 저장소에는 ACK 2025 논문에서 제안한 최종 LoRA-PETM 아키텍처의 전체 구현 코드와 최종 실험 코드는 포함되어 있지 않습니다. 최종 연구 결과는 `LoRA-PETM: Prediction Error Tracking을 위한 메모리 효율적 아키텍처` 논문을 참고해 주세요.

## 주요 코드

- `src/base_model.py`: CIFAR-100 기반 기본 모델 학습 및 오분류 데이터 추출
- `src/prediction_error_tracking_module.py`: 오분류 데이터 기반 A′ 모델 학습
- `wafer_map/petm_wafer_map.py`: WM-811K wafer map 데이터셋 적용 실험
- `c&w.py`, `mc_dropout(fgsm).py`: 신뢰도 및 강건성 관련 보조 실험

## 실행 흐름

```text
Dataset
  -> Base Model A
  -> Misclassified samples
  -> Fine-tune Model A′
  -> Compare A and A′
  -> Confidence estimation
```

## 실행 환경

PyTorch, torchvision, pandas, numpy, Pillow를 사용합니다. 데이터셋 경로와 Colab 경로는 실행 환경에 맞게 수정해야 합니다.

```bash
python src/base_model.py
python src/prediction_error_tracking_module.py
python wafer_map/petm_wafer_map.py
```

## 참고

- Prediction Error Tracking Method 기반 개별 예측 신뢰도 추정
- 오분류 데이터 기반 보조 모델 학습
- 데이터셋별 일반화 가능성 검토