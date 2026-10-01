# Model Discriminator Module

2025 한이음 ICT 멘토링 프로젝트

Prediction Error Tracking Method 기반 딥러닝 검증 모듈입니다. 기본 분류 모델의 오분류 샘플로 보조 모델 A′를 학습하고, 두 모델의 예측 일치 여부를 이용해 입력별 신뢰도를 추정합니다.

## 프로젝트 흐름

```text
Dataset
  -> Base Model A
  -> Misclassified samples
  -> Fine-tune Model A′
  -> Compare A and A′
  -> Confidence output
```

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| CNN_base_model.py | CIFAR-10 기반 기본 CNN 학습 및 기본 모델 저장 |
| extract_dt_mis.py | 기본 모델의 오분류 샘플 추출 |
| aprime_model.py | 오분류 데이터 기반 A′ 모델 미세 조정 |
| PETM_module.py | A와 A′의 예측 일치 여부를 이용한 신뢰도 분석 |
| Ensemble_model.py | A와 A′의 앙상블 예측 및 정확도 비교 |
| Dt_mis_inputs.pt / Dt_mis_labels.pt | 오분류 입력 데이터와 정답 라벨 |
| cnn_cifar10.pth / cnn_cifar10_aprime.pth | A와 A′ 모델 가중치 |

## 실행 순서

PyTorch와 torchvision을 설치한 뒤 CIFAR-10 데이터셋을 사용합니다.

```bash
python CNN_base_model.py
python extract_dt_mis.py
python aprime_model.py
python PETM_module.py
python Ensemble_model.py
```

프로젝트에서는 이 구조를 반도체 wafer 공정 이미지 데이터에도 적용할 수 있도록 데이터 전처리와 클래스 수를 확장하는 방향을 검토했습니다.

## 참고 주제

- Prediction Error Tracking 기반 개별 예측 신뢰도 추정
- Adversarial Training과 FGSM
- MC Dropout 기반 불확실성 추정