# AdamAbs: Adam 최적화 알고리즘의 변형 실험

이 프로젝트는 Adam 최적화 알고리즘의 변형인 AdamAbs를 제안하고 실험하는 연구 코드입니다.

## 🎯 연구 목표

기존 Adam 알고리즘에서 제곱(`g²`) 대신 절댓값(`|g|`)을 사용하고, 제곱근(`√v`) 대신 직접 나눗셈(`/v`)을 사용하는 변형 알고리즘의 성능을 평가합니다.

## 🔬 알고리즘 비교

### 기존 Adam
```
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

### 제안하는 AdamAbs
```
v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|
θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)
```

## 📁 프로젝트 구조

```
2025_AI_Paper/
├── adam_abs_optimizer.py    # AdamAbs 최적화 알고리즘 구현
├── mnist_experiment.py      # MNIST 데이터셋 실험
├── run_experiments.py       # 전체 실험 실행 스크립트
├── README.md               # 프로젝트 설명
└── results/                # 실험 결과 저장 디렉토리
```

## 🚀 실행 방법

### 1. 전체 실험 실행
```bash
python run_experiments.py
```

### 2. 옵션별 실행
```bash
# 빠른 실험 (10 에포크)
python run_experiments.py --epochs 10

# 특정 실험만 실행
python run_experiments.py --skip_theory --skip_synthetic

# 배치 크기 조정
python run_experiments.py --batch_size 256
```

### 3. 개별 실험 실행
```bash
# MNIST 실험만 실행
python mnist_experiment.py

# 최적화 알고리즘 이론 비교
python adam_abs_optimizer.py
```

## 📊 실험 내용

### 1. 이론적 분석
- Adam과 AdamAbs의 수학적 차이점 분석
- 계산 복잡도 비교
- 수치적 안정성 분석

### 2. 합성 데이터 실험
- 회귀 문제에서의 성능 비교
- 분류 문제에서의 성능 비교
- 다양한 데이터 분포에서의 강건성 테스트

### 3. MNIST 실험
- CNN 모델에서의 성능 비교
- 완전연결 모델에서의 성능 비교
- 훈련 안정성 및 수렴 속도 분석

### 4. 하이퍼파라미터 민감도 분석
- 학습률 민감도 테스트
- 베타 파라미터 영향 분석
- 배치 크기 영향 분석

## 📈 주요 결과

### 성능 지표
- **정확도**: 테스트 데이터셋에서의 분류 정확도
- **수렴 속도**: 목표 성능에 도달하는 에포크 수
- **훈련 안정성**: 검증 손실의 변동성
- **계산 효율성**: 에포크당 훈련 시간

### 예상 이점
1. **계산 효율성**: 제곱근 연산 제거로 속도 향상
2. **수치 안정성**: 절댓값 사용으로 안정성 증가
3. **이상치 강건성**: 제곱 대신 절댓값으로 이상치 영향 감소
4. **메모리 효율성**: 동일한 메모리 사용량

## 🔧 요구사항

```
torch>=1.12.0
torchvision>=0.13.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## 📝 결과 분석

실험 결과는 다음과 같이 저장됩니다:

- `results/`: JSON 형태의 수치 결과
- `*.png`: 그래프 및 시각화 결과
- `paper_summary.json`: 논문 작성을 위한 요약

## 🎯 논문 기여도

1. **새로운 최적화 알고리즘 제안**: AdamAbs
2. **포괄적인 실험 평가**: 다양한 데이터셋과 모델에서 검증
3. **이론적 분석**: 수학적 근거와 특성 분석
4. **실용적 고려사항**: 계산 효율성과 구현 용이성

## 🔍 향후 연구 방향

1. **다른 데이터셋 확장**: CIFAR-10, ImageNet 등
2. **대규모 모델 적용**: Transformer, ResNet 등
3. **이론적 수렴성 증명**: 수학적 수렴성 보장
4. **적응적 하이퍼파라미터**: 자동 튜닝 메커니즘

## 📞 문의사항

실험 결과나 코드에 대한 문의사항이 있으시면 이슈를 등록해주세요.

---

**참고**: 이 연구는 실험적 성격이 강하며, 실제 프로덕션 환경에서의 사용 전에는 충분한 검증이 필요합니다.

1. CIFAR-10/100 + ResNet18

  - 32x32x3 이미지 (3,072차원)
  - 깊은 CNN 모델 (11M+ 파라미터)
  - 데이터 증강 적용

  2. 고차원 합성 데이터 + 깊은 MLP

  - 2,000차원 입력
  - 5층 깊은 MLP (2M+ 파라미터)
  - 20개 클래스 분류

  3. Vision Transformer

  - CIFAR-10에서 패치 기반 attention
  - 6층 Transformer (1M+ 파라미터)
  - Self-attention 메커니즘

  🚀 실행 방법:

  전체 실험 실행:

  cd /Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper
  python run_high_dimensional_experiments.py

  특정 실험만 실행:

  # CIFAR-10만 실행
  python run_high_dimensional_experiments.py --experiment cifar10

  # 빠른 테스트 (5-10 에포크)
  python run_high_dimensional_experiments.py --experiment quick

  # 에포크 수 조정
  python run_high_dimensional_experiments.py --epochs 100 --batch_size 64

  📊 분석 내용:

  - 성능 비교: 테스트 정확도, 검증 정확도
  - 수렴 속도: 에포크별 수렴 분석
  - 계산 효율성: 훈련 시간, 메모리 사용량
  - 안정성: Gradient norm, 파라미터 변화
  - 종합 비교: 모든 실험 결과 통합 분석

  이 코드로 고차원 데이터에서 AdamAbs의 성능을 체계적으로 검증할 수
  있습니다!

    cuda_benchmark.py

  - GPU 성능 벤치마크
  - 메모리 사용량 모니터링
  - FP32 vs FP16 속도 비교
  - 최적화 알고리즘별 효율성 측정

  🎯 실행 방법:

  메인 실험 (CUDA 최적화):

  cd /Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper
  python run_high_dimensional_experiments.py

  CUDA 벤치마크:

  python cuda_benchmark.py

  특정 실험 (GPU 메모리 부족 시):

  python run_high_dimensional_experiments.py --experiment quick
  --batch_size 32