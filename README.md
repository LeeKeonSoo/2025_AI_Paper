# Adam vs AdamABS 옵티마이저 비교 실험

새로운 AdamABS 옵티마이저와 기존 Adam, AdamW 옵티마이저를 비교하는 연구 프로젝트입니다.

## 🚀 AdamABS 알고리즘

기존 Adam에서 다음과 같은 혁신적인 변화를 적용한 새로운 옵티마이저:

- **절댓값 사용**: gradient 제곱(`g²`) 대신 절댓값(`|g|`) 사용
- **제곱근 제거**: 분모에서 제곱근 연산 제거
- **계산 효율성**: 더 빠른 연산과 수치 안정성 향상

### 수식 비교

**Adam:**
```
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**AdamABS:**
```
v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|  ← 절댓값
θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)   ← 제곱근 제거
```

## 📁 프로젝트 구조

```
2025_AI_Paper/
├── 📁 v1.0/                          # 기존 실험 코드 (레거시)
│   ├── adam_abs_optimizer.py          # 구 AdamABS 구현
│   ├── mnist_experiment.py            # MNIST 구 실험
│   ├── cifar_basic_experiment.py      # CIFAR-10 구 실험
│   ├── imagenet_basic_experiment.py   # ImageNet 구 실험
│   └── ...                           # 기타 구 버전 파일들
│
├── 📄 모듈화된 새 버전 (v2.0):
│   ├── optimizers.py                  # 통합 옵티마이저 모듈
│   ├── data_loaders.py               # 통합 데이터 로더 모듈
│   ├── models.py                     # 통합 모델 정의 모듈
│   ├── trainer.py                    # 통합 훈련/평가 모듈
│   ├── visualizer.py                 # 시각화 및 결과 저장 모듈
│   └── main_experiment.py            # 메인 실험 실행 파일
│
├── 📁 data/                          # 데이터셋
├── 📁 results/                       # 실험 결과 (그래프, JSON)
├── README.md
└── requirements.txt
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 필요한 패키지
- PyTorch
- torchvision  
- matplotlib
- seaborn
- numpy
- pandas
- Pillow (Tiny ImageNet용)

## 🎯 사용 방법

### 1. 대화형 모드 (추천)
```bash
python main_experiment.py
```
- 데이터셋과 실험 설정을 대화형으로 선택
- 초보자에게 가장 친화적

### 2. 명령행 옵션

#### 기본 실험
```bash
# MNIST 전체 옵티마이저 비교
python main_experiment.py --dataset 1

# CIFAR-10 전체 옵티마이저 비교  
python main_experiment.py --dataset 2

# Tiny ImageNet 전체 옵티마이저 비교
python main_experiment.py --dataset 3
```

#### 빠른 테스트
```bash
# MNIST 빠른 테스트 (3 에포크, Adam vs AdamABS)
python main_experiment.py --dataset 1 --quick

# CIFAR-10 빠른 테스트
python main_experiment.py --dataset 2 --quick
```

#### Adam vs AdamABS 집중 비교
```bash
python main_experiment.py --dataset 1 --adam-vs-adamabs
python main_experiment.py --dataset 2 --adam-vs-adamabs
```

#### 사용자 정의 설정
```bash
# 에포크 수와 학습률 지정
python main_experiment.py --dataset 1 --epochs 30 --lr 0.001

# 배치 크기 조정
python main_experiment.py --dataset 2 --batch-size 64

# 특정 옵티마이저만 테스트
python main_experiment.py --dataset 1 --optimizers Adam AdamABS

# 간단한 모델 사용
python main_experiment.py --dataset 1 --model simple
```

### 3. 데이터셋 선택

| 번호 | 데이터셋 | 설명 | 권장 사용 |
|------|----------|------|-----------|
| `1` | MNIST | 손글씨 숫자 (28x28) | 빠른 테스트, 개념 증명 |
| `2` | CIFAR-10 | 자연 이미지 (32x32) | 중간 규모 실험 |
| `3` | Tiny ImageNet | ImageNet 축소판 (64x64→224x224) | 대규모 실험 |

### 4. Tiny ImageNet 설정

Tiny ImageNet 사용 시 데이터 다운로드가 필요합니다:

1. [Tiny ImageNet 다운로드](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
2. `data/tiny-imagenet-200/` 폴더에 압축 해제
3. 폴더 구조: `data/tiny-imagenet-200/train/`, `data/tiny-imagenet-200/val/`

## 📊 실험 결과

실험 완료 후 `results/` 폴더에 다음 파일들이 생성됩니다:

### 자동 생성 파일들
- **훈련 곡선**: `*_training_curves.png`
- **성능 비교**: `*_performance.png`  
- **옵티마이저 분석**: `*_analysis.png`
- **종합 보고서**: `*_report.png`
- **실험 데이터**: `*_results.json`

### 결과 예시
```
results/
├── mnist_20250120_143022_training_curves.png
├── mnist_20250120_143022_performance.png
├── mnist_20250120_143022_analysis.png
├── mnist_20250120_143022_report.png
└── mnist_20250120_143022_results.json
```

## 🔬 지원하는 옵티마이저

| 옵티마이저 | 설명 | 특징 |
|------------|------|------|
| **Adam** | 표준 Adam 직접 구현 | 기준선 (baseline) |
| **AdamW** | 분리된 가중치 감소 | 정규화 개선 |
| **AdamABS** | 절댓값 + 제곱근 제거 | 🆕 새로운 아이디어 |

## 🎮 사용 예시

### 예시 1: 빠른 MNIST 테스트
```bash
python main_experiment.py --dataset 1 --quick
```
**결과**: 3분 내에 Adam vs AdamABS 비교 완료

### 예시 2: CIFAR-10 전체 비교
```bash
python main_experiment.py --dataset 2 --epochs 50
```
**결과**: 모든 옵티마이저 50 에포크 비교 (약 1-2시간)

### 예시 3: 사용자 정의 실험
```bash
python main_experiment.py --dataset 2 --epochs 30 --lr 0.0005 --batch-size 64 --optimizers Adam AdamABS
```

## 📈 성능 모니터링

실험 중 다음과 같은 정보가 실시간으로 출력됩니다:

```
📚 Epoch   1/20 - Training
------------------------------------------------------------
   Batch  100/391 ( 25.6%) | Loss: 0.4521 | Acc:  86.23% | Time: 0.120s, GPU: 1250MB
   Batch  200/391 ( 51.2%) | Loss: 0.3845 | Acc:  88.91% | Time: 0.118s, GPU: 1250MB
📈 Train Result: Loss=0.3654, Acc=89.45%, Time=47.2s

🔍 Validation
📊 Validation Result: Loss=0.2134, Acc=92.18%, Time=5.3s

🏆 New Best Validation Accuracy: 92.18%
📋 Epoch 1 Summary: Train(89.45%) | Val(92.18%) | Best(92.18%) | Time(47.2s)
```

## 🔧 고급 설정

### 개발자용 모듈 사용
```python
from optimizers import CustomAdamABS, create_optimizer
from data_loaders import get_dataset_loader
from models import create_model
from trainer import OptimizerExperiment
from visualizer import ExperimentVisualizer

# 사용자 정의 실험 구성
loader = get_dataset_loader(dataset_type=1, batch_size=128)
model = create_model(dataset_type=1, model_type='default')
optimizer = create_optimizer('adamabs', model.parameters(), lr=0.001)
```

## 📋 명령행 옵션 전체 목록

```bash
python main_experiment.py [OPTIONS]

옵션:
  --dataset {1,2,3}           데이터셋 선택 (필수)
  --epochs INT                훈련 에포크 수
  --lr FLOAT                  학습률
  --batch-size INT            배치 크기
  --model {default,simple}    모델 타입
  --optimizers LIST           테스트할 옵티마이저
  --quick                     빠른 테스트 모드
  --adam-vs-adamabs           Adam vs AdamABS만 비교
  -h, --help                  도움말 출력
```

## 🔍 문제 해결

### 자주 발생하는 문제

1. **CUDA 메모리 부족**
   ```bash
   python main_experiment.py --dataset 2 --batch-size 32
   ```

2. **Tiny ImageNet 데이터 없음**
   ```
   ❌ Tiny ImageNet 데이터를 찾을 수 없습니다
   📥 다운로드: http://cs231n.stanford.edu/tiny-imagenet-200.zip
   ```

3. **느린 훈련 속도**
   - GPU 사용 확인: `torch.cuda.is_available()`
   - 배치 크기 증가: `--batch-size 256`
   - 워커 수 조정 (data_loaders.py 수정)

## 🎉 예상 결과

### AdamABS의 예상 장점
- ⚡ **빠른 연산**: 제곱근 계산 제거
- 🔒 **수치 안정성**: 절댓값 사용으로 안정적
- 💪 **강건성**: 이상치 gradient에 덜 민감
- 📈 **수렴성**: 더 부드러운 수렴 패턴

### 벤치마크 목표
- MNIST: 99%+ 정확도
- CIFAR-10: 90%+ 정확도  
- Tiny ImageNet: 60%+ 정확도

## 📚 참고 문헌

1. **Adam**: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
2. **AdamW**: Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
3. **AdamABS**: 본 연구에서 제안하는 새로운 알고리즘

## 📧 문의사항

실험 관련 문의사항이나 버그 리포트는 이슈로 등록해 주세요.

---

**⚠️ 참고사항**: `v1.0/` 폴더의 파일들은 레거시 코드입니다. 새로운 실험은 루트 디렉토리의 모듈화된 코드를 사용하세요.