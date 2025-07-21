# Weight Management 기능 사용법

AdamABS 실험에서 훈련된 모델 가중치를 저장하고 재사용할 수 있는 기능입니다.

## 주요 기능

### 1. 자동 체크포인트 저장
- **최고 성능 달성시 자동 저장**: 검증 정확도가 향상될 때마다 자동으로 체크포인트 저장
- **주기적 저장**: 설정한 에포크마다 정기적으로 체크포인트 저장 (기본: 5에포크마다)
- **완전한 상태 보존**: 모델 가중치, 옵티마이저 상태, 스케줄러 상태, 훈련 히스토리 모두 저장

### 2. 훈련 재개 기능
- **자동 재개**: 기존 최고 성능 체크포인트에서 자동으로 훈련 재개
- **상태 복원**: 모델, 옵티마이저, 스케줄러의 정확한 상태 복원
- **히스토리 연속성**: 이전 훈련 기록을 이어서 그래프 생성

### 3. 체크포인트 관리
- **메타데이터 관리**: 각 체크포인트의 상세 정보 추적
- **검색 기능**: 조건에 맞는 체크포인트 찾기
- **정리 기능**: 오래된 체크포인트 자동 정리

## 사용 방법

### 명령줄에서 사용

#### 1. 체크포인트 목록 보기
```bash
python main_experiment.py --list-checkpoints
```

#### 2. 기존 체크포인트에서 훈련 재개
```bash
# MNIST에서 Adam 옵티마이저로 훈련 재개
python main_experiment.py --dataset 1 --optimizers Adam --resume

# CIFAR-10에서 AdamABS 옵티마이저로 훈련 재개  
python main_experiment.py --dataset 2 --optimizers AdamABS --resume

# 전체 옵티마이저 비교 (기존 체크포인트 활용)
python main_experiment.py --dataset 1 --resume
```

#### 3. 새로운 훈련 시작 (체크포인트 저장 활성화)
```bash
# 일반적인 실험 실행 (자동으로 체크포인트 저장됨)
python main_experiment.py --dataset 1 --epochs 20

# AdamABS만 테스트
python main_experiment.py --dataset 2 --optimizers AdamABS --epochs 50
```

### 대화형 모드에서 사용

```bash
python main_experiment.py
```

대화형 모드에서는:
1. 체크포인트 목록을 먼저 확인할 수 있음
2. 실험 진행시 자동으로 재개 옵션을 물어봄
3. 모든 설정을 단계별로 선택 가능

## 파일 구조

```
./weights/                          # 체크포인트 저장 디렉토리
├── checkpoints_metadata.json       # 체크포인트 메타데이터
└── [모델]_[데이터셋]_[옵티마이저]_ep[에포크]_[타임스탬프].pth
```

### 체크포인트 파일 예시
```
Sequential_MNIST_Adam_ep010_20250721_143022.pth
Sequential_MNIST_AdamABS_ep015_20250721_145533.pth
Sequential_CIFAR-10_AdamW_ep025_20250721_152044.pth
```

## 저장되는 정보

각 체크포인트에는 다음 정보가 저장됩니다:

### 핵심 훈련 상태
- **모델 가중치**: `model.state_dict()`
- **옵티마이저 상태**: `optimizer.state_dict()` 
- **스케줄러 상태**: `scheduler.state_dict()` (있는 경우)

### 훈련 기록
- **훈련 히스토리**: loss, accuracy, epoch times 등
- **최고 성능**: 지금까지의 최고 검증 정확도
- **에포크 정보**: 현재 에포크, 총 훈련 시간

### 메타데이터
- **모델 정보**: 클래스명, 파라미터 수
- **옵티마이저 정보**: 클래스명, 하이퍼파라미터
- **실험 정보**: 데이터셋명, 타임스탬프
- **파일 정보**: 해시값, 파일 크기

## 코드 예시

### 직접 WeightManager 사용하기

```python
from weight_manager import WeightManager, ContinuousTrainer
import torch

# WeightManager 초기화
wm = WeightManager("./my_weights")

# 체크포인트 저장
wm.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    model_name="MyModel",
    dataset_name="MNIST", 
    optimizer_name="AdamABS",
    epoch=10,
    best_val_acc=95.5,
    training_time=120.5,
    training_history=history
)

# 최고 성능 체크포인트 찾기
best_checkpoint = wm.get_best_checkpoint("MyModel", "MNIST")

# 연속 훈련
ct = ContinuousTrainer(wm)
result = ct.find_and_resume_best(model, optimizer, scheduler, "MyModel", "MNIST")
if result:
    start_epoch, training_history = result
    # start_epoch부터 훈련 재개
```

### Trainer와 함께 사용하기

```python
from trainer import Trainer
from weight_manager import WeightManager

# WeightManager와 함께 Trainer 생성
wm = WeightManager("./weights")
trainer = Trainer(model, weight_manager=wm, auto_save_epochs=5)

# 메타데이터 설정
trainer.set_metadata("MyModel", "MNIST", "AdamABS")

# 훈련 실행 (자동으로 체크포인트 저장됨)
results = trainer.train_model(
    train_loader, val_loader, test_loader,
    optimizer, criterion, epochs=20
)
```

## 고급 기능

### 1. 체크포인트 검색
```python
# 조건에 맞는 체크포인트 검색
checkpoints = wm.find_checkpoints(
    dataset_name="MNIST",
    optimizer_name="AdamABS", 
    min_accuracy=90.0
)

# 정확도 순으로 정렬된 결과
for checkpoint in checkpoints:
    print(f"{checkpoint.optimizer_name}: {checkpoint.best_val_acc:.2f}%")
```

### 2. 체크포인트 정리
```python
# 성능 기준 상위 5개만 유지
wm.cleanup_old_checkpoints(keep_best_n=5)

# 특정 체크포인트 삭제
wm.delete_checkpoint(checkpoint_index=3)
```

### 3. 메타데이터 확인
```python
# 상세한 체크포인트 목록
wm.list_checkpoints(detailed=True)

# 간단한 요약
wm.list_checkpoints(detailed=False)
```

## 실험 전략

### 장시간 훈련시 권장사항

1. **주기적 저장 활성화**: `auto_save_epochs=5` (기본값)
2. **재개 기능 활용**: 실험이 중단되어도 `--resume` 옵션으로 재개
3. **체크포인트 관리**: 정기적으로 오래된 체크포인트 정리

### AdamABS 성능 추적

1. **옵티마이저별 비교**: 각 옵티마이저의 최고 성능 체크포인트 보존
2. **데이터셋별 분석**: MNIST, CIFAR-10, Tiny ImageNet별 최적 설정 추적
3. **하이퍼파라미터 실험**: 다양한 learning rate, weight decay 조합 테스트

## 주의사항

1. **디스크 공간**: 체크포인트 파일이 클 수 있으므로 주기적 정리 필요
2. **호환성**: PyTorch 버전이 다르면 체크포인트 로드 실패 가능
3. **메타데이터**: 실험시 반드시 `set_metadata()` 호출하여 정확한 추적 보장

## 문제 해결

### 체크포인트 로드 실패시
1. PyTorch 버전 확인
2. 모델 아키텍처 일치 여부 확인  
3. 파일 무결성 확인 (해시값 비교)

### 자동 저장이 안될 때
1. `weight_manager` 파라미터 전달 확인
2. `set_metadata()` 호출 확인
3. 디스크 공간 확인

이 기능을 통해 AdamABS 실험의 효율성을 크게 향상시킬 수 있습니다!