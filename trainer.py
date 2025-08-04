"""
훈련 및 평가 로직 통합 모듈
모든 데이터셋에 대해 동일한 출력 규격 제공

Author: AI Research
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
from weight_manager import WeightManager, ContinuousTrainer


class Trainer:
    """통합 훈련 클래스"""
    
    def __init__(self, model: nn.Module, device: str = 'auto', 
                 print_every_n_batches: int = 100, 
                 weight_manager: Optional[WeightManager] = None):
        """
        Args:
            model: 훈련할 모델
            device: 사용할 디바이스 ('auto', 'cuda', 'cpu')
            print_every_n_batches: 배치마다 출력할 간격
            weight_manager: WeightManager 인스턴스 (최고 성능시 자동 저장)
        """
        self.model = model
        self.print_every_n_batches = print_every_n_batches
        self.weight_manager = weight_manager
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 훈련 상태 변수
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'lr_history': []
        }
        
        # 메타데이터 (체크포인트 저장용)
        self.model_name = None
        self.dataset_name = None
        self.optimizer_name = None
        
        print(f"✅ Trainer 초기화 완료")
        print(f"   디바이스: {self.device}")
        print(f"   모델: {self.model.__class__.__name__}")
        print(f"   파라미터 수: {self._count_parameters():,}")
        if self.weight_manager:
            print(f"   체크포인트: 최고 성능 달성시 자동 저장")
    
    def _count_parameters(self) -> int:
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        한 에포크 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            optimizer: 옵티마이저
            criterion: 손실 함수
            epoch: 현재 에포크 (0부터 시작)
            total_epochs: 총 에포크 수
        
        Returns:
            Dict: 훈련 결과 (loss, accuracy, time 등)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []
        
        epoch_start_time = time.time()
        
        # 에포크 시작 출력
        print(f"\n📚 Epoch {epoch+1:3d}/{total_epochs} - Training")
        print("-" * 60)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 데이터를 디바이스로 이동
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 배치별 출력 (긴 에포크의 경우)
            if batch_idx % self.print_every_n_batches == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                progress = 100.0 * batch_idx / len(train_loader)
                
                # GPU 메모리 사용량 (CUDA 사용 시)
                gpu_memory = ""
                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / 1024**2
                    gpu_memory = f", GPU: {gpu_mb:.0f}MB"
                
                print(f"   Batch {batch_idx:4d}/{len(train_loader)} ({progress:5.1f}%) | "
                      f"Loss: {current_loss:.4f} | Acc: {current_acc:6.2f}% | "
                      f"Time: {batch_time:.3f}s{gpu_memory}")
        
        # 에포크 결과 계산
        epoch_time = time.time() - epoch_start_time
        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        avg_batch_time = np.mean(batch_times)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time,
            'total_samples': total
        }
        
        # 에포크 완료 출력
        print(f"📈 Train Result: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={epoch_time:.1f}s")
        
        return results
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module, 
                phase: str = "Validation") -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            data_loader: 평가 데이터 로더
            criterion: 손실 함수
            phase: 평가 단계 이름 ("Validation", "Test" 등)
        
        Returns:
            Dict: 평가 결과
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"🔍 {phase}")
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        eval_time = time.time() - eval_start_time
        avg_loss = running_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'eval_time': eval_time,
            'total_samples': total
        }
        
        # 평가 결과 출력
        print(f"📊 {phase} Result: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={eval_time:.1f}s")
        
        return results
    
    def set_metadata(self, model_name: str, dataset_name: str, optimizer_name: str):
        """체크포인트 저장을 위한 메타데이터 설정"""
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.optimizer_name = optimizer_name
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: Optional[DataLoader], optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, epochs: int,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   early_stopping_patience: int = None,
                   start_epoch: int = 0) -> Dict[str, Any]:
        """
        전체 모델 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더 (Optional)
            optimizer: 옵티마이저
            criterion: 손실 함수
            epochs: 훈련 에포크 수
            scheduler: 학습률 스케줄러 (Optional)
            early_stopping_patience: 조기 종료 patience (Optional)
            start_epoch: 시작 에포크 (체크포인트 재개시 사용)
        
        Returns:
            Dict: 전체 훈련 결과
        """
        print("=" * 80)
        print("🚀 모델 훈련 시작")
        print("=" * 80)
        print(f"총 에포크: {epochs}")
        if start_epoch > 0:
            print(f"시작 에포크: {start_epoch + 1} (체크포인트에서 재개)")
        print(f"옵티마이저: {optimizer.__class__.__name__}")
        if scheduler:
            print(f"스케줄러: {scheduler.__class__.__name__}")
        print(f"손실 함수: {criterion.__class__.__name__}")
        if early_stopping_patience:
            print(f"조기 종료: {early_stopping_patience} 에포크")
        if self.weight_manager:
            print(f"체크포인트: 최고 성능 달성시 자동 저장")
        print("=" * 80)
        
        # 훈련 시작 시간
        training_start_time = time.time()
        
        # Early stopping 변수
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # 현재 학습률 출력
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n🎯 Epoch {epoch+1}/{epochs} (LR: {current_lr:.2e})")
            
            # 훈련
            train_results = self.train_epoch(
                train_loader, optimizer, criterion, epoch, epochs
            )
            
            # 검증
            val_results = self.evaluate(val_loader, criterion, "Validation")
            
            # 학습률 스케줄러 업데이트
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results['loss'])
                else:
                    scheduler.step()
            
            # 히스토리 업데이트
            self.training_history['train_loss'].append(train_results['loss'])
            self.training_history['train_acc'].append(train_results['accuracy'])
            self.training_history['val_loss'].append(val_results['loss'])
            self.training_history['val_acc'].append(val_results['accuracy'])
            self.training_history['epoch_times'].append(train_results['epoch_time'])
            self.training_history['lr_history'].append(current_lr)
            
            # 최고 성능 업데이트
            if val_results['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_results['accuracy']
                print(f"🏆 New Best Validation Accuracy: {self.best_val_acc:.2f}%")
                
                # 최고 성능 달성시에만 체크포인트 저장
                if self.weight_manager and all([self.model_name, self.dataset_name, self.optimizer_name]):
                    try:
                        saved_path = self.weight_manager.save_checkpoint(
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            model_name=self.model_name,
                            dataset_name=self.dataset_name,
                            optimizer_name=self.optimizer_name,
                            epoch=epoch,
                            best_val_acc=self.best_val_acc,
                            training_time=time.time() - training_start_time,
                            training_history=self.training_history,
                            additional_info={
                                'is_best': True,
                                'val_loss': val_results['loss'],
                                'train_acc': train_results['accuracy']
                            }
                        )
                        if saved_path:
                            print(f"💾 체크포인트 갱신됨")
                    except Exception as e:
                        print(f"⚠️ 체크포인트 저장 실패: {e}")
            
            # Early stopping 체크
            if early_stopping_patience:
                if val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"⏹️  Early stopping at epoch {epoch+1} (patience: {early_stopping_patience})")
                        break
            
            # 에포크 요약
            print(f"📋 Epoch {epoch+1} Summary: "
                  f"Train({train_results['accuracy']:.2f}%) | "
                  f"Val({val_results['accuracy']:.2f}%) | "
                  f"Best({self.best_val_acc:.2f}%) | "
                  f"Time({train_results['epoch_time']:.1f}s)")
        
        # 총 훈련 시간
        total_training_time = time.time() - training_start_time
        
        # 최종 테스트 (제공된 경우)
        test_results = None
        if test_loader:
            print("\n" + "="*60)
            test_results = self.evaluate(test_loader, criterion, "Final Test")
            print("="*60)
        
        # 훈련 완료 요약
        print("\n" + "="*80)
        print("🎉 훈련 완료!")
        print("="*80)
        print(f"총 훈련 시간: {total_training_time/3600:.2f}시간 ({total_training_time:.1f}초)")
        print(f"평균 에포크 시간: {total_training_time/len(self.training_history['epoch_times']):.1f}초")
        print(f"최고 검증 정확도: {self.best_val_acc:.2f}%")
        if test_results:
            print(f"최종 테스트 정확도: {test_results['accuracy']:.2f}%")
        print("="*80)
        
        # 결과 반환
        final_results = {
            'training_history': self.training_history,
            'best_val_acc': self.best_val_acc,
            'total_training_time': total_training_time,
            'avg_epoch_time': total_training_time / len(self.training_history['epoch_times']),
            'final_train_acc': self.training_history['train_acc'][-1],
            'final_val_acc': self.training_history['val_acc'][-1],
            'total_epochs_trained': len(self.training_history['train_loss'])
        }
        
        if test_results:
            final_results['test_results'] = test_results
        
        return final_results


class OptimizerExperiment:
    """옵티마이저 비교 실험 클래스"""
    
    def __init__(self, dataset_type: int, model_type: str = 'default', 
                 enable_checkpoints: bool = True, weights_dir: str = "./weights"):
        """
        Args:
            dataset_type: 1(MNIST), 2(CIFAR-10), 3(Fashion-MNIST)
            model_type: 모델 타입
            enable_checkpoints: 체크포인트 저장 활성화 여부
            weights_dir: 체크포인트 저장 디렉토리
        """
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.experiment_results = {}
        self.enable_checkpoints = enable_checkpoints
        
        # 데이터셋 이름 매핑
        self.dataset_names = {1: "MNIST", 2: "CIFAR-10", 3: "Fashion-MNIST"}
        
        # WeightManager 초기화
        self.weight_manager = WeightManager(weights_dir) if enable_checkpoints else None
        
        print(f"🧪 OptimizerExperiment 초기화")
        print(f"   데이터셋: {self.dataset_names[dataset_type]}")
        print(f"   모델: {model_type}")
        if enable_checkpoints:
            print(f"   체크포인트: 활성화 ({weights_dir})")
    
    def run_single_optimizer_experiment(self, optimizer_name: str, 
                                      optimizer: torch.optim.Optimizer,
                                      train_loader: DataLoader,
                                      val_loader: DataLoader,
                                      test_loader: Optional[DataLoader],
                                      model: nn.Module,
                                      epochs: int,
                                      scheduler=None,
                                      resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """단일 옵티마이저 실험 실행"""
        
        print("\n" + "="*100)
        print(f"🔬 {optimizer_name.upper()} 실험 시작")
        print("="*100)
        
        # Trainer 생성
        trainer = Trainer(model, device='auto', weight_manager=self.weight_manager)
        
        # 메타데이터 설정
        trainer.set_metadata(
            model_name=model.__class__.__name__,
            dataset_name=self.dataset_names[self.dataset_type],
            optimizer_name=optimizer_name
        )
        
        # 체크포인트에서 재개 여부 확인
        start_epoch = 0
        if resume_from_checkpoint and self.weight_manager:
            continuous_trainer = ContinuousTrainer(self.weight_manager)
            resume_result = continuous_trainer.find_and_resume_best(
                model, optimizer, scheduler,
                model.__class__.__name__,
                self.dataset_names[self.dataset_type],
                optimizer_name
            )
            if resume_result:
                start_epoch, trainer.training_history = resume_result
                trainer.best_val_acc = max(trainer.training_history.get('val_acc', [0]))
        
        # 손실 함수
        criterion = create_standard_criterion()
        
        # 훈련 실행
        early_stopping_patience = None
        
        results = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=early_stopping_patience,
            start_epoch=start_epoch
        )
        
        # 실험 결과에 옵티마이저 정보 추가
        results['optimizer_name'] = optimizer_name
        results['optimizer_config'] = {
            'name': optimizer.__class__.__name__,
            'lr': optimizer.param_groups[0]['lr'],
            'params': {k: v for k, v in optimizer.param_groups[0].items() 
                      if k not in ['params', 'lr']}
        }
        
        return results
    
    def run_comparison_experiment(self, optimizers_config: Dict[str, Dict],
                                train_loader: DataLoader,
                                val_loader: DataLoader, 
                                test_loader: Optional[DataLoader],
                                model_factory_fn,
                                epochs: int,
                                scheduler_factory_fn=None,
                                resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        여러 옵티마이저 비교 실험 실행
        
        Args:
            optimizers_config: 옵티마이저 설정 딕셔너리
            train_loader, val_loader, test_loader: 데이터 로더들
            model_factory_fn: 모델 생성 함수
            epochs: 훈련 에포크 수
            scheduler_factory_fn: 스케줄러 생성 함수 (Optional)
            resume_from_checkpoint: 체크포인트에서 재개할지 여부
        
        Returns:
            Dict: 모든 옵티마이저 실험 결과
        """
        
        print("\n" + "="*100)
        print(f"🔬 {self.dataset_names[self.dataset_type]} 옵티마이저 비교 실험")
        print("="*100)
        print(f"실험할 옵티마이저: {list(optimizers_config.keys())}")
        print(f"에포크 수: {epochs}")
        print("="*100)
        
        all_results = {}
        
        for opt_name, opt_config in optimizers_config.items():
            try:
                # 새로운 모델 생성 (공정한 비교를 위해)
                model = model_factory_fn()
                
                # 옵티마이저 생성
                optimizer = opt_config['optimizer_class'](
                    model.parameters(), **opt_config['params']
                )
                
                # 스케줄러 생성 (있는 경우)
                scheduler = None
                if scheduler_factory_fn:
                    scheduler = scheduler_factory_fn(optimizer)
                
                # 실험 실행
                results = self.run_single_optimizer_experiment(
                    optimizer_name=opt_name,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model=model,
                    epochs=epochs,
                    scheduler=scheduler,
                    resume_from_checkpoint=resume_from_checkpoint
                )
                
                all_results[opt_name] = results
                
                # 중간 결과 출력
                print(f"\n✅ {opt_name} 실험 완료!")
                print(f"   최고 검증 정확도: {results['best_val_acc']:.2f}%")
                if 'test_results' in results:
                    print(f"   최종 테스트 정확도: {results['test_results']['accuracy']:.2f}%")
                print(f"   훈련 시간: {results['total_training_time']:.1f}초")
                
                # GPU 메모리 정리
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ {opt_name} 실험 실패: {e}")
                continue
        
        # 실험 결과 저장
        self.experiment_results = all_results
        
        # 결과 요약 출력
        self._print_experiment_summary()
        
        return all_results
    
    def _print_experiment_summary(self):
        """실험 결과 요약 출력"""
        if not self.experiment_results:
            return
        
        print("\n" + "="*100)
        print("📊 실험 결과 요약")
        print("="*100)
        
        # 성능 비교
        print(f"{'Optimizer':<15} {'Best Val Acc':<12} {'Test Acc':<10} {'Train Time':<12} {'Avg Epoch':<10}")
        print("-" * 80)
        
        for opt_name, results in self.experiment_results.items():
            test_acc = results.get('test_results', {}).get('accuracy', 0.0)
            
            print(f"{opt_name:<15} {results['best_val_acc']:>10.2f}% "
                  f"{test_acc:>8.2f}% "
                  f"{results['total_training_time']:>10.1f}s "
                  f"{results['avg_epoch_time']:>8.1f}s")
        
        # 최고 성능 옵티마이저
        best_opt = max(self.experiment_results.keys(), 
                      key=lambda x: self.experiment_results[x]['best_val_acc'])
        
        print(f"\n🏆 최고 성능: {best_opt} ({self.experiment_results[best_opt]['best_val_acc']:.2f}%)")
        
        # 가장 빠른 옵티마이저
        fastest_opt = min(self.experiment_results.keys(),
                         key=lambda x: self.experiment_results[x]['total_training_time'])
        
        print(f"⚡ 가장 빠름: {fastest_opt} ({self.experiment_results[fastest_opt]['total_training_time']:.1f}초)")
        
        print("="*100)


def create_standard_criterion(label_smoothing: float = 0.0) -> nn.Module:
    """표준 손실 함수 생성"""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_standard_scheduler(optimizer: torch.optim.Optimizer, 
                            scheduler_type: str = 'cosine',
                            epochs: int = 100) -> torch.optim.lr_scheduler._LRScheduler:
    """
    표준 스케줄러 생성
    
    Args:
        optimizer: 옵티마이저
        scheduler_type: 스케줄러 타입 ('cosine', 'step', 'plateau')
        epochs: 총 에포크 수
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    else:
        raise ValueError(f"지원하지 않는 스케줄러: {scheduler_type}")


if __name__ == "__main__":
    # 간단한 테스트
    print("Trainer 모듈 테스트")
    print("="*60)
    
    # 더미 모델과 데이터로 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"테스트 디바이스: {device}")
    
    # 간단한 모델 생성
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )
    
    # 더미 데이터 생성
    dummy_data = torch.randn(100, 10)
    dummy_targets = torch.randint(0, 5, (100,))
    
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10, shuffle=True)
    
    # Trainer 테스트
    trainer = Trainer(model, device='auto', print_every_n_batches=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\n간단한 훈련 테스트 실행...")
    results = trainer.train_model(
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        test_loader=dummy_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=3
    )
    
    print(f"\n✅ 테스트 완료!")
    print(f"   최고 검증 정확도: {results['best_val_acc']:.2f}%")
    print(f"   총 훈련 시간: {results['total_training_time']:.2f}초")