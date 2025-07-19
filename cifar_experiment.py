"""
CIFAR-10 데이터셋에서 Adam, AdamW, AdamAbs 비교 실험
CUDA 환경 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
import json

# 커스텀 최적화 알고리즘 import
from adam_abs_optimizer import AdamAbs, create_optimizer


class ResNet18(nn.Module):
    """ResNet-18 모델"""
    
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class ResNetBlock(nn.Module):
    """ResNet 기본 블록"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARExperiment:
    """CIFAR-10 실험 클래스 (CUDA 최적화)"""
    
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        
        # CUDA 환경 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            print(f"CUDA 환경:")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Mixed precision 지원 확인
            self.use_amp = hasattr(torch.cuda, 'amp')
            print(f"  Mixed Precision: {'사용 가능' if self.use_amp else '사용 불가'}")
        else:
            print("CPU 모드로 실행됩니다.")
            self.use_amp = False
        
        # 데이터 로더 설정
        self.setup_data_loaders()
    
    def setup_data_loaders(self):
        """CIFAR-10 데이터 로더 설정"""
        # 데이터 변환 정의
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 데이터셋 로드
        full_train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # 훈련/검증 데이터 분할
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성 (CUDA 최적화)
        num_workers = 8 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, 
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        
        print(f"데이터 로더 설정 완료:")
        print(f"  훈련 데이터: {len(train_dataset)}개")
        print(f"  검증 데이터: {len(val_dataset)}개")
        print(f"  테스트 데이터: {len(test_dataset)}개")
    
    def train_epoch(self, model, optimizer, criterion, epoch):
        """한 에포크 훈련 (CUDA 최적화)"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Mixed precision 스케일러
        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision 사용
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%, '
                      f'GPU: {gpu_memory:.1f}GB')
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, model, criterion, data_loader):
        """모델 평가 (CUDA 최적화)"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = test_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def run_experiment(self, epochs=100, lr=0.001, weight_decay=5e-4):
        """Adam, AdamW, AdamAbs 비교 실험"""
        print(f"\nCIFAR-10 실험 시작")
        print(f"에포크: {epochs}, 학습률: {lr}, 가중치 감소: {weight_decay}")
        print("="*60)
        
        # 최적화 알고리즘 설정
        optimizers = {
            'Adam': 'adam',
            'AdamW': 'adamw',
            'AdamAbs': 'adamabs'
        }
        
        results = {}
        
        for opt_name, opt_type in optimizers.items():
            print(f"\n{opt_name} 실험 시작...")
            
            # 모델 초기화
            model = ResNet18(num_classes=10).to(self.device)
            
            # 파라미터 수 출력
            total_params = sum(p.numel() for p in model.parameters())
            print(f"총 파라미터 수: {total_params:,}")
            
            # 최적화 알고리즘 및 스케줄러 설정
            if opt_type == 'adamabs':
                # 간단한 AdamAbs 사용 (오류 수정됨)
                from optimized_adamabs import OptimizedAdamAbs
                optimizer = OptimizedAdamAbs(model.parameters(), lr=lr, weight_decay=weight_decay)
                print("  수정된 AdamAbs 사용")
            else:
                optimizer = create_optimizer(opt_type, model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # 훈련 히스토리
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'learning_rate': []
            }
            
            # 훈련 시작
            start_time = time.time()
            best_val_acc = 0.0
            
            for epoch in range(epochs):
                # 훈련
                train_loss, train_acc = self.train_epoch(model, optimizer, criterion, epoch)
                
                # 검증
                val_loss, val_acc = self.evaluate(model, criterion, self.val_loader)
                
                # 스케줄러 업데이트
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                
                # 히스토리 업데이트
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['learning_rate'].append(current_lr)
                
                # 최고 성능 추적
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # 진행 상황 출력 (매 에포크)
                print(f'  Epoch {epoch+1:3d}/{epochs}: '
                      f'Train {train_loss:.4f}/{train_acc:.2f}%, '
                      f'Val {val_loss:.4f}/{val_acc:.2f}%, '
                      f'LR {current_lr:.2e}')
            
            total_time = time.time() - start_time
            
            # 최종 테스트
            test_loss, test_acc = self.evaluate(model, criterion, self.test_loader)
            
            # 결과 저장
            results[opt_name] = {
                'history': history,
                'best_val_acc': best_val_acc,
                'final_test_acc': test_acc,
                'total_time': total_time,
                'total_params': total_params
            }
            
            print(f"  {opt_name} 결과:")
            print(f"    최고 검증 정확도: {best_val_acc:.2f}%")
            print(f"    최종 테스트 정확도: {test_acc:.2f}%")
            print(f"    총 훈련 시간: {total_time:.1f}초")
            
            # GPU 메모리 정리
            del model
            torch.cuda.empty_cache()
        
        return results
    
    def plot_results(self, results, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 훈련 손실
        ax = axes[0, 0]
        for opt_name, result in results.items():
            ax.plot(result['history']['train_loss'], label=opt_name, linewidth=2)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. 검증 정확도
        ax = axes[0, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_acc'], label=opt_name, linewidth=2)
        ax.set_title('Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 최종 테스트 정확도
        ax = axes[1, 0]
        opt_names = list(results.keys())
        test_accs = [results[name]['final_test_acc'] for name in opt_names]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(opt_names, test_accs, color=colors)
        ax.set_title('Final Test Accuracy')
        ax.set_ylabel('Accuracy (%)')
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. 훈련 시간
        ax = axes[1, 1]
        times = [results[name]['total_time'] for name in opt_names]
        bars = ax.bar(opt_names, times, color=colors)
        ax.set_title('Training Time')
        ax.set_ylabel('Time (seconds)')
        
        # 막대 위에 값 표시
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"결과 그래프 저장: {save_path}")
        
        plt.show()
    
    def analyze_results(self, results):
        """결과 분석"""
        print(f"\n{'='*60}")
        print("CIFAR-10 실험 결과 분석")
        print(f"{'='*60}")
        
        # 성능 비교
        print("\n1. 최종 성능 비교:")
        for opt_name, result in results.items():
            print(f"   {opt_name:8s}: 테스트 정확도 = {result['final_test_acc']:6.2f}%, "
                  f"검증 정확도 = {result['best_val_acc']:6.2f}%")
        
        # 훈련 시간 비교
        print("\n2. 훈련 시간 비교:")
        for opt_name, result in results.items():
            print(f"   {opt_name:8s}: {result['total_time']:6.1f}초")
        
        # 최고 성능자
        best_test_optimizer = max(results.keys(), key=lambda x: results[x]['final_test_acc'])
        fastest_optimizer = min(results.keys(), key=lambda x: results[x]['total_time'])
        
        print(f"\n3. 최고 성능:")
        print(f"   정확도: {best_test_optimizer} ({results[best_test_optimizer]['final_test_acc']:.2f}%)")
        print(f"   속도: {fastest_optimizer} ({results[fastest_optimizer]['total_time']:.1f}초)")
        
        # AdamAbs vs Adam 비교
        if 'Adam' in results and 'AdamAbs' in results:
            adam_acc = results['Adam']['final_test_acc']
            adamabs_acc = results['AdamAbs']['final_test_acc']
            acc_improvement = (adamabs_acc - adam_acc) / adam_acc * 100
            
            adam_time = results['Adam']['total_time']
            adamabs_time = results['AdamAbs']['total_time']
            time_difference = (adamabs_time - adam_time) / adam_time * 100
            
            print(f"\n4. AdamAbs vs Adam:")
            print(f"   정확도 차이: {acc_improvement:+.2f}%")
            print(f"   시간 차이: {time_difference:+.2f}%")
        
        return results
    
    def save_results(self, results, save_dir='./results'):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'cifar10_results_{timestamp}.json')
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for opt_name, result in results.items():
            serializable_results[opt_name] = {
                'best_val_acc': result['best_val_acc'],
                'final_test_acc': result['final_test_acc'],
                'total_time': result['total_time'],
                'total_params': result['total_params']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n결과 저장: {results_file}")
        return results_file


def main():
    """메인 실행 함수"""
    print("CIFAR-10에서 Adam, AdamW, AdamAbs 비교 실험")
    print("="*60)
    
    # 실험 설정
    experiment = CIFARExperiment(batch_size=256)
    
    # 실험 실행
    results = experiment.run_experiment(epochs=100, lr=0.001, weight_decay=5e-4)
    
    # 결과 분석
    experiment.analyze_results(results)
    
    # 시각화
    save_path = 'cifar10_results.png'
    experiment.plot_results(results, save_path)
    
    # 결과 저장
    experiment.save_results(results, 'results')
    
    print("\n실험 완료!")


if __name__ == "__main__":
    main()