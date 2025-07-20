"""
MNIST 데이터셋을 활용한 Adam vs AdamAbs 비교 실험
논문 작성을 위한 포괄적인 실험 및 분석
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import os
from datetime import datetime
import json

# 커스텀 최적화 알고리즘 import
from adam_abs_optimizer import AdamAbs, AdamAbsW, create_optimizer

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MNISTNet(nn.Module):
    """MNIST 분류를 위한 CNN 모델"""
    
    def __init__(self, dropout_rate=0.25):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        # 첫 번째 컨볼루션 블록
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        
        # 두 번째 컨볼루션 블록
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout1(x)
        
        # 세 번째 컨볼루션 블록
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # 완전연결층
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class SimpleMNISTNet(nn.Module):
    """간단한 MNIST 분류 모델"""
    
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class MNISTExperiment:
    """MNIST 실험 클래스"""
    
    def __init__(self, data_dir='./data', batch_size=128, device=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 데이터 로더 설정
        self.setup_data_loaders()
        
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def setup_data_loaders(self):
        """데이터 로더 설정"""
        # 데이터 변환 정의
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 데이터셋 로드
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )
        
        # 훈련/검증 데이터 분할
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
    
    def train_epoch(self, model, optimizer, criterion, epoch, verbose=False):
        """한 에포크 훈련"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        grad_norms = []
        param_norms = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient norm 계산
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            grad_norms.append(total_norm)
            
            # Parameter norm 계산
            param_norm = 0
            for p in model.parameters():
                param_norm += p.data.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            param_norms.append(param_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if verbose and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.6f}, Acc: {100*correct/total:.2f}%')
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        avg_grad_norm = np.mean(grad_norms)
        avg_param_norm = np.mean(param_norms)
        
        return avg_loss, accuracy, avg_grad_norm, avg_param_norm
    
    def evaluate(self, model, criterion, data_loader):
        """모델 평가"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = test_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def run_experiment(self, model_class, optimizer_configs, epochs=20, verbose=True):
        """실험 실행"""
        results = {}
        
        for opt_name, opt_config in optimizer_configs.items():
            print(f"\n{'='*60}")
            print(f"실험: {opt_name.upper()} 최적화 알고리즘")
            print(f"{'='*60}")
            
            # 모델 초기화
            model = model_class().to(self.device)
            
            # 최적화 알고리즘 설정
            optimizer = create_optimizer(opt_config['name'], model.parameters(), **opt_config['params'])
            criterion = nn.CrossEntropyLoss()
            
            # 훈련 히스토리 초기화
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'grad_norm': [],
                'param_norm': [],
                'time_per_epoch': []
            }
            
            # 훈련 시작
            start_time = time.time()
            best_val_acc = 0.0
            
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # 훈련
                train_loss, train_acc, grad_norm, param_norm = self.train_epoch(
                    model, optimizer, criterion, epoch, verbose=verbose
                )
                
                # 검증
                val_loss, val_acc = self.evaluate(model, criterion, self.val_loader)
                
                epoch_time = time.time() - epoch_start_time
                
                # 히스토리 업데이트
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['grad_norm'].append(grad_norm)
                history['param_norm'].append(param_norm)
                history['time_per_epoch'].append(epoch_time)
                
                # 최고 성능 추적
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if verbose:
                    print(f'Epoch {epoch+1:2d}/{epochs}: '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                          f'Time: {epoch_time:.2f}s')
            
            total_time = time.time() - start_time
            
            # 최종 테스트
            test_loss, test_acc = self.evaluate(model, criterion, self.test_loader)
            
            results[opt_name] = {
                'history': history,
                'best_val_acc': best_val_acc,
                'final_test_acc': test_acc,
                'total_time': total_time,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'config': opt_config
            }
            
            print(f"\n결과 요약:")
            print(f"최고 검증 정확도: {best_val_acc:.2f}%")
            print(f"최종 테스트 정확도: {test_acc:.2f}%")
            print(f"총 훈련 시간: {total_time:.2f}초")
        
        return results
    
    def plot_results(self, results, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 훈련 손실
        ax = axes[0, 0]
        for opt_name, result in results.items():
            ax.plot(result['history']['train_loss'], label=opt_name, linewidth=2)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 검증 손실
        ax = axes[0, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_loss'], label=opt_name, linewidth=2)
        ax.set_title('Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 훈련 정확도
        ax = axes[0, 2]
        for opt_name, result in results.items():
            ax.plot(result['history']['train_acc'], label=opt_name, linewidth=2)
        ax.set_title('Training Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 검증 정확도
        ax = axes[1, 0]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_acc'], label=opt_name, linewidth=2)
        ax.set_title('Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Gradient Norm
        ax = axes[1, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['grad_norm'], label=opt_name, linewidth=2)
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. 최종 성능 비교
        ax = axes[1, 2]
        opt_names = list(results.keys())
        test_accs = [results[name]['final_test_acc'] for name in opt_names]
        
        bars = ax.bar(opt_names, test_accs, color=plt.cm.Set3(np.linspace(0, 1, len(opt_names))))
        ax.set_title('Final Test Accuracy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(95, 100)
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, results):
        """결과 분석"""
        print("\n" + "="*80)
        print("MNIST 실험 결과 분석")
        print("="*80)
        
        analysis = {}
        
        for opt_name, result in results.items():
            analysis[opt_name] = {
                'convergence_epoch': self.find_convergence_epoch(result['history']['val_loss']),
                'stability': np.std(result['history']['val_acc'][-5:]),  # 마지막 5 에포크 안정성
                'efficiency': result['total_time'] / result['best_val_acc'],  # 시간당 성능
                'final_gradient_norm': result['history']['grad_norm'][-1]
            }
        
        # 성능 비교
        print("\n1. 최종 성능 비교:")
        for opt_name, result in results.items():
            print(f"   {opt_name:12s}: Test Acc = {result['final_test_acc']:6.2f}%, "
                  f"Val Acc = {result['best_val_acc']:6.2f}%")
        
        # 수렴 분석
        print("\n2. 수렴 분석:")
        for opt_name, analysis_data in analysis.items():
            print(f"   {opt_name:12s}: 수렴 에포크 = {analysis_data['convergence_epoch']:2d}, "
                  f"안정성 = {analysis_data['stability']:6.3f}")
        
        # 효율성 분석
        print("\n3. 효율성 분석:")
        for opt_name, result in results.items():
            time_per_epoch = np.mean(result['history']['time_per_epoch'])
            print(f"   {opt_name:12s}: 총 시간 = {result['total_time']:6.1f}초, "
                  f"에포크당 시간 = {time_per_epoch:5.2f}초")
        
        # 최고 성능자 찾기
        best_optimizer = max(results.keys(), key=lambda x: results[x]['final_test_acc'])
        print(f"\n4. 최고 성능: {best_optimizer} ({results[best_optimizer]['final_test_acc']:.2f}%)")
        
        return analysis
    
    def find_convergence_epoch(self, val_losses, patience=3, min_improvement=0.01):
        """수렴 에포크 찾기"""
        if len(val_losses) < patience:
            return len(val_losses)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch, loss in enumerate(val_losses):
            if loss < best_loss - min_improvement:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                return epoch - patience + 1
        
        return len(val_losses)
    
    def save_results(self, results, save_dir='./results'):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과를 JSON으로 저장 (히스토리 제외)
        results_summary = {}
        for opt_name, result in results.items():
            results_summary[opt_name] = {
                'best_val_acc': result['best_val_acc'],
                'final_test_acc': result['final_test_acc'],
                'total_time': result['total_time'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss'],
                'config': result['config']
            }
        
        with open(f'{save_dir}/results_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n결과가 {save_dir}/results_{timestamp}.json에 저장되었습니다.")
        
        return f'{save_dir}/results_{timestamp}.json'


def main():
    """메인 실험 함수"""
    print("MNIST 데이터셋을 활용한 Adam vs AdamAbs 비교 실험")
    print("="*80)
    
    # 실험 설정
    experiment = MNISTExperiment(batch_size=128)
    
    # 최적화 알고리즘 설정
    optimizer_configs = {
        'adam': {
            'name': 'adam',
            'params': {'lr': 0.001, 'weight_decay': 1e-4}
        },
        'adamw': {
            'name': 'adamw',
            'params': {'lr': 0.001, 'weight_decay': 1e-4}
        },
        'adamabs': {
            'name': 'adamabs',
            'params': {'lr': 0.001, 'weight_decay': 1e-4}
        },
        'adamabsw': {
            'name': 'adamabsw',
            'params': {'lr': 0.001, 'weight_decay': 1e-4}
        }
    }
    
    # CNN 모델 실험
    print("\n1. CNN 모델 실험")
    cnn_results = experiment.run_experiment(MNISTNet, optimizer_configs, epochs=20)
    
    # 간단한 모델 실험
    print("\n2. 간단한 FC 모델 실험")
    fc_results = experiment.run_experiment(SimpleMNISTNet, optimizer_configs, epochs=15)
    
    # 결과 분석
    print("\n" + "="*80)
    print("CNN 모델 결과 분석")
    experiment.analyze_results(cnn_results)
    
    print("\n" + "="*80)
    print("FC 모델 결과 분석")
    experiment.analyze_results(fc_results)
    
    # 시각화
    experiment.plot_results(cnn_results, 
                           save_path='/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/mnist_cnn_results.png')
    
    experiment.plot_results(fc_results, 
                           save_path='/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/mnist_fc_results.png')
    
    # 결과 저장
    experiment.save_results(cnn_results, '/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results')
    experiment.save_results(fc_results, '/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results')
    
    print("\n모든 실험이 완료되었습니다!")


if __name__ == "__main__":
    main()