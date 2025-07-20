"""
CIFAR-10에서 Custom Adam, AdamW, AdamAbs 직접 구현 비교 실험
모든 최적화 알고리즘을 동일한 방식으로 구현하여 공정한 비교
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
from typing import Dict, List, Any, Optional


class CustomAdam(torch.optim.Optimizer):
    """
    Adam 최적화 알고리즘 직접 구현
    
    수식:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Adam 최적화 스텝 수행"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                # Weight decay 적용 (L2 regularization)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    # 1차 모멘텀 (gradient의 지수이동평균)
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # 2차 모멘텀 (gradient 제곱의 지수이동평균)
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 1차 모멘텀 업데이트: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2차 모멘텀 업데이트: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                # Bias correction 계산
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 편향 보정된 추정값들
                # m̂_t = m_t / (1 - β₁^t)
                bias_corrected_exp_avg = exp_avg / bias_correction1
                # v̂_t = v_t / (1 - β₂^t)
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # 분모 계산: √v̂_t + ε
                denominator = bias_corrected_exp_avg_sq.sqrt().add_(group['eps'])
                
                # 파라미터 업데이트: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-group['lr'])
        
        return loss


class CustomAdamW(torch.optim.Optimizer):
    """
    AdamW 최적화 알고리즘 직접 구현 (Decoupled Weight Decay)
    
    수식:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """AdamW 최적화 스텝 수행"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 1차 모멘텀 업데이트
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2차 모멘텀 업데이트
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # 분모 계산
                denominator = bias_corrected_exp_avg_sq.sqrt().add_(group['eps'])
                
                # AdamW: Gradient-based update
                gradient_update = bias_corrected_exp_avg / denominator
                
                # AdamW: Decoupled weight decay
                # θ_t = θ_{t-1} - α * (gradient_update + λ * θ_{t-1})
                p.data.add_(gradient_update, alpha=-group['lr'])
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


class CustomAdamAbs(torch.optim.Optimizer):
    """
    AdamAbs 최적화 알고리즘 직접 구현
    
    수식:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|    ← 절댓값 사용
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)     ← 제곱근 제거
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamAbs, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """AdamAbs 최적화 스텝 수행"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                # Weight decay 적용
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 1차 모멘텀 업데이트 (Adam과 동일)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2차 모멘텀 업데이트 (절댓값 사용): v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|
                exp_avg_sq.mul_(beta2).add_(grad.abs(), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # 분모 계산 (제곱근 없음): v̂_t + ε
                denominator = bias_corrected_exp_avg_sq.add_(group['eps'])
                
                # 파라미터 업데이트 (제곱근 없음): θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-group['lr'])
        
        return loss


class ResNet18(nn.Module):
    """CIFAR-10용 ResNet-18 모델"""
    
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
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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


class CustomOptimizerExperiment:
    """커스텀 최적화 알고리즘 실험 클래스"""
    
    def __init__(self, batch_size=128, data_dir='./data'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"실험 환경:")
        print(f"  Device: {self.device}")
        print(f"  Batch Size: {self.batch_size}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
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
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )
        
        # 훈련/검증 데이터 분할 (90:10)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        print(f"데이터 로더 설정 완료:")
        print(f"  훈련 데이터: {len(train_dataset):,}개")
        print(f"  검증 데이터: {len(val_dataset):,}개")
        print(f"  테스트 데이터: {len(test_dataset):,}개")
    
    def train_epoch(self, model, optimizer, criterion, epoch, verbose=False):
        """한 에포크 훈련"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 시간 및 성능 지표 수집
        batch_times = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if verbose and batch_idx % 100 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f'    Batch {batch_idx:3d}/{len(self.train_loader)}: '
                      f'Loss={loss.item():.4f}, Acc={100*correct/total:.2f}%, '
                      f'Time={batch_time:.3f}s, GPU={gpu_memory:.1f}GB')
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        avg_batch_time = np.mean(batch_times)
        
        return avg_loss, accuracy, avg_batch_time
    
    def evaluate(self, model, criterion, data_loader):
        """모델 평가"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = test_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def run_single_experiment(self, optimizer_name, optimizer_class, lr=0.001, weight_decay=5e-4, epochs=100):
        """단일 최적화 알고리즘 실험"""
        print(f"\n{'='*80}")
        print(f"{optimizer_name} 실험 시작")
        print(f"{'='*80}")
        
        # 모델 초기화 (매번 동일한 초기 가중치를 위해 시드 설정)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = ResNet18(num_classes=10).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"총 파라미터 수: {total_params:,}")
        
        # 최적화 알고리즘 및 스케줄러 설정
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 훈련 히스토리
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'batch_time': [],
            'epoch_time': []
        }
        
        # 훈련 시작
        start_time = time.time()
        best_val_acc = 0.0
        best_test_acc = 0.0
        
        print(f"훈련 시작... (에포크: {epochs}, 학습률: {lr}, 가중치 감소: {weight_decay})")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss, train_acc, avg_batch_time = self.train_epoch(
                model, optimizer, criterion, epoch, verbose=(epoch % 10 == 0)
            )
            
            # 검증
            val_loss, val_acc = self.evaluate(model, criterion, self.val_loader)
            
            # 스케줄러 업데이트
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            epoch_time = time.time() - epoch_start_time
            
            # 히스토리 업데이트
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(current_lr)
            history['batch_time'].append(avg_batch_time)
            history['epoch_time'].append(epoch_time)
            
            # 최고 성능 추적
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 최고 검증 성능일 때 테스트 성능도 측정
                test_loss, test_acc = self.evaluate(model, criterion, self.test_loader)
                best_test_acc = test_acc
            
            # 에포크 결과 출력
            print(f'  Epoch {epoch+1:3d}/{epochs}: '
                  f'Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
                  f'LR={current_lr:.2e}, Time={epoch_time:.1f}s')
        
        total_time = time.time() - start_time
        
        # 최종 테스트 (가장 마지막 모델로)
        final_test_loss, final_test_acc = self.evaluate(model, criterion, self.test_loader)
        
        results = {
            'optimizer_name': optimizer_name,
            'history': history,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'total_time': total_time,
            'avg_epoch_time': total_time / epochs,
            'total_params': total_params,
            'config': {
                'lr': lr,
                'weight_decay': weight_decay,
                'epochs': epochs
            }
        }
        
        print(f"\n{optimizer_name} 실험 완료:")
        print(f"  최고 검증 정확도: {best_val_acc:.2f}%")
        print(f"  최고 테스트 정확도: {best_test_acc:.2f}%")
        print(f"  최종 테스트 정확도: {final_test_acc:.2f}%")
        print(f"  총 훈련 시간: {total_time:.1f}초")
        print(f"  평균 에포크 시간: {total_time/epochs:.1f}초")
        
        # GPU 메모리 정리
        del model, optimizer
        torch.cuda.empty_cache()
        
        return results
    
    def run_comparison_experiment(self, epochs=100, lr=0.001, weight_decay=5e-4):
        """모든 최적화 알고리즘 비교 실험"""
        print("CIFAR-10에서 Custom Optimizer 비교 실험")
        print(f"에포크: {epochs}, 학습률: {lr}, 가중치 감소: {weight_decay}")
        print("="*80)
        
        # 최적화 알고리즘 정의
        optimizers = {
            'Custom_Adam': CustomAdam,
            'Custom_AdamW': CustomAdamW,
            'Custom_AdamAbs': CustomAdamAbs
        }
        
        results = {}
        
        for opt_name, opt_class in optimizers.items():
            result = self.run_single_experiment(
                opt_name, opt_class, lr=lr, weight_decay=weight_decay, epochs=epochs
            )
            results[opt_name] = result
        
        return results
    
    def analyze_results(self, results):
        """결과 분석"""
        print(f"\n{'='*80}")
        print("Custom Optimizer 실험 결과 분석")
        print(f"{'='*80}")
        
        # 1. 최종 성능 비교
        print("\n1. 최종 성능 비교:")
        print(f"{'Optimizer':<15} {'Best Test Acc':<12} {'Final Test Acc':<13} {'Best Val Acc':<12}")
        print("-" * 60)
        for opt_name, result in results.items():
            print(f"{opt_name:<15} {result['best_test_acc']:>10.2f}% "
                  f"{result['final_test_acc']:>11.2f}% "
                  f"{result['best_val_acc']:>10.2f}%")
        
        # 2. 훈련 시간 비교
        print("\n2. 훈련 시간 비교:")
        print(f"{'Optimizer':<15} {'Total Time':<12} {'Avg Epoch Time':<15} {'Avg Batch Time':<15}")
        print("-" * 70)
        for opt_name, result in results.items():
            avg_batch_time = np.mean(result['history']['batch_time']) * 1000  # ms로 변환
            print(f"{opt_name:<15} {result['total_time']:>9.1f}s "
                  f"{result['avg_epoch_time']:>12.1f}s "
                  f"{avg_batch_time:>12.1f}ms")
        
        # 3. 수렴 분석
        print("\n3. 수렴 분석:")
        for opt_name, result in results.items():
            val_acc_history = result['history']['val_acc']
            convergence_epoch = self.find_convergence_epoch(val_acc_history)
            final_stability = np.std(val_acc_history[-10:])  # 마지막 10 에포크 안정성
            
            print(f"  {opt_name}:")
            print(f"    수렴 에포크: {convergence_epoch}")
            print(f"    최종 안정성 (std): {final_stability:.3f}")
            print(f"    최고 성능 달성 에포크: {np.argmax(val_acc_history) + 1}")
        
        # 4. Adam vs AdamAbs 상세 비교
        if 'Custom_Adam' in results and 'Custom_AdamAbs' in results:
            adam_result = results['Custom_Adam']
            adamabs_result = results['Custom_AdamAbs']
            
            print("\n4. Custom_Adam vs Custom_AdamAbs 상세 비교:")
            
            # 성능 차이
            acc_diff = adamabs_result['best_test_acc'] - adam_result['best_test_acc']
            time_diff = adamabs_result['total_time'] - adam_result['total_time']
            time_pct = (time_diff / adam_result['total_time']) * 100
            
            print(f"  테스트 정확도 차이: {acc_diff:+.3f}% (AdamAbs - Adam)")
            print(f"  훈련 시간 차이: {time_diff:+.1f}초 ({time_pct:+.2f}%)")
            
            # 배치당 시간 비교
            adam_batch_time = np.mean(adam_result['history']['batch_time']) * 1000
            adamabs_batch_time = np.mean(adamabs_result['history']['batch_time']) * 1000
            batch_time_diff = adamabs_batch_time - adam_batch_time
            batch_time_pct = (batch_time_diff / adam_batch_time) * 100
            
            print(f"  배치당 시간 차이: {batch_time_diff:+.2f}ms ({batch_time_pct:+.2f}%)")
        
        # 5. 최고 성능자 찾기
        best_acc_optimizer = max(results.keys(), key=lambda x: results[x]['best_test_acc'])
        fastest_optimizer = min(results.keys(), key=lambda x: results[x]['total_time'])
        most_stable = min(results.keys(), key=lambda x: np.std(results[x]['history']['val_acc'][-10:]))
        
        print(f"\n5. 종합 평가:")
        print(f"  최고 정확도: {best_acc_optimizer} ({results[best_acc_optimizer]['best_test_acc']:.2f}%)")
        print(f"  최고 속도: {fastest_optimizer} ({results[fastest_optimizer]['total_time']:.1f}초)")
        print(f"  최고 안정성: {most_stable} (std: {np.std(results[most_stable]['history']['val_acc'][-10:]):.3f})")
        
        return results
    
    def find_convergence_epoch(self, val_acc_history, patience=5, min_improvement=0.1):
        """수렴 에포크 찾기"""
        if len(val_acc_history) < patience:
            return len(val_acc_history)
        
        best_acc = 0
        patience_counter = 0
        
        for epoch, acc in enumerate(val_acc_history):
            if acc > best_acc + min_improvement:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                return epoch - patience + 1
        
        return len(val_acc_history)
    
    def plot_results(self, results, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 색상 설정
        colors = {'Custom_Adam': '#1f77b4', 'Custom_AdamW': '#ff7f0e', 'Custom_AdamAbs': '#2ca02c'}
        
        # 1. 훈련 손실
        ax = axes[0, 0]
        for opt_name, result in results.items():
            ax.plot(result['history']['train_loss'], label=opt_name, 
                   color=colors.get(opt_name, 'black'), linewidth=2)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. 검증 정확도
        ax = axes[0, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_acc'], label=opt_name, 
                   color=colors.get(opt_name, 'black'), linewidth=2)
        ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 학습률 변화
        ax = axes[0, 2]
        for opt_name, result in results.items():
            ax.plot(result['history']['learning_rate'], label=opt_name, 
                   color=colors.get(opt_name, 'black'), linewidth=2)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. 최종 테스트 정확도
        ax = axes[1, 0]
        opt_names = list(results.keys())
        test_accs = [results[name]['best_test_acc'] for name in opt_names]
        colors_list = [colors.get(name, 'gray') for name in opt_names]
        
        bars = ax.bar(opt_names, test_accs, color=colors_list, alpha=0.8)
        ax.set_title('Best Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(min(test_accs) - 1, max(test_accs) + 1)
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # x축 레이블 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 5. 훈련 시간
        ax = axes[1, 1]
        times = [results[name]['total_time'] for name in opt_names]
        bars = ax.bar(opt_names, times, color=colors_list, alpha=0.8)
        ax.set_title('Training Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        # 막대 위에 값 표시
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                   f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. 배치당 처리 시간
        ax = axes[1, 2]
        batch_times = [np.mean(results[name]['history']['batch_time']) * 1000 for name in opt_names]
        bars = ax.bar(opt_names, batch_times, color=colors_list, alpha=0.8)
        ax.set_title('Average Batch Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (ms)')
        
        # 막대 위에 값 표시
        for bar, time_val in zip(bars, batch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(batch_times)*0.01,
                   f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n결과 그래프 저장: {save_path}")
        
        plt.show()
    
    def save_results(self, results, save_dir='./results'):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'custom_optimizer_results_{timestamp}.json')
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for opt_name, result in results.items():
            serializable_results[opt_name] = {
                'optimizer_name': result['optimizer_name'],
                'best_val_acc': result['best_val_acc'],
                'best_test_acc': result['best_test_acc'],
                'final_test_acc': result['final_test_acc'],
                'total_time': result['total_time'],
                'avg_epoch_time': result['avg_epoch_time'],
                'total_params': result['total_params'],
                'config': result['config'],
                'convergence_epoch': self.find_convergence_epoch(result['history']['val_acc']),
                'final_stability': float(np.std(result['history']['val_acc'][-10:]))
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n결과 저장: {results_file}")
        return results_file
    
    def compare_operations_benchmark(self, tensor_size=1000000, iterations=1000):
        """개별 연산 성능 벤치마크"""
        print(f"\n{'='*60}")
        print("개별 연산 성능 벤치마크")
        print(f"텐서 크기: {tensor_size:,}, 반복 횟수: {iterations:,}")
        print(f"{'='*60}")
        
        # 테스트용 텐서 생성
        grad = torch.randn(tensor_size, device=self.device)
        v = torch.randn(tensor_size, device=self.device)
        
        # 연산별 시간 측정
        operations = {
            'Square (g²)': lambda: grad.pow(2),
            'Absolute (|g|)': lambda: grad.abs(),
            'Sqrt + Division': lambda: grad / (torch.sqrt(v.abs() + 1e-8)),
            'Direct Division': lambda: grad / (v.abs() + 1e-8)
        }
        
        results = {}
        
        for op_name, operation in operations.items():
            # GPU 워밍업
            for _ in range(10):
                _ = operation()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                _ = operation()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations * 1000  # ms
            results[op_name] = avg_time
            
            print(f"  {op_name:<20}: {avg_time:.4f} ms")
        
        # 상대적 성능 비교
        print(f"\n상대적 성능 (Square 기준):")
        square_time = results['Square (g²)']
        for op_name, time_val in results.items():
            relative = time_val / square_time
            print(f"  {op_name:<20}: {relative:.2f}x")
        
        return results


def main():
    """메인 실행 함수"""
    print("CIFAR-10에서 Custom Adam, AdamW, AdamAbs 비교 실험")
    print("모든 최적화 알고리즘을 동일한 방식으로 구현하여 공정한 비교")
    print("="*80)
    
    # 실험 설정
    experiment = CustomOptimizerExperiment(batch_size=128)
    
    # 개별 연산 벤치마크 (선택사항)
    print("\n연산 벤치마크를 실행하시겠습니까? (y/n): ", end="")
    if input().lower() == 'y':
        experiment.compare_operations_benchmark()
    
    # 메인 실험 실행
    results = experiment.run_comparison_experiment(epochs=100, lr=0.001, weight_decay=5e-4)
    
    # 결과 분석
    experiment.analyze_results(results)
    
    # 시각화
    save_path = 'custom_optimizer_comparison.png'
    experiment.plot_results(results, save_path)
    
    # 결과 저장
    experiment.save_results(results)
    
    print("\n" + "="*80)
    print("실험 완료!")
    print("="*80)


if __name__ == "__main__":
    main()