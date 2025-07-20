"""
ImageNet에서 Custom Adam, AdamW, AdamAbs 비교 실험
대규모 데이터셋에서의 성능 및 효율성 분석
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')


class CustomAdam(torch.optim.Optimizer):
    """Adam 최적화 알고리즘 직접 구현"""
    
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
                
                # Weight decay
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
                
                # 모멘텀 업데이트
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # 파라미터 업데이트
                denominator = bias_corrected_exp_avg_sq.sqrt().add_(group['eps'])
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-group['lr'])
        
        return loss


class CustomAdamW(torch.optim.Optimizer):
    """AdamW 최적화 알고리즘 직접 구현"""
    
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
                
                # 모멘텀 업데이트
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Gradient 업데이트
                denominator = bias_corrected_exp_avg_sq.sqrt().add_(group['eps'])
                gradient_update = bias_corrected_exp_avg / denominator
                
                p.data.add_(gradient_update, alpha=-group['lr'])
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


class CustomAdamAbs(torch.optim.Optimizer):
    """AdamAbs 최적화 알고리즘 직접 구현"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, 
                 gradient_clip=None, warmup_steps=0):
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
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       gradient_clip=gradient_clip, warmup_steps=warmup_steps)
        super(CustomAdamAbs, self).__init__(params, defaults)
        self.global_step = 0
    
    def step(self, closure=None):
        """AdamAbs 최적화 스텝 수행"""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.global_step += 1
        
        for group in self.param_groups:
            # Warmup learning rate
            if self.global_step <= group['warmup_steps']:
                warmup_lr = group['lr'] * self.global_step / group['warmup_steps']
            else:
                warmup_lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                # Gradient clipping
                if group['gradient_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_([p], group['gradient_clip'])
                    grad = p.grad.data
                
                # Weight decay
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
                
                # 1차 모멘텀 업데이트
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2차 모멘텀 업데이트 (절댓값 사용)
                exp_avg_sq.mul_(beta2).add_(grad.abs(), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # 파라미터 업데이트 (제곱근 없음)
                denominator = bias_corrected_exp_avg_sq.add_(group['eps'])
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-warmup_lr)
        
        return loss


class ImageNetExperiment:
    """ImageNet 실험 클래스"""
    
    def __init__(self, data_dir, batch_size=256, num_workers=8, model_name='resnet50'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"ImageNet 실험 환경:")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model_name}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Num Workers: {self.num_workers}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 데이터 로더 설정
        self.setup_data_loaders()
    
    def setup_data_loaders(self):
        """ImageNet 데이터 로더 설정"""
        # ImageNet 표준 변환
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # 데이터셋 로드
        train_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split='train', transform=train_transform
        )
        
        val_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split='val', transform=val_transform
        )
        
        # 작은 subset으로 테스트 (전체 데이터는 너무 큼)
        # 실제 실험에서는 이 부분을 제거하고 전체 데이터 사용
        subset_size = min(100000, len(train_dataset))  # 10만개 샘플로 제한
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        
        print(f"데이터셋 크기:")
        print(f"  훈련 데이터: {len(train_subset):,}개 (원본: {len(train_dataset):,}개)")
        print(f"  검증 데이터: {len(val_dataset):,}개")
        
        # 데이터 로더 생성
        self.train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        print(f"  배치 수: 훈련 {len(self.train_loader)}, 검증 {len(self.val_loader)}")
    
    def create_model(self):
        """모델 생성"""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=None, num_classes=1000)
        elif self.model_name == 'resnet18':
            model = models.resnet18(weights=None, num_classes=1000)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None, num_classes=1000)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    def train_epoch(self, model, optimizer, criterion, epoch, total_epochs, verbose=True):
        """한 에포크 훈련"""
        model.train()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        batch_times = []
        data_times = []
        
        end_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 데이터 로딩 시간 측정
            data_time = time.time() - end_time
            data_times.append(data_time)
            
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
            
            # Top-1, Top-5 정확도 계산
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            target_reshaped = target.view(-1, 1)
            correct_top1 += pred[:, :1].eq(target_reshaped).sum().item()
            correct_top5 += pred.eq(target_reshaped).sum().item()
            
            running_loss += loss.item()
            total += target.size(0)
            
            if verbose and batch_idx % 50 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                progress = 100. * batch_idx / len(self.train_loader)
                print(f'    Epoch {epoch+1:2d}/{total_epochs} [{batch_idx:4d}/{len(self.train_loader)} ({progress:3.0f}%)] '
                      f'Loss: {loss.item():.4f}, '
                      f'Top1: {100.*correct_top1/total:.2f}%, '
                      f'Top5: {100.*correct_top5/total:.2f}%, '
                      f'Time: {batch_time:.3f}s, '
                      f'GPU: {gpu_memory:.1f}GB')
            
            end_time = time.time()
        
        avg_loss = running_loss / len(self.train_loader)
        top1_acc = 100. * correct_top1 / total
        top5_acc = 100. * correct_top5 / total
        avg_batch_time = np.mean(batch_times)
        avg_data_time = np.mean(data_times)
        
        return avg_loss, top1_acc, top5_acc, avg_batch_time, avg_data_time
    
    def validate(self, model, criterion):
        """모델 검증"""
        model.eval()
        val_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = model(data)
                val_loss += criterion(output, target).item()
                
                # Top-1, Top-5 정확도 계산
                _, pred = output.topk(5, 1, largest=True, sorted=True)
                target_reshaped = target.view(-1, 1)
                correct_top1 += pred[:, :1].eq(target_reshaped).sum().item()
                correct_top5 += pred.eq(target_reshaped).sum().item()
                
                total += target.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        top1_acc = 100. * correct_top1 / total
        top5_acc = 100. * correct_top5 / total
        
        return avg_loss, top1_acc, top5_acc
    
    def run_single_experiment(self, optimizer_name, optimizer_class, 
                            lr=0.1, weight_decay=1e-4, epochs=90, warmup_epochs=5):
        """단일 최적화 알고리즘 실험"""
        print(f"\n{'='*100}")
        print(f"{optimizer_name} 실험 시작 - {self.model_name.upper()}")
        print(f"{'='*100}")
        
        # 모델 초기화
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = self.create_model()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"총 파라미터: {total_params:,}")
        print(f"훈련 파라미터: {trainable_params:,}")
        
        # 최적화 알고리즘 설정
        if optimizer_name == 'Custom_AdamAbs':
            # AdamAbs는 더 작은 학습률과 안정화 기법 사용
            warmup_steps = warmup_epochs * len(self.train_loader)
            optimizer = optimizer_class(model.parameters(), lr=lr*0.3, weight_decay=weight_decay,
                                      gradient_clip=1.0, warmup_steps=warmup_steps)
            print(f"AdamAbs 특별 설정: lr={lr*0.3}, gradient_clip=1.0, warmup_steps={warmup_steps}")
        else:
            optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        
        # 학습률 스케줄러 (ImageNet 표준)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # 훈련 히스토리
        history = {
            'train_loss': [], 'train_top1': [], 'train_top5': [],
            'val_loss': [], 'val_top1': [], 'val_top5': [],
            'learning_rate': [], 'batch_time': [], 'data_time': [], 'epoch_time': []
        }
        
        # 훈련 시작
        start_time = time.time()
        best_top1 = 0.0
        best_top5 = 0.0
        
        print(f"훈련 시작: {epochs} 에포크")
        print("-" * 100)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss, train_top1, train_top5, batch_time, data_time = self.train_epoch(
                model, optimizer, criterion, epoch, epochs, verbose=(epoch % 5 == 0)
            )
            
            # 검증 (매 5 에포크마다)
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_loss, val_top1, val_top5 = self.validate(model, criterion)
            else:
                val_loss = val_top1 = val_top5 = 0.0  # 시간 절약을 위해 생략
            
            # 스케줄러 업데이트
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            epoch_time = time.time() - epoch_start_time
            
            # 히스토리 업데이트
            history['train_loss'].append(train_loss)
            history['train_top1'].append(train_top1)
            history['train_top5'].append(train_top5)
            history['val_loss'].append(val_loss)
            history['val_top1'].append(val_top1)
            history['val_top5'].append(val_top5)
            history['learning_rate'].append(current_lr)
            history['batch_time'].append(batch_time)
            history['data_time'].append(data_time)
            history['epoch_time'].append(epoch_time)
            
            # 최고 성능 추적
            if val_top1 > best_top1:
                best_top1 = val_top1
                best_top5 = val_top5
            
            # 에포크 결과 출력
            print(f'Epoch {epoch+1:2d}/{epochs}: '
                  f'Train Loss={train_loss:.4f}, Train Top1={train_top1:.2f}%, '
                  f'Val Top1={val_top1:.2f}% (Best: {best_top1:.2f}%), '
                  f'LR={current_lr:.2e}, Time={epoch_time:.1f}s')
        
        total_time = time.time() - start_time
        
        # 최종 검증
        final_val_loss, final_top1, final_top5 = self.validate(model, criterion)
        
        results = {
            'optimizer_name': optimizer_name,
            'model_name': self.model_name,
            'history': history,
            'best_top1': best_top1,
            'best_top5': best_top5,
            'final_top1': final_top1,
            'final_top5': final_top5,
            'total_time': total_time,
            'avg_epoch_time': total_time / epochs,
            'avg_batch_time': np.mean(history['batch_time']),
            'total_params': total_params,
            'config': {
                'lr': lr if optimizer_name != 'Custom_AdamAbs' else lr*0.3,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'batch_size': self.batch_size
            }
        }
        
        print(f"\n{optimizer_name} 실험 완료:")
        print(f"  최고 Top-1 정확도: {best_top1:.2f}%")
        print(f"  최고 Top-5 정확도: {best_top5:.2f}%")
        print(f"  최종 Top-1 정확도: {final_top1:.2f}%")
        print(f"  총 훈련 시간: {total_time/3600:.1f}시간")
        print(f"  평균 배치 시간: {np.mean(history['batch_time'])*1000:.1f}ms")
        
        # 메모리 정리
        del model, optimizer
        torch.cuda.empty_cache()
        
        return results
    
    def run_comparison_experiment(self, epochs=30, lr=0.1, weight_decay=1e-4):
        """모든 최적화 알고리즘 비교 실험 (짧은 버전)"""
        print("ImageNet Custom Optimizer 비교 실험")
        print(f"모델: {self.model_name}, 에포크: {epochs}, 학습률: {lr}, 가중치 감소: {weight_decay}")
        print("="*100)
        
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
            
            # 중간 결과 저장 (긴 실험이므로)
            self.save_intermediate_results(results, f'imagenet_intermediate_{opt_name.lower()}.json')
        
        return results
    
    def analyze_results(self, results):
        """결과 분석"""
        print(f"\n{'='*100}")
        print("ImageNet 실험 결과 분석")
        print(f"{'='*100}")
        
        # 성능 비교
        print("\n1. 최종 성능 비교:")
        print(f"{'Optimizer':<15} {'Best Top-1':<10} {'Best Top-5':<10} {'Final Top-1':<11} {'Final Top-5':<11}")
        print("-" * 70)
        for opt_name, result in results.items():
            print(f"{opt_name:<15} {result['best_top1']:>8.2f}% "
                  f"{result['best_top5']:>8.2f}% "
                  f"{result['final_top1']:>9.2f}% "
                  f"{result['final_top5']:>9.2f}%")
        
        # 시간 효율성
        print("\n2. 시간 효율성:")
        print(f"{'Optimizer':<15} {'Total Time':<12} {'Avg Epoch':<12} {'Avg Batch':<12}")
        print("-" * 60)
        for opt_name, result in results.items():
            total_hours = result['total_time'] / 3600
            epoch_minutes = result['avg_epoch_time'] / 60
            batch_ms = result['avg_batch_time'] * 1000
            print(f"{opt_name:<15} {total_hours:>9.1f}h "
                  f"{epoch_minutes:>9.1f}m "
                  f"{batch_ms:>9.1f}ms")
        
        # 효율성 지표
        print("\n3. 효율성 지표 (성능/시간):")
        for opt_name, result in results.items():
            efficiency = result['best_top1'] / (result['total_time'] / 3600)  # 정확도/시간
            print(f"  {opt_name}: {efficiency:.2f} (Top-1% per hour)")
        
        return results
    
    def plot_results(self, results, save_path=None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # 2. Top-1 정확도
        ax = axes[0, 1]
        for opt_name, result in results.items():
            # 검증 데이터가 있는 에포크만 플롯
            val_epochs = [i for i, val in enumerate(result['history']['val_top1']) if val > 0]
            val_accs = [val for val in result['history']['val_top1'] if val > 0]
            if val_accs:
                ax.plot(val_epochs, val_accs, label=opt_name, 
                       color=colors.get(opt_name, 'black'), linewidth=2, marker='o')
        ax.set_title('Validation Top-1 Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 학습률 스케줄
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
        
        # 4. 최고 Top-1 정확도
        ax = axes[1, 0]
        opt_names = list(results.keys())
        top1_accs = [results[name]['best_top1'] for name in opt_names]
        colors_list = [colors.get(name, 'gray') for name in opt_names]
        
        bars = ax.bar(opt_names, top1_accs, color=colors_list, alpha=0.8)
        ax.set_title('Best Top-1 Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, top1_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 5. 훈련 시간 (시간 단위)
        ax = axes[1, 1]
        times_hours = [results[name]['total_time'] / 3600 for name in opt_names]
        bars = ax.bar(opt_names, times_hours, color=colors_list, alpha=0.8)
        ax.set_title('Total Training Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (hours)')
        
        # 막대 위에 값 표시
        for bar, time_val in zip(bars, times_hours):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times_hours)*0.01,
                   f'{time_val:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. 배치 처리 시간
        ax = axes[1, 2]
        batch_times = [results[name]['avg_batch_time'] * 1000 for name in opt_names]
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
    
    def save_intermediate_results(self, results, filename):
        """중간 결과 저장"""
        save_dir = './imagenet_results'
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, filename)
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for opt_name, result in results.items():
            serializable_results[opt_name] = {
                'optimizer_name': result['optimizer_name'],
                'model_name': result['model_name'],
                'best_top1': result['best_top1'],
                'best_top5': result['best_top5'],
                'final_top1': result['final_top1'],
                'final_top5': result['final_top5'],
                'total_time': result['total_time'],
                'avg_epoch_time': result['avg_epoch_time'],
                'avg_batch_time': result['avg_batch_time'],
                'total_params': result['total_params'],
                'config': result['config']
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"중간 결과 저장: {filepath}")
    
    def save_results(self, results, save_dir='./imagenet_results'):
        """최종 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'imagenet_results_{timestamp}.json')
        
        # 상세 결과 저장 (히스토리 포함)
        detailed_results = {}
        for opt_name, result in results.items():
            detailed_results[opt_name] = {
                'optimizer_name': result['optimizer_name'],
                'model_name': result['model_name'],
                'best_top1': result['best_top1'],
                'best_top5': result['best_top5'],
                'final_top1': result['final_top1'],
                'final_top5': result['final_top5'],
                'total_time': result['total_time'],
                'avg_epoch_time': result['avg_epoch_time'],
                'avg_batch_time': result['avg_batch_time'],
                'total_params': result['total_params'],
                'config': result['config'],
                'history_summary': {
                    'final_train_loss': result['history']['train_loss'][-1] if result['history']['train_loss'] else 0,
                    'max_train_top1': max(result['history']['train_top1']) if result['history']['train_top1'] else 0,
                    'efficiency_score': result['best_top1'] / (result['total_time'] / 3600)  # 성능/시간
                }
            }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n최종 결과 저장: {results_file}")
        return results_file


def benchmark_operations(device='cuda', tensor_size=10000000, iterations=1000):
    """ImageNet 규모에서 연산 벤치마크"""
    print(f"\n{'='*80}")
    print("ImageNet 규모 연산 벤치마크")
    print(f"텐서 크기: {tensor_size:,}, 반복 횟수: {iterations:,}")
    print(f"{'='*80}")
    
    # 대규모 텐서로 테스트 (ImageNet ResNet50 파라미터 수준)
    grad = torch.randn(tensor_size, device=device)
    v = torch.randn(tensor_size, device=device)
    
    operations = {
        'Square (g²)': lambda: grad.pow(2),
        'Absolute (|g|)': lambda: grad.abs(),
        'Sqrt + Division': lambda: grad / (torch.sqrt(v.abs() + 1e-8)),
        'Direct Division': lambda: grad / (v.abs() + 1e-8),
        'Combined Adam': lambda: grad / (torch.sqrt(grad.pow(2) * 0.999 + grad.pow(2) * 0.001) + 1e-8),
        'Combined AdamAbs': lambda: grad / (grad.abs() * 0.999 + grad.abs() * 0.001 + 1e-8)
    }
    
    results = {}
    
    for op_name, operation in operations.items():
        # GPU 워밍업
        for _ in range(10):
            _ = operation()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(iterations):
            _ = operation()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        results[op_name] = avg_time
        
        print(f"  {op_name:<20}: {avg_time:.4f} ms")
    
    # 상대적 성능
    print(f"\n상대적 성능 (Square 기준):")
    square_time = results['Square (g²)']
    for op_name, time_val in results.items():
        relative = time_val / square_time
        print(f"  {op_name:<20}: {relative:.2f}x")
    
    # AdamAbs vs Adam 전체 연산 비교
    adam_time = results['Combined Adam']
    adamabs_time = results['Combined AdamAbs']
    speedup = adam_time / adamabs_time
    print(f"\nAdamAbs 전체 연산 속도: {speedup:.2f}x {'빠름' if speedup > 1 else '느림'}")
    
    return results


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='ImageNet Custom Optimizer Experiment')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='ImageNet 데이터 디렉토리 경로')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                       help='사용할 모델')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='배치 크기')
    parser.add_argument('--epochs', type=int, default=30,
                       help='훈련 에포크 수')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='학습률')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='가중치 감소')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='데이터 로더 워커 수')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='연산 벤치마크만 실행')
    
    args = parser.parse_args()
    
    print("ImageNet Custom Optimizer 비교 실험")
    print("="*100)
    print(f"설정:")
    print(f"  모델: {args.model}")
    print(f"  배치 크기: {args.batch_size}")
    print(f"  에포크: {args.epochs}")
    print(f"  학습률: {args.lr}")
    print(f"  가중치 감소: {args.weight_decay}")
    print("="*100)
    
    # 연산 벤치마크
    if torch.cuda.is_available():
        benchmark_operations(device='cuda')
    else:
        print("CUDA를 사용할 수 없어 연산 벤치마크를 건너뜁니다.")
    
    if args.benchmark_only:
        return
    
    # ImageNet 실험
    experiment = ImageNetExperiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model
    )
    
    # 실험 실행
    results = experiment.run_comparison_experiment(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 결과 분석
    experiment.analyze_results(results)
    
    # 시각화
    save_path = f'imagenet_{args.model}_comparison.png'
    experiment.plot_results(results, save_path)
    
    # 결과 저장
    experiment.save_results(results)
    
    print("\n" + "="*100)
    print("ImageNet 실험 완료!")
    print("="*100)


# 주피터 노트북이나 스크립트에서 직접 실행할 때
def run_quick_experiment(data_dir, model='resnet18', epochs=10):
    """빠른 실험 실행 (테스트용)"""
    print("빠른 ImageNet 실험 (테스트 모드)")
    
    experiment = ImageNetExperiment(
        data_dir=data_dir,
        batch_size=128,  # 작은 배치 크기
        num_workers=4,
        model_name=model
    )
    
    results = experiment.run_comparison_experiment(
        epochs=epochs,
        lr=0.01,  # 작은 학습률
        weight_decay=1e-4
    )
    
    experiment.analyze_results(results)
    experiment.plot_results(results, f'quick_imagenet_{model}.png')
    
    return results


if __name__ == "__main__":
    main()