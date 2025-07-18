"""
고차원 데이터셋을 활용한 Adam, AdamW, AdamAbs 비교 실험
CIFAR-10/100, ImageNet-subset, 고차원 합성 데이터 등을 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import time
import os
from datetime import datetime
import json
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 커스텀 최적화 알고리즘 import
from adam_abs_optimizer import AdamAbs, create_optimizer

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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


class VisionTransformerBlock(nn.Module):
    """Vision Transformer 블록"""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(VisionTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class SimpleViT(nn.Module):
    """간단한 Vision Transformer"""
    def __init__(self, image_size=32, patch_size=4, num_classes=10, embed_dim=256, num_heads=8, num_layers=6, mlp_dim=512):
        super(SimpleViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use class token
        x = self.head(x)
        
        return x


class HighDimensionalExperiment:
    """고차원 데이터셋 실험 클래스 (CUDA 최적화)"""
    
    def __init__(self, data_dir='./data', device=None):
        self.data_dir = data_dir
        
        # CUDA 환경 최적화
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # CUDA 최적화 설정
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"CUDA Benchmark: {torch.backends.cudnn.benchmark}")
            
            # 메모리 사용량 최적화
            torch.cuda.empty_cache()
            
            # Mixed precision 지원 확인
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                print("Mixed precision (AMP) support: Available")
                self.use_amp = True
            else:
                print("Mixed precision (AMP) support: Not available")
                self.use_amp = False
        else:
            self.use_amp = False
    
    def create_high_dimensional_synthetic_data(self, n_samples=5000, n_features=1000, n_classes=10):
        """고차원 합성 데이터 생성"""
        print(f"고차원 합성 데이터 생성: {n_samples} samples, {n_features} features, {n_classes} classes")
        
        # 분류 데이터
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # 표준화
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 노이즈 추가
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def create_cifar_data_loaders(self, dataset='cifar10', batch_size=128, augmentation=True):
        """CIFAR 데이터 로더 생성"""
        print(f"CIFAR 데이터 로더 생성: {dataset.upper()}")
        
        # 데이터 증강
        if augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 데이터셋 선택
        if dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
            num_classes = 10
        else:
            dataset_class = torchvision.datasets.CIFAR100
            num_classes = 100
        
        # 데이터셋 로드
        train_dataset = dataset_class(root=self.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = dataset_class(root=self.data_dir, train=False, download=True, transform=transform_test)
        
        # 검증 데이터 분할
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # 데이터 로더 생성 (CUDA 최적화)
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=pin_memory, 
                                 persistent_workers=True if num_workers > 0 else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=pin_memory,
                               persistent_workers=True if num_workers > 0 else False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=True if num_workers > 0 else False)
        
        return train_loader, val_loader, test_loader, num_classes
    
    def create_high_dim_synthetic_loader(self, n_samples=5000, n_features=1000, n_classes=10, batch_size=128):
        """고차원 합성 데이터 로더 생성"""
        X, y = self.create_high_dimensional_synthetic_data(n_samples, n_features, n_classes)
        
        # 훈련/검증/테스트 분할
        total_size = len(X)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        dataset = TensorDataset(X, y)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # CUDA 최적화된 데이터 로더
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader, n_features, n_classes
    
    def create_deep_mlp(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        """깊은 MLP 모델 생성"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def train_epoch(self, model, optimizer, criterion, train_loader, epoch, verbose=False):
        """한 에포크 훈련 (CUDA 최적화 및 Mixed Precision)"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        grad_norms = []
        param_norms = []
        lr_effective_list = []
        
        # Mixed precision 스케일러
        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터를 GPU로 이동 (non_blocking=True로 최적화)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision 사용
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                # 역전파 (scaled)
                scaler.scale(loss).backward()
                
                # Gradient norm 계산
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                grad_norms.append(total_norm)
                
                # 효과적인 학습률 계산
                if hasattr(optimizer, 'get_lr_effective'):
                    lr_eff = optimizer.get_lr_effective()
                    lr_effective_list.extend(lr_eff)
                
                # 옵티마이저 스텝
                scaler.step(optimizer)
                scaler.update()
            else:
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
                
                # 효과적인 학습률 계산
                if hasattr(optimizer, 'get_lr_effective'):
                    lr_eff = optimizer.get_lr_effective()
                    lr_effective_list.extend(lr_eff)
                
                optimizer.step()
            
            # Parameter norm 계산
            param_norm = 0
            for p in model.parameters():
                param_norm += p.data.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            param_norms.append(param_norm)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if verbose and batch_idx % 100 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, Acc: {100*correct/total:.2f}%, '
                      f'GPU Mem: {gpu_memory:.2f}GB')
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        avg_param_norm = np.mean(param_norms) if param_norms else 0
        avg_lr_effective = np.mean(lr_effective_list) if lr_effective_list else 0
        
        return avg_loss, accuracy, avg_grad_norm, avg_param_norm, avg_lr_effective
    
    def evaluate(self, model, criterion, data_loader):
        """모델 평가 (CUDA 최적화)"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                # 데이터를 GPU로 이동 (non_blocking=True로 최적화)
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Mixed precision 사용
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
    
    def run_experiment(self, model_factory, train_loader, val_loader, test_loader, 
                      experiment_name, epochs=50, lr=0.001, weight_decay=1e-4):
        """실험 실행"""
        print(f"\n{'='*80}")
        print(f"실험: {experiment_name}")
        print(f"{'='*80}")
        
        # 최적화 알고리즘 설정
        optimizers = {
            'Adam': 'adam',
            'AdamW': 'adamw', 
            'AdamAbs': 'adamabs'
        }
        
        results = {}
        
        for opt_name, opt_type in optimizers.items():
            print(f"\n{'-'*60}")
            print(f"최적화 알고리즘: {opt_name}")
            print(f"{'-'*60}")
            
            # 모델 초기화 및 GPU 이동
            model = model_factory().to(self.device)
            
            # 모델 컴파일 (PyTorch 2.0+)
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    model = torch.compile(model, mode='max-autotune')
                    print("모델 컴파일 성공 (PyTorch 2.0+)")
                except Exception as e:
                    print(f"모델 컴파일 실패: {e}")
            
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"총 파라미터 수: {total_params:,}")
            print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
            
            # 최적화 알고리즘 설정
            optimizer = create_optimizer(opt_type, model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # 스케줄러 설정
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
            
            # GPU 메모리 사용량 체크
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # 훈련 히스토리 초기화
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'grad_norm': [],
                'param_norm': [],
                'lr_effective': [],
                'learning_rate': [],
                'time_per_epoch': []
            }
            
            # 훈련 시작
            start_time = time.time()
            best_val_acc = 0.0
            patience_counter = 0
            patience = 10
            
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # 훈련
                train_loss, train_acc, grad_norm, param_norm, lr_effective = self.train_epoch(
                    model, optimizer, criterion, train_loader, epoch, verbose=(epoch % 10 == 0)
                )
                
                # 검증
                val_loss, val_acc = self.evaluate(model, criterion, val_loader)
                
                # 스케줄러 업데이트
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                
                epoch_time = time.time() - epoch_start_time
                
                # 히스토리 업데이트
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['grad_norm'].append(grad_norm)
                history['param_norm'].append(param_norm)
                history['lr_effective'].append(lr_effective)
                history['learning_rate'].append(current_lr)
                history['time_per_epoch'].append(epoch_time)
                
                # 최고 성능 추적
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if epoch % 5 == 0:
                    print(f'Epoch {epoch+1:3d}/{epochs}: '
                          f'Train: {train_loss:.4f}/{train_acc:.2f}%, '
                          f'Val: {val_loss:.4f}/{val_acc:.2f}%, '
                          f'LR: {current_lr:.2e}, '
                          f'Time: {epoch_time:.2f}s')
            
            total_time = time.time() - start_time
            
            # 최종 테스트
            test_loss, test_acc = self.evaluate(model, criterion, test_loader)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results[opt_name] = {
                'history': history,
                'best_val_acc': best_val_acc,
                'final_test_acc': test_acc,
                'total_time': total_time,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'epochs_trained': len(history['train_loss'])
            }
            
            print(f"\n{opt_name} 결과:")
            print(f"  최고 검증 정확도: {best_val_acc:.2f}%")
            print(f"  최종 테스트 정확도: {test_acc:.2f}%")
            print(f"  총 훈련 시간: {total_time:.2f}초")
            print(f"  훈련된 에포크: {len(history['train_loss'])}")
        
        return results
    
    def plot_comprehensive_results(self, results, experiment_name, save_path=None):
        """포괄적인 결과 시각화"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
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
        
        # 2. 검증 손실
        ax = axes[0, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_loss'], label=opt_name, linewidth=2)
        ax.set_title('Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. 검증 정확도
        ax = axes[0, 2]
        for opt_name, result in results.items():
            ax.plot(result['history']['val_acc'], label=opt_name, linewidth=2)
        ax.set_title('Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Gradient Norm
        ax = axes[1, 0]
        for opt_name, result in results.items():
            ax.plot(result['history']['grad_norm'], label=opt_name, linewidth=2)
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 5. Parameter Norm
        ax = axes[1, 1]
        for opt_name, result in results.items():
            ax.plot(result['history']['param_norm'], label=opt_name, linewidth=2)
        ax.set_title('Parameter Norm')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Learning Rate
        ax = axes[1, 2]
        for opt_name, result in results.items():
            ax.plot(result['history']['learning_rate'], label=opt_name, linewidth=2)
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 7. 최종 테스트 정확도
        ax = axes[2, 0]
        opt_names = list(results.keys())
        test_accs = [results[name]['final_test_acc'] for name in opt_names]
        bars = ax.bar(opt_names, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title('Final Test Accuracy')
        ax.set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 8. 훈련 시간
        ax = axes[2, 1]
        times = [results[name]['total_time'] for name in opt_names]
        bars = ax.bar(opt_names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title('Training Time')
        ax.set_ylabel('Time (seconds)')
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 9. 수렴 비교 (최고 검증 정확도)
        ax = axes[2, 2]
        best_val_accs = [results[name]['best_val_acc'] for name in opt_names]
        bars = ax.bar(opt_names, best_val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title('Best Validation Accuracy')
        ax.set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, best_val_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'{experiment_name} - Comprehensive Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, results, experiment_name):
        """결과 분석"""
        print(f"\n{'='*80}")
        print(f"{experiment_name} 결과 분석")
        print(f"{'='*80}")
        
        # 성능 비교
        print("\n1. 최종 성능 비교:")
        for opt_name, result in results.items():
            print(f"   {opt_name:8s}: Test Acc = {result['final_test_acc']:6.2f}%, "
                  f"Best Val Acc = {result['best_val_acc']:6.2f}%, "
                  f"Time = {result['total_time']:6.1f}s")
        
        # 수렴 분석
        print("\n2. 수렴 분석:")
        for opt_name, result in results.items():
            epochs_trained = result['epochs_trained']
            convergence_rate = result['best_val_acc'] / epochs_trained
            print(f"   {opt_name:8s}: 에포크 = {epochs_trained:3d}, "
                  f"수렴률 = {convergence_rate:.3f}%/epoch")
        
        # 효율성 분석
        print("\n3. 효율성 분석:")
        for opt_name, result in results.items():
            time_per_epoch = result['total_time'] / result['epochs_trained']
            efficiency = result['best_val_acc'] / result['total_time']
            print(f"   {opt_name:8s}: 에포크당 시간 = {time_per_epoch:5.2f}s, "
                  f"효율성 = {efficiency:.3f}%/s")
        
        # 최고 성능자
        best_test_optimizer = max(results.keys(), key=lambda x: results[x]['final_test_acc'])
        best_val_optimizer = max(results.keys(), key=lambda x: results[x]['best_val_acc'])
        fastest_optimizer = min(results.keys(), key=lambda x: results[x]['total_time'])
        
        print(f"\n4. 최고 성능:")
        print(f"   테스트 정확도: {best_test_optimizer} ({results[best_test_optimizer]['final_test_acc']:.2f}%)")
        print(f"   검증 정확도: {best_val_optimizer} ({results[best_val_optimizer]['best_val_acc']:.2f}%)")
        print(f"   훈련 속도: {fastest_optimizer} ({results[fastest_optimizer]['total_time']:.1f}s)")
        
        # 상대적 성능 개선
        print(f"\n5. AdamAbs vs Adam 비교:")
        if 'Adam' in results and 'AdamAbs' in results:
            adam_test = results['Adam']['final_test_acc']
            adamabs_test = results['AdamAbs']['final_test_acc']
            test_improvement = (adamabs_test - adam_test) / adam_test * 100
            
            adam_time = results['Adam']['total_time']
            adamabs_time = results['AdamAbs']['total_time']
            time_difference = (adamabs_time - adam_time) / adam_time * 100
            
            print(f"   테스트 정확도 개선: {test_improvement:+.2f}%")
            print(f"   훈련 시간 차이: {time_difference:+.2f}%")
        
        return results


def main():
    """메인 실행 함수"""
    print("고차원 데이터셋을 활용한 Adam, AdamW, AdamAbs 비교 실험")
    print("="*80)
    
    experiment = HighDimensionalExperiment()
    
    # 실험 1: CIFAR-10 + ResNet18
    print("\n실험 1: CIFAR-10 + ResNet18")
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders('cifar10', batch_size=128)
    
    def resnet18_factory():
        return ResNet18(num_classes=num_classes)
    
    cifar10_results = experiment.run_experiment(
        resnet18_factory, train_loader, val_loader, test_loader,
        "CIFAR-10 + ResNet18", epochs=100, lr=0.001, weight_decay=5e-4
    )
    
    # 실험 2: CIFAR-100 + ResNet18
    print("\n실험 2: CIFAR-100 + ResNet18")
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders('cifar100', batch_size=128)
    
    def resnet18_cifar100_factory():
        return ResNet18(num_classes=num_classes)
    
    cifar100_results = experiment.run_experiment(
        resnet18_cifar100_factory, train_loader, val_loader, test_loader,
        "CIFAR-100 + ResNet18", epochs=100, lr=0.001, weight_decay=5e-4
    )
    
    # 실험 3: 고차원 합성 데이터 + 깊은 MLP
    print("\n실험 3: 고차원 합성 데이터 + 깊은 MLP")
    train_loader, val_loader, test_loader, n_features, n_classes = experiment.create_high_dim_synthetic_loader(
        n_samples=10000, n_features=2000, n_classes=20, batch_size=128
    )
    
    def deep_mlp_factory():
        return experiment.create_deep_mlp(n_features, [1024, 512, 256, 128], n_classes, dropout=0.3)
    
    synthetic_results = experiment.run_experiment(
        deep_mlp_factory, train_loader, val_loader, test_loader,
        "고차원 합성 데이터 + 깊은 MLP", epochs=50, lr=0.001, weight_decay=1e-4
    )
    
    # 실험 4: CIFAR-10 + Vision Transformer
    print("\n실험 4: CIFAR-10 + Vision Transformer")
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders('cifar10', batch_size=64)
    
    def vit_factory():
        return SimpleViT(image_size=32, patch_size=4, num_classes=num_classes, 
                        embed_dim=256, num_heads=8, num_layers=6, mlp_dim=512)
    
    vit_results = experiment.run_experiment(
        vit_factory, train_loader, val_loader, test_loader,
        "CIFAR-10 + Vision Transformer", epochs=100, lr=0.0005, weight_decay=1e-4
    )
    
    # 결과 분석 및 시각화
    experiments = [
        (cifar10_results, "CIFAR-10 + ResNet18"),
        (cifar100_results, "CIFAR-100 + ResNet18"),
        (synthetic_results, "고차원 합성 데이터 + 깊은 MLP"),
        (vit_results, "CIFAR-10 + Vision Transformer")
    ]
    
    for results, name in experiments:
        experiment.analyze_results(results, name)
        save_path = f"/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/{name.replace(' ', '_').replace('+', '_')}_results.png"
        experiment.plot_comprehensive_results(results, name, save_path)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results/{name.replace(' ', '_')}_{timestamp}.json"
        os.makedirs('/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results', exist_ok=True)
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for opt_name, result in results.items():
            serializable_results[opt_name] = {
                'best_val_acc': result['best_val_acc'],
                'final_test_acc': result['final_test_acc'],
                'total_time': result['total_time'],
                'epochs_trained': result['epochs_trained'],
                'total_params': result['total_params'],
                'trainable_params': result['trainable_params']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n결과 저장 완료: {results_file}")
    
    print("\n모든 고차원 실험이 완료되었습니다!")


if __name__ == "__main__":
    main()