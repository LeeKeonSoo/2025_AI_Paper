"""
모델 정의 통합 모듈
MNIST, CIFAR-10, Tiny ImageNet용 모델들

Author: AI Research  
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


# =============================================================================
# MNIST 모델들
# =============================================================================

class MNISTNet(nn.Module):
    """MNIST용 CNN 모델"""
    
    def __init__(self, dropout_rate: float = 0.25):
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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
    """간단한 MNIST 완전연결 모델"""
    
    def __init__(self, dropout_rate: float = 0.2):
        super(SimpleMNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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


# =============================================================================
# CIFAR-10 모델들
# =============================================================================

class ResNetBlock(nn.Module):
    """ResNet 기본 블록"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10ResNet(nn.Module):
    """CIFAR-10용 ResNet"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        super(CIFAR10ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int):
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


class SimpleCIFAR10Net(nn.Module):
    """간단한 CIFAR-10 CNN 모델"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        super(SimpleCIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Tiny ImageNet 모델들
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet-18/34용 기본 블록"""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """Tiny ImageNet용 ResNet-18 (성공했던 커밋의 단순한 구조 복원)"""
    
    def __init__(self, num_classes: int = 200, dropout_rate: float = 0.3):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        
        # 첫 번째 컨볼루션 - Tiny ImageNet에 최적화 (64x64 입력)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 64x64는 작으므로 maxpool 제거
        
        # ResNet-18 구조: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🔧 과적합 방지를 위한 dropout 레이어들 추가 (성공했던 커밋 구조)
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)  # 중간 층에 약한 dropout
        self.dropout2 = nn.Dropout(dropout_rate)         # 최종 층에 강한 dropout
        
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 개선된 Linear layer 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # 64x64 입력에서는 maxpool 제거
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)  # 중간에 약한 dropout
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(x)  # 최종 분류 전에 강한 dropout
        x = self.fc(x)
        return x


class AdamOptimizedResNet18(nn.Module):
    """과적합 방지를 위한 균형 잡힌 ResNet-18"""
    
    def __init__(self, num_classes: int = 200, dropout_rate: float = 0.4):
        super(AdamOptimizedResNet18, self).__init__()
        self.in_planes = 64
        
        # 첫 번째 컨볼루션 - Tiny ImageNet에 최적화
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet-18 구조: [2, 2, 2, 2] - 기본 구조 유지
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 과적합 방지를 위한 단순한 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # 강한 dropout 적용
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Adam에 최적화된 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 🚀 Adam에 유리한 He 초기화 (Xavier보다 Adam과 더 잘 맞음)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 🎯 Adam에 최적화된 Linear layer 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 단순한 분류
        x = self.classifier(x)
        
        return x


class SimpleTinyImageNetNet(nn.Module):
    """간단한 Tiny ImageNet CNN 모델"""
    
    def __init__(self, num_classes: int = 200, dropout_rate: float = 0.3):
        super(SimpleTinyImageNetNet, self).__init__()
        
        self.features = nn.Sequential(
            # 첫 번째 블록 (64x64 -> 32x32)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # 두 번째 블록 (32x32 -> 16x16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.7),
            
            # 세 번째 블록 (16x16 -> 8x8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # 네 번째 블록 (8x8 -> 4x4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 모델 팩토리 함수들
# =============================================================================

def create_model(dataset_type: int, model_type: str = 'default', **kwargs) -> nn.Module:
    """
    데이터셋과 모델 타입에 따라 적절한 모델 생성
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
        model_type: 'default', 'simple', 'resnet' 등
        **kwargs: 모델별 추가 파라미터
    
    Returns:
        nn.Module: 생성된 모델
    """
    dropout_rate = kwargs.get('dropout_rate', 0.2)
    
    if dataset_type == 1:  # MNIST
        if model_type.lower() in ['default', 'cnn']:
            return MNISTNet(dropout_rate=dropout_rate)
        elif model_type.lower() == 'simple':
            return SimpleMNISTNet(dropout_rate=dropout_rate)
        else:
            raise ValueError(f"MNIST에서 지원하지 않는 모델 타입: {model_type}")
    
    elif dataset_type == 2:  # CIFAR-10
        if model_type.lower() in ['default', 'resnet']:
            return CIFAR10ResNet(num_classes=10, dropout_rate=dropout_rate)
        elif model_type.lower() == 'simple':
            return SimpleCIFAR10Net(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"CIFAR-10에서 지원하지 않는 모델 타입: {model_type}")
    
    elif dataset_type == 3:  # Tiny ImageNet
        if model_type.lower() in ['default', 'resnet', 'resnet18']:
            # 🚀 Adam 계열 성능 부각을 위해 Adam 최적화 모델을 기본값으로 설정
            return AdamOptimizedResNet18(num_classes=200, dropout_rate=dropout_rate)
        elif model_type.lower() == 'simple':
            return SimpleTinyImageNetNet(num_classes=200, dropout_rate=dropout_rate)
        elif model_type.lower() in ['original_resnet', 'basic_resnet']:
            # 원래 ResNet18은 별도 옵션으로 제공
            return ResNet18(num_classes=200, dropout_rate=dropout_rate)
        # 🚀 Adam 최적화 모델 추가
        elif model_type.lower() in ['adam_optimized', 'adam_resnet']:
            return AdamOptimizedResNet18(num_classes=200, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Tiny ImageNet에서 지원하지 않는 모델 타입: {model_type}")
    
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")


def get_model_info(dataset_type: int) -> Dict[str, Any]:
    """데이터셋별 사용 가능한 모델 정보 반환"""
    model_info = {
        1: {  # MNIST
            'dataset': 'MNIST',
            'models': {
                'default': 'MNISTNet (CNN with BatchNorm)',
                'cnn': 'MNISTNet (same as default)',
                'simple': 'SimpleMNISTNet (Fully Connected)'
            },
            'num_classes': 10,
            'input_size': (1, 28, 28)
        },
        2: {  # CIFAR-10
            'dataset': 'CIFAR-10',
            'models': {
                'default': 'CIFAR10ResNet (ResNet-like architecture)',
                'resnet': 'CIFAR10ResNet (same as default)',
                'simple': 'SimpleCIFAR10Net (Basic CNN)'
            },
            'num_classes': 10,
            'input_size': (3, 32, 32)
        },
        3: {  # Tiny ImageNet
            'dataset': 'Tiny ImageNet',
            'models': {
                'default': 'AdamOptimizedResNet18 (Adam-friendly ResNet-18)',
                'resnet': 'AdamOptimizedResNet18 (same as default)',
                'resnet18': 'AdamOptimizedResNet18 (same as default)',
                'simple': 'SimpleTinyImageNetNet (Basic CNN)',
                'original_resnet': 'ResNet18 (Original ResNet-18)',
                'basic_resnet': 'ResNet18 (same as original_resnet)',
                'adam_optimized': 'AdamOptimizedResNet18 (Adam-friendly ResNet-18)',
                'adam_resnet': 'AdamOptimizedResNet18 (same as adam_optimized)'
            },
            'num_classes': 200,
            'input_size': (3, 64, 64)
        }
    }
    
    if dataset_type not in model_info:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")
    
    return model_info[dataset_type]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델의 파라미터 개수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model: nn.Module, dataset_type: int):
    """모델 요약 정보 출력"""
    info = get_model_info(dataset_type)
    params = count_parameters(model)
    
    print("=" * 60)
    print(f"모델 요약: {model.__class__.__name__}")
    print("=" * 60)
    print(f"데이터셋: {info['dataset']}")
    print(f"클래스 수: {info['num_classes']}")
    print(f"입력 크기: {info['input_size']}")
    print(f"총 파라미터: {params['total']:,}")
    print(f"훈련 가능 파라미터: {params['trainable']:,}")
    print(f"훈련 불가 파라미터: {params['non_trainable']:,}")
    
    # 모델 구조 출력 (간단히)
    print(f"\n모델 구조:")
    print(model)
    print("=" * 60)


def print_all_models_info():
    """모든 지원 모델 정보 출력"""
    print("=" * 80)
    print("지원하는 모델들")
    print("=" * 80)
    
    for dataset_type in [1, 2, 3]:
        info = get_model_info(dataset_type)
        print(f"\n{dataset_type}. {info['dataset']}")
        print(f"   클래스 수: {info['num_classes']}")
        print(f"   입력 크기: {info['input_size']}")
        print("   사용 가능한 모델:")
        
        for model_key, model_desc in info['models'].items():
            print(f"     - {model_key}: {model_desc}")
    
    print("\n사용법:")
    print("   model = create_model(dataset_type=1, model_type='default')")
    print("   model = create_model(dataset_type=2, model_type='simple')")
    print("   model = create_model(dataset_type=3, model_type='resnet')  # Tiny ImageNet")
    print("   model = create_model(dataset_type=3, model_type='adam_optimized')  # 🚀 Adam 최적화")
    print("=" * 80)


if __name__ == "__main__":
    # 모든 모델 정보 출력
    print_all_models_info()
    
    # 각 데이터셋별로 모델 생성 테스트
    print("\n모델 생성 테스트")
    print("=" * 80)
    
    test_cases = [
        (1, 'default'),  # MNIST CNN
        (1, 'simple'),   # MNIST FC
        (2, 'default'),  # CIFAR-10 ResNet
        (2, 'simple'),   # CIFAR-10 Simple
        (3, 'default'),  # Tiny ImageNet ResNet-18
        (3, 'simple'),   # Tiny ImageNet Simple
        (3, 'adam_optimized'),  # 🚀 Adam 최적화 모델 테스트 추가
    ]
    
    for dataset_type, model_type in test_cases:
        try:
            print(f"\n데이터셋 {dataset_type}, 모델 '{model_type}' 테스트...")
            model = create_model(dataset_type, model_type)
            params = count_parameters(model)
            
            print(f"   모델: {model.__class__.__name__}")
            print(f"   파라미터: {params['total']:,}개")
            print(f"   ✅ 성공!")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 샘플 입력으로 forward pass 테스트
    print(f"\nForward pass 테스트")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    for dataset_type in [1, 2, 3]:
        try:
            info = get_model_info(dataset_type)
            model = create_model(dataset_type, 'default').to(device)
            
            # 샘플 입력 생성
            batch_size = 2
            input_tensor = torch.randn(batch_size, *info['input_size']).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            expected_shape = (batch_size, info['num_classes'])
            actual_shape = output.shape
            
            print(f"데이터셋 {dataset_type}: 입력 {input_tensor.shape} → 출력 {actual_shape}")
            
            if actual_shape == expected_shape:
                print(f"   ✅ 성공! (예상: {expected_shape})")
            else:
                print(f"   ❌ 실패! 예상: {expected_shape}, 실제: {actual_shape}")
                
        except Exception as e:
            print(f"데이터셋 {dataset_type}: ❌ 실패 - {e}")
    
    print("\n✅ 모델 테스트 완료!")