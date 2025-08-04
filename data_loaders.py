"""
데이터셋 로더 통합 모듈
MNIST, CIFAR-10, Tiny ImageNet 지원

Dataset Selection:
1 - MNIST
2 - CIFAR-10  
3 - Fashion-MNIST

Author: AI Research
Date: 2025
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, Dict, Any


# Fashion-MNIST uses torchvision dataset, no custom class needed


class DatasetLoader:
    """통합 데이터셋 로더"""
    
    def __init__(self, dataset_type: int, data_dir: str = './data', batch_size: int = 128, 
                 num_workers: int = 4, validation_split: float = 0.1):
        """
        Args:
            dataset_type: 1(MNIST), 2(CIFAR-10), 3(Fashion-MNIST)
            data_dir: 데이터 디렉토리 경로
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
            validation_split: 검증 데이터 분할 비율
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        
        # 데이터셋 정보 설정
        self.dataset_info = self._get_dataset_info()
        
        # 데이터 로더 생성
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        print(f"✅ {self.dataset_info['name']} 데이터셋 로드 완료")
        print(f"   클래스 수: {self.dataset_info['num_classes']}")
        print(f"   이미지 크기: {self.dataset_info['image_size']}")
        print(f"   훈련 샘플: {len(self.train_loader.dataset):,}개")
        print(f"   검증 샘플: {len(self.val_loader.dataset):,}개")
        print(f"   테스트 샘플: {len(self.test_loader.dataset):,}개")
        print(f"   배치 크기: {self.batch_size}")
        
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        if self.dataset_type == 1:  # MNIST
            return {
                'name': 'MNIST',
                'num_classes': 10,
                'image_size': (1, 28, 28),
                'mean': (0.1307,),
                'std': (0.3081,),
                'channels': 1
            }
        elif self.dataset_type == 2:  # CIFAR-10
            return {
                'name': 'CIFAR-10',
                'num_classes': 10,
                'image_size': (3, 32, 32),
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2023, 0.1994, 0.2010),
                'channels': 3
            }
        elif self.dataset_type == 3:  # Fashion-MNIST
            return {
                'name': 'Fashion-MNIST',
                'num_classes': 10,
                'image_size': (1, 28, 28),
                'mean': (0.2860,),
                'std': (0.3530,),
                'channels': 1
            }
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입: {self.dataset_type}")
    
    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """데이터셋별 변환 함수 반환"""
        info = self.dataset_info
        
        if self.dataset_type == 1:  # MNIST
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
            
        elif self.dataset_type == 2:  # CIFAR-10
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
            
        elif self.dataset_type == 3:  # Fashion-MNIST
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std'])
            ])
        
        return train_transform, test_transform
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 로더 생성"""
        train_transform, test_transform = self._get_transforms()
        
        if self.dataset_type == 1:  # MNIST
            return self._create_mnist_loaders(train_transform, test_transform)
        elif self.dataset_type == 2:  # CIFAR-10
            return self._create_cifar10_loaders(train_transform, test_transform)
        elif self.dataset_type == 3:  # Fashion-MNIST
            return self._create_fashion_mnist_loaders(train_transform, test_transform)
    
    def _create_mnist_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """MNIST 데이터 로더 생성"""
        # 전체 훈련 데이터셋 로드
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        
        # 테스트 데이터셋 로드
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        
        # 훈련/검증 데이터 분할
        train_size = int((1 - self.validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_cifar10_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """CIFAR-10 데이터 로더 생성"""
        # 전체 훈련 데이터셋 로드
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        
        # 테스트 데이터셋 로드
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        
        # 훈련/검증 데이터 분할
        train_size = int((1 - self.validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_fashion_mnist_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Fashion-MNIST 데이터 로더 생성"""
        # 전체 훈련 데이터셋 로드
        full_train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        
        # 테스트 데이터셋 로드
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        
        # 훈련/검증 데이터 분할
        train_size = int((1 - self.validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        return self.dataset_info
    
    def get_class_names(self) -> list:
        """클래스 이름 반환"""
        if self.dataset_type == 1:  # MNIST
            return [str(i) for i in range(10)]
        elif self.dataset_type == 2:  # CIFAR-10
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_type == 3:  # Fashion-MNIST
            return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def get_sample_batch(self):
        """샘플 배치 반환 (시각화나 디버깅용)"""
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        return images, labels


def get_dataset_loader(dataset_type: int, **kwargs) -> DatasetLoader:
    """
    데이터셋 로더 팩토리 함수
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
        **kwargs: DatasetLoader 생성자에 전달할 추가 인자
    
    Returns:
        DatasetLoader: 구성된 데이터셋 로더
    """
    if dataset_type not in [1, 2, 3]:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}. 1(MNIST), 2(CIFAR-10), 3(Fashion-MNIST) 중 선택하세요.")
    
    return DatasetLoader(dataset_type=dataset_type, **kwargs)


def print_dataset_info():
    """지원하는 데이터셋 정보 출력"""
    print("=" * 60)
    print("지원하는 데이터셋")
    print("=" * 60)
    print("1. MNIST")
    print("   - 손글씨 숫자 분류 (0-9)")
    print("   - 이미지 크기: 28x28 (흑백)")
    print("   - 클래스 수: 10개")
    print("   - 훈련 샘플: 60,000개")
    print("   - 테스트 샘플: 10,000개")
    
    print("\n2. CIFAR-10")
    print("   - 자연 이미지 분류")
    print("   - 이미지 크기: 32x32 (컬러)")
    print("   - 클래스 수: 10개")
    print("   - 훈련 샘플: 50,000개")
    print("   - 테스트 샘플: 10,000개")
    
    print("\n3. Fashion-MNIST")
    print("   - 패션 아이템 분류")
    print("   - 이미지 크기: 28x28 (흑백)")
    print("   - 클래스 수: 10개")
    print("   - 훈련 샘플: 60,000개")
    print("   - 테스트 샘플: 10,000개")
    print("   - 자동 다운로드됨")
    
    print("\n사용법:")
    print("   loader = get_dataset_loader(dataset_type=1)  # MNIST")
    print("   loader = get_dataset_loader(dataset_type=2)  # CIFAR-10")
    print("   loader = get_dataset_loader(dataset_type=3)  # Fashion-MNIST")
    print("=" * 60)


if __name__ == "__main__":
    # 데이터셋 정보 출력
    print_dataset_info()
    
    # 각 데이터셋 테스트
    print("\n데이터셋 로더 테스트")
    print("=" * 60)
    
    for dataset_type in [1, 2, 3]:  # MNIST, CIFAR-10, Fashion-MNIST 모두 테스트
        try:
            print(f"\n{dataset_type} 번 데이터셋 테스트 중...")
            loader = get_dataset_loader(
                dataset_type=dataset_type, 
                batch_size=32, 
                num_workers=2
            )
            
            # 샘플 배치 테스트
            images, labels = loader.get_sample_batch()
            print(f"   샘플 배치 크기: {images.shape}")
            print(f"   라벨 크기: {labels.shape}")
            print(f"   데이터 타입: {images.dtype}")
            print(f"   ✅ 성공!")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # Fashion-MNIST 테스트
    try:
        print(f"\n3번 데이터셋 (Fashion-MNIST) 테스트 중...")
        loader = get_dataset_loader(dataset_type=3, batch_size=32, num_workers=2)
        images, labels = loader.get_sample_batch()
        print(f"   샘플 배치 크기: {images.shape}")
        print(f"   라벨 크기: {labels.shape}")
        print(f"   데이터 타입: {images.dtype}")
        print(f"   ✅ Fashion-MNIST 성공!")
    except Exception as e:
        print(f"   ❌ Fashion-MNIST 실패: {e}")
    
    print("\n✅ 데이터셋 로더 테스트 완료!")