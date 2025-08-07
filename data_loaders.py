"""
데이터셋 로더 통합 모듈
MNIST, CIFAR-10, Tiny ImageNet 지원

Dataset Selection:
1 - MNIST
2 - CIFAR-10  
3 - Tiny ImageNet

Author: AI Research
Date: 2025
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
from typing import Tuple, Optional, Dict, Any


class TinyImageNetDataset(torch.utils.data.Dataset):
    """간단한 Tiny ImageNet 데이터셋 클래스"""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Tiny ImageNet 데이터셋 루트 디렉토리 (tiny-imagenet-200)
            split: 'train' 또는 'val'
            transform: 이미지 변환
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 클래스 정보 로드 (train 폴더 기준으로 간단하게)
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 이미지 경로와 라벨 로드
        self.samples = self._load_samples()
    
    def _load_classes(self):
        """클래스 목록 로드 (train 폴더 기준)"""
        train_dir = os.path.join(self.root_dir, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train 폴더를 찾을 수 없습니다: {train_dir}")
        
        classes = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
        classes.sort()  # 일관성을 위해 정렬
        return classes
    
    def _load_samples(self):
        """이미지 경로와 라벨 로드"""
        samples = []
        
        if self.split == 'train':
            # Train 데이터 로드
            train_dir = os.path.join(self.root_dir, 'train')
            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                            img_path = os.path.join(class_dir, img_name)
                            samples.append((img_path, self.class_to_idx[class_name]))
        
        elif self.split == 'val':
            # Val 데이터 로드
            val_dir = os.path.join(self.root_dir, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            
                            if class_name in self.class_to_idx:
                                img_path = os.path.join(val_dir, 'images', img_name)
                                if os.path.exists(img_path):
                                    samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"✅ Tiny ImageNet {self.split} 로드: {len(samples):,}개 샘플")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DatasetLoader:
    """통합 데이터셋 로더"""
    
    def __init__(self, dataset_type: int, data_dir: str = './data', batch_size: int = 32, 
                 num_workers: int = 0, validation_split: float = 0.1):
        """
        Args:
            dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
            data_dir: 데이터 디렉토리 경로
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
            validation_split: 검증 데이터 분할 비율 (Tiny ImageNet은 무시됨)
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
        elif self.dataset_type == 3:  # Tiny ImageNet
            return {
                'name': 'Tiny ImageNet',
                'num_classes': 200,
                'image_size': (3, 64, 64),
                'mean': (0.485, 0.456, 0.406),  # ImageNet 표준값
                'std': (0.229, 0.224, 0.225),   # ImageNet 표준값
                'channels': 3
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
            
        elif self.dataset_type == 3:  # Tiny ImageNet - 오버피팅 방지 강화
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),    # 🔧 스케일 범위 확대
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),                  # 🔧 회전 각도 증가
                transforms.ColorJitter(                                # 🔧 색상 증강 강화
                    brightness=0.2,
                    contrast=0.2, 
                    saturation=0.1,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(info['mean'], info['std']),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))    # 🔧 랜덤 삭제 추가
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize(64),  # 정확한 크기로 리사이즈
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
        elif self.dataset_type == 3:  # Tiny ImageNet
            return self._create_tiny_imagenet_loaders(train_transform, test_transform)
    
    def _create_mnist_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """MNIST 데이터 로더 생성 - GPU 최적화"""
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
        
        # GPU 최적화된 데이터 로더 생성
        optimal_workers = min(os.cpu_count() or 4, 8)
        use_pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_cifar10_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """CIFAR-10 데이터 로더 생성 - GPU 최적화"""
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
        
        # GPU 최적화된 데이터 로더 생성
        optimal_workers = min(os.cpu_count() or 4, 8)
        use_pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_tiny_imagenet_loaders(self, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Tiny ImageNet 데이터 로더 생성 - 단순화"""
        tiny_imagenet_dir = os.path.join(self.data_dir, 'tiny-imagenet-200')
        
        if not os.path.exists(tiny_imagenet_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet 데이터셋을 찾을 수 없습니다: {tiny_imagenet_dir}\n"
                f"다운로드: http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
        
        # 데이터셋 생성
        train_dataset = TinyImageNetDataset(tiny_imagenet_dir, 'train', train_transform)
        val_dataset = TinyImageNetDataset(tiny_imagenet_dir, 'val', test_transform)
        test_dataset = TinyImageNetDataset(tiny_imagenet_dir, 'val', test_transform)  # val을 test로 사용
        
        # GPU 최적화된 데이터 로더 생성
        # CPU 코어 수에 따른 워커 수 자동 조정
        optimal_workers = min(os.cpu_count() or 4, 8)  # 최대 8개 워커
        use_pin_memory = torch.cuda.is_available()  # GPU 있을 때만 pin_memory 사용
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=optimal_workers, pin_memory=use_pin_memory,
            persistent_workers=True, prefetch_factor=2
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
        elif self.dataset_type == 3:  # Tiny ImageNet
            # Tiny ImageNet은 200개 클래스가 있으므로 간략히 표시
            return [f'class_{i}' for i in range(200)]
        else:
            return []
    
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
        raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}. 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet) 중 선택하세요.")
    
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
    
    print("\n3. Tiny ImageNet")
    print("   - 소형 ImageNet 분류")
    print("   - 이미지 크기: 64x64 (컬러)")
    print("   - 클래스 수: 200개")
    print("   - 훈련 샘플: 100,000개")
    print("   - 검증 샘플: 10,000개")
    print("   - 수동 다운로드 필요: ./data/tiny-imagenet-200/")
    
    print("\n사용법:")
    print("   loader = get_dataset_loader(dataset_type=1)  # MNIST")
    print("   loader = get_dataset_loader(dataset_type=2)  # CIFAR-10")
    print("   loader = get_dataset_loader(dataset_type=3)  # Tiny ImageNet")
    print("=" * 60)


if __name__ == "__main__":
    # 데이터셋 정보 출력
    print_dataset_info()
    
    # 각 데이터셋 테스트
    print("\n데이터셋 로더 테스트")
    print("=" * 60)
    
    for dataset_type in [1, 2, 3]:  # MNIST, CIFAR-10, Tiny ImageNet 모두 테스트
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
    
    # Tiny ImageNet 테스트
    try:
        print(f"\n3번 데이터셋 (Tiny ImageNet) 테스트 중...")
        loader = get_dataset_loader(dataset_type=3, batch_size=32, num_workers=2)
        images, labels = loader.get_sample_batch()
        print(f"   샘플 배치 크기: {images.shape}")
        print(f"   라벨 크기: {labels.shape}")
        print(f"   데이터 타입: {images.dtype}")
        print(f"   ✅ Tiny ImageNet 성공!")
    except Exception as e:
        print(f"   ❌ Tiny ImageNet 실패: {e}")
    
    print("\n✅ 데이터셋 로더 테스트 완료!")