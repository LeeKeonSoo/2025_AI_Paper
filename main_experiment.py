"""
메인 실험 실행 파일
Adam, AdamW, AdamABS 옵티마이저 비교 실험

데이터셋 선택:
1 - MNIST
2 - CIFAR-10  
3 - Tiny ImageNet

Author: AI Research
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import sys
import time
from datetime import datetime
import os

# 우리가 만든 모듈들 import
from optimizers import create_optimizer, CustomAdam, CustomAdamW, CustomAdamABS
from data_loaders import get_dataset_loader, print_dataset_info
from models import create_model, print_model_summary, get_model_info
from trainer import OptimizerExperiment, create_standard_scheduler
from visualizer import ExperimentVisualizer
from weight_manager import WeightManager, ContinuousTrainer


def setup_experiment_config(dataset_type: int, epochs: int = None, lr: float = None) -> dict:
    """
    데이터셋별 기본 실험 설정
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
        epochs: 사용자 지정 에포크 수 (None이면 기본값 사용)
        lr: 사용자 지정 학습률 (None이면 기본값 사용)
    
    Returns:
        dict: 실험 설정
    """
    # 데이터셋별 기본 설정
    default_configs = {
        1: {  # MNIST
            'epochs': 15,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 128,
            'model_type': 'default',
            'scheduler_type': 'cosine'
        },
        2: {  # CIFAR-10
            'epochs': 50,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'batch_size': 128,
            'model_type': 'default',
            'scheduler_type': 'cosine'
        },
        3: {  # Tiny ImageNet - 과적합 방지 강화 설정 (ResNet)
            'epochs': 50,         # 더 긴 학습으로 일반화 성능 향상
            'lr': 0.0005,         # 더 보수적인 학습률 (0.001 → 0.0005)
            'weight_decay': 5e-4, # 더 강한 L2 정규화 (1e-4 → 5e-4)
            'batch_size': 128,    # 유지
            'model_type': 'default',  # ResNet-18 사용
            'scheduler_type': 'cosine'
        }
    }
    
    config = default_configs[dataset_type].copy()
    
    # 사용자 지정값으로 오버라이드
    if epochs is not None:
        config['epochs'] = epochs
    if lr is not None:
        config['lr'] = lr
    
    return config


def create_optimizers_config(base_lr: float, weight_decay: float) -> dict:
    """
    옵티마이저 설정 생성
    
    Args:
        base_lr: 기본 학습률
        weight_decay: 가중치 감소
    
    Returns:
        dict: 옵티마이저 설정 딕셔너리
    """
    return {
        'Adam': {
            'optimizer_class': CustomAdam,
            'params': {
                'lr': base_lr,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': weight_decay
            }
        },
        'AdamW': {
            'optimizer_class': CustomAdamW,
            'params': {
                'lr': base_lr,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': weight_decay
            }
        },
        'AdamABS': {
            'optimizer_class': CustomAdamABS,
            'params': {
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': weight_decay
            }
        },
    }


def run_experiment(dataset_type: int, epochs: int = None, lr: float = None, 
                  batch_size: int = None, model_type: str = None,
                  optimizers_to_test: list = None, resume_training: bool = False) -> dict:
    """
    메인 실험 실행
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
        epochs: 훈련 에포크 수
        lr: 학습률
        batch_size: 배치 크기
        model_type: 모델 타입
        optimizers_to_test: 테스트할 옵티마이저 리스트
        resume_training: 기존 체크포인트에서 훈련 재개 여부
    
    Returns:
        dict: 실험 결과
    """
    
    # 실험 설정
    config = setup_experiment_config(dataset_type, epochs, lr)
    if batch_size is not None:
        config['batch_size'] = batch_size
    if model_type is not None:
        config['model_type'] = model_type
    
    # 데이터셋 이름 매핑
    dataset_names = {1: "MNIST", 2: "CIFAR-10", 3: "Tiny ImageNet"}
    dataset_name = dataset_names[dataset_type]
    
    print("=" * 100)
    print(f"🚀 {dataset_name} 옵티마이저 비교 실험 시작")
    print("=" * 100)
    print(f"📋 실험 설정:")
    print(f"   데이터셋: {dataset_name}")
    print(f"   에포크: {config['epochs']}")
    print(f"   학습률: {config['lr']}")
    print(f"   배치 크기: {config['batch_size']}")
    print(f"   가중치 감소: {config['weight_decay']}")
    print(f"   모델: {config['model_type']}")
    print(f"   스케줄러: {config['scheduler_type']}")
    print("=" * 100)
    
    # 1. 데이터 로더 생성
    print("\n🔄 데이터 로더 생성 중...")
    try:
        data_loader = get_dataset_loader(
            dataset_type=dataset_type,
            batch_size=config['batch_size'],
            num_workers=4 if torch.cuda.is_available() else 2
        )
    except Exception as e:
        print(f"❌ 데이터 로더 생성 실패: {e}")
        return None
    
    # 2. 모델 정보 출력
    dataset_info = data_loader.get_dataset_info()
    print(f"\n📊 데이터셋 정보:")
    print(f"   이름: {dataset_info['name']}")
    print(f"   클래스 수: {dataset_info['num_classes']}")
    print(f"   이미지 크기: {dataset_info['image_size']}")
    
    # 3. 옵티마이저 설정
    optimizers_config = create_optimizers_config(config['lr'], config['weight_decay'])
    
    # 사용자가 특정 옵티마이저만 테스트하고 싶은 경우
    if optimizers_to_test:
        optimizers_config = {name: optimizers_config[name] 
                           for name in optimizers_to_test if name in optimizers_config}
    
    print(f"\n🔧 테스트할 옵티마이저: {list(optimizers_config.keys())}")
    
    # 4. 모델 팩토리 함수 정의
    def model_factory():
        """모델 생성 함수"""
        model = create_model(dataset_type, config['model_type'])
        
        # 모델 요약 출력 (첫 번째 모델에서만)
        if not hasattr(model_factory, '_first_call_done'):
            print(f"\n🏗️  모델 정보:")
            model_info = get_model_info(dataset_type)
            print(f"   모델: {model.__class__.__name__}")
            print(f"   파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"   입력 크기: {model_info['input_size']}")
            print(f"   출력 클래스: {model_info['num_classes']}")
            model_factory._first_call_done = True
        
        return model
    
    # 5. 스케줄러 팩토리 함수 정의
    def scheduler_factory(optimizer):
        """스케줄러 생성 함수"""
        return create_standard_scheduler(
            optimizer, 
            scheduler_type=config['scheduler_type'], 
            epochs=config['epochs']
        )
    
    # 6. 실험 실행
    print(f"\n🧪 실험 시작...")
    experiment = OptimizerExperiment(dataset_type, config['model_type'], enable_checkpoints=True)
    
    # 체크포인트 재개 옵션 표시
    if resume_training:
        print("🔄 기존 체크포인트에서 훈련 재개 활성화")
    
    start_time = time.time()
    
    results = experiment.run_comparison_experiment(
        optimizers_config=optimizers_config,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        test_loader=data_loader.test_loader,
        model_factory_fn=model_factory,
        epochs=config['epochs'],
        scheduler_factory_fn=scheduler_factory,
        resume_from_checkpoint=resume_training
    )
    
    total_experiment_time = time.time() - start_time
    
    # 7. 결과 시각화 및 저장
    print(f"\n📊 결과 시각화 중...")
    visualizer = ExperimentVisualizer('./results')
    saved_files = visualizer.generate_all_visualizations(results, dataset_name)
    
    # 8. 실험 완료 메시지
    print("\n" + "=" * 100)
    print("🎉 실험 완료!")
    print("=" * 100)
    print(f"총 실험 시간: {total_experiment_time/3600:.2f}시간 ({total_experiment_time:.1f}초)")
    print(f"결과 파일 위치: {os.path.abspath('./results')}")
    print("\n📁 생성된 파일:")
    for file_type, file_path in saved_files.items():
        print(f"   {file_type}: {os.path.basename(file_path)}")
    
    # 최종 요약
    best_optimizer = max(results.keys(), key=lambda x: results[x]['best_val_acc'])
    print(f"\n🏆 최고 성능 옵티마이저: {best_optimizer}")
    print(f"   검증 정확도: {results[best_optimizer]['best_val_acc']:.2f}%")
    if 'test_results' in results[best_optimizer]:
        print(f"   테스트 정확도: {results[best_optimizer]['test_results']['accuracy']:.2f}%")
    
    print("=" * 100)
    
    return results


def quick_test(dataset_type: int = 1):
    """
    빠른 테스트 실행 (적은 에포크로)
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
    """
    print("⚡ 빠른 테스트 모드")
    return run_experiment(
        dataset_type=dataset_type,
        epochs=3,  # 적은 에포크
        optimizers_to_test=['Adam', 'AdamABS']  # 2개만 테스트
    )


def compare_adam_vs_adamabs(dataset_type: int = 1):
    """
    Adam vs AdamABS 집중 비교
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
    """
    print("🔬 Adam vs AdamABS 집중 비교")
    return run_experiment(
        dataset_type=dataset_type,
        optimizers_to_test=['Adam', 'AdamABS']
    )


def full_comparison(dataset_type: int = 1):
    """
    모든 옵티마이저 전체 비교
    
    Args:
        dataset_type: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)
    """
    print("🔬 전체 옵티마이저 비교")
    return run_experiment(dataset_type=dataset_type)


def interactive_mode():
    """대화형 모드"""
    print("🎯 대화형 실험 모드")
    print("=" * 60)
    
    # 데이터셋 선택
    print_dataset_info()
    print()
    
    while True:
        try:
            dataset_choice = int(input("데이터셋을 선택하세요 (1-3): "))
            if dataset_choice in [1, 2, 3]:
                break
            else:
                print("❌ 1, 2, 3 중에서 선택해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    # 체크포인트 관리 옵션
    print("\n체크포인트 관리:")
    print("1. 저장된 체크포인트 목록 보기")
    print("2. 실험 진행하기")
    
    while True:
        try:
            checkpoint_choice = int(input("선택하세요 (1-2): "))
            if checkpoint_choice in [1, 2]:
                break
            else:
                print("❌ 1, 2 중에서 선택해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    if checkpoint_choice == 1:
        # 체크포인트 목록 표시
        wm = WeightManager("./weights")
        wm.list_checkpoints(detailed=True)
        return None
    
    # 실험 모드 선택
    print("\n실험 모드를 선택하세요:")
    print("1. 빠른 테스트 (3 에포크, Adam vs AdamABS)")
    print("2. Adam vs AdamABS 집중 비교")
    print("3. 전체 옵티마이저 비교")
    print("4. 사용자 정의 설정")
    
    while True:
        try:
            mode_choice = int(input("모드를 선택하세요 (1-4): "))
            if mode_choice in [1, 2, 3, 4]:
                break
            else:
                print("❌ 1, 2, 3, 4 중에서 선택해주세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    # 훈련 재개 옵션
    resume_input = input("\n기존 체크포인트에서 훈련을 재개하시겠습니까? (y/N): ").strip().lower()
    resume_training = resume_input in ['y', 'yes', '예']
    
    # 실험 실행
    if mode_choice == 1:
        return run_experiment(dataset_choice, epochs=3, optimizers_to_test=['Adam', 'AdamABS'], resume_training=resume_training)
    elif mode_choice == 2:
        return run_experiment(dataset_choice, optimizers_to_test=['Adam', 'AdamABS'], resume_training=resume_training)
    elif mode_choice == 3:
        return run_experiment(dataset_choice, resume_training=resume_training)
    elif mode_choice == 4:
        # 사용자 정의 설정
        print("\n사용자 정의 설정:")
        
        epochs = None
        epochs_input = input("에포크 수 (기본값 사용하려면 엔터): ").strip()
        if epochs_input:
            try:
                epochs = int(epochs_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값을 사용합니다.")
        
        lr = None
        lr_input = input("학습률 (기본값 사용하려면 엔터): ").strip()
        if lr_input:
            try:
                lr = float(lr_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값을 사용합니다.")
        
        return run_experiment(dataset_choice, epochs=epochs, lr=lr, resume_training=resume_training)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Adam, AdamW, AdamABS 옵티마이저 비교 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main_experiment.py                     # 대화형 모드
  python main_experiment.py --dataset 1        # MNIST 전체 비교
  python main_experiment.py --dataset 2 --quick # CIFAR-10 빠른 테스트
  python main_experiment.py --dataset 3 --epochs 20 --lr 0.0001
        """
    )
    
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3],
                       help='데이터셋 선택: 1(MNIST), 2(CIFAR-10), 3(Tiny ImageNet)')
    parser.add_argument('--epochs', type=int,
                       help='훈련 에포크 수')
    parser.add_argument('--lr', type=float,
                       help='학습률')
    parser.add_argument('--batch-size', type=int,
                       help='배치 크기')
    parser.add_argument('--model', type=str, choices=['default', 'simple', 'resnet'],
                       help='모델 타입')
    parser.add_argument('--optimizers', nargs='+', 
                       choices=['Adam', 'AdamW', 'AdamABS'],
                       help='테스트할 옵티마이저 (기본: 모두)')
    parser.add_argument('--quick', action='store_true',
                       help='빠른 테스트 모드 (3 에포크)')
    parser.add_argument('--adam-vs-adamabs', action='store_true',
                       help='Adam vs AdamABS만 비교')
    parser.add_argument('--resume', action='store_true',
                       help='기존 체크포인트에서 훈련 재개')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='저장된 체크포인트 목록 표시')
    
    args = parser.parse_args()
    
    # 체크포인트 목록 표시
    if args.list_checkpoints:
        wm = WeightManager("./weights")
        wm.list_checkpoints(detailed=True)
        return None
    
    # 명령행 인자가 없으면 대화형 모드
    if len(sys.argv) == 1:
        return interactive_mode()
    
    # 데이터셋이 지정되지 않으면 에러
    if args.dataset is None:
        print("❌ --dataset 옵션을 지정해주세요. (1: MNIST, 2: CIFAR-10, 3: Tiny ImageNet)")
        return None
    
    # 실험 모드에 따른 실행
    if args.quick:
        return quick_test(args.dataset)
    elif args.adam_vs_adamabs:
        return compare_adam_vs_adamabs(args.dataset)
    else:
        return run_experiment(
            dataset_type=args.dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            model_type=args.model,
            optimizers_to_test=args.optimizers,
            resume_training=args.resume
        )


if __name__ == "__main__":
    # 시작 메시지
    print("🧠 Adam vs AdamABS 옵티마이저 비교 실험")
    print("=" * 60)
    print("📅 실행 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🖥️  디바이스:", "CUDA" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        print("🎮 GPU:", torch.cuda.get_device_name(0))
    print("=" * 60)
    
    try:
        results = main()
        if results:
            print("\n✅ 실험이 성공적으로 완료되었습니다!")
        else:
            print("\n❌ 실험이 실패했습니다.")
    except KeyboardInterrupt:
        print("\n⏹️  실험이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 실험 종료")