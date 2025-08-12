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

# =============================================================================
# CUDA 디버깅 설정 (import 전에 설정 필요)
# =============================================================================
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 정확한 오류 위치 파악
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # device-side assertion 활성화

# =============================================================================
# 시각화 설정
# =============================================================================
SHOW_PLOTS = False  # True: 그래프 창 표시, False: 자동 저장만
SAVE_PLOTS = True   # True: 파일로 저장, False: 저장 안함

# 시각화 모드 설명:
# SHOW_PLOTS = True:  그래프가 화면에 나타나고 X 버튼을 눌러야 다음 진행
# SHOW_PLOTS = False: 그래프가 자동으로 저장되고 바로 다음 진행
# =============================================================================

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import sys
import time
from datetime import datetime
import os
import gc
from typing import Optional

# 우리가 만든 모듈들 import
from optimizers import create_optimizer, CustomAdaGrad, CustomRMSProp, CustomRMSPropABS, CustomAdam, CustomAdamW, CustomAdamABS
from data_loaders import get_dataset_loader, print_dataset_info
from models import create_model, print_model_summary, get_model_info
from trainer import OptimizerExperiment, create_standard_scheduler
from visualizer import ExperimentVisualizer, set_visualization_mode
from weight_manager import WeightManager, ContinuousTrainer

# 시각화 모드 설정 적용 (조용히)
set_visualization_mode(show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS, show_message=False)


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
        3: {  # Tiny ImageNet - 검증된 하이퍼파라미터 (ResNet-18 논문 기반)
            'epochs': 30,       # 충분한 학습 (논문 기준)
            'lr': 0.001,        # SGD 기준 검증된 학습률
            'weight_decay': 1e-4, # ResNet에 적합한 정규화 강도
            'batch_size': 100,  # 논문에서 검증된 배치 크기
            'model_type': 'default',
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


def create_optimizers_config(base_lr: float, weight_decay: float, eps: float = 1e-8) -> dict:
    """
    5개 옵티마이저 설정 생성 (논문 실험용)
    
    Args:
        base_lr: 기본 학습률 (0.0005 권장)
        weight_decay: 가중치 감소
        eps: epsilon 값 (1e-8 권장)
    
    Returns:
        dict: 옵티마이저 설정 딕셔너리
    """
    return {
        'RMSProp': {
            'optimizer_class': CustomRMSProp,
            'params': {
                'lr': base_lr,
                'alpha': 0.99,
                'eps': eps,
                'momentum': 0,
                'centered': False,
                'weight_decay': weight_decay
            }
        },
        'RMSPropABS': {
            'optimizer_class': CustomRMSPropABS,
            'params': {
                'lr': base_lr,
                'alpha': 0.99,
                'eps': eps,
                'momentum': 0,
                'centered': False,
                'weight_decay': weight_decay
            }
        },
        'Adam': {
            'optimizer_class': CustomAdam,
            'params': {
                'lr': base_lr,
                'betas': (0.9, 0.999),
                'eps': eps,
                'weight_decay': weight_decay
            }
        },
        'AdamW': {
            'optimizer_class': CustomAdamW,
            'params': {
                'lr': base_lr,
                'betas': (0.9, 0.999),
                'eps': eps,
                'weight_decay': weight_decay
            }
        },
        'AdamABS': {
            'optimizer_class': CustomAdamABS,
            'params': {
                'lr': base_lr,
                'betas': (0.9, 0.999),
                'eps': eps,
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
    
    # 시각화 모드 안내 (실험 시작 시에만)
    set_visualization_mode(show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS, show_message=True)
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
            num_workers=0  # Windows 안정성을 위해 0으로 설정
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
    
    # 3. 간단한 데이터 검증
    print(f"\n🔍 데이터 검증 중...")
    try:
        sample_data, sample_labels = next(iter(data_loader.train_loader))
        max_label = sample_labels.max().item()
        expected_classes = dataset_info['num_classes']
        
        print(f"   레이블 범위: 0 ~ {max_label}")
        print(f"   예상 클래스 수: {expected_classes}")
        
        if max_label >= expected_classes:
            raise ValueError(f"레이블({max_label}) >= 클래스 수({expected_classes})")
        
        print(f"✅ 데이터 검증 통과")
        
    except Exception as e:
        print(f"❌ 데이터 검증 실패: {e}")
        return None
    
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
        # 🔧 클래스 수를 명시적으로 전달하여 일치성 보장
        model = create_model(dataset_type, config['model_type'])
        
        # 모델 생성 직후 클래스 수 검증 (단순화)
        expected_classes = dataset_info['num_classes']
        actual_output_classes = "확인불가"
        
        # 마지막 Linear 레이어 찾기
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                actual_output_classes = module.out_features
                break
        
        if isinstance(actual_output_classes, int) and actual_output_classes != expected_classes:
            raise ValueError(f"모델 클래스 수({actual_output_classes}) != 데이터 클래스 수({expected_classes})")
        
        print(f"✅ 모델 검증 완료: {actual_output_classes}개 클래스")
        
        # 모델 요약 출력 (첫 번째 모델에서만)
        if not hasattr(model_factory, '_first_call_done'):
            print(f"\n🏗️  모델 정보:")
            model_info = get_model_info(dataset_type)
            print(f"   모델: {model.__class__.__name__}")
            print(f"   파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"   입력 크기: {model_info['input_size']}")
            print(f"   출력 클래스: {model_info['num_classes']}")
            print(f"   실제 모델 출력: {actual_output_classes}")
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
    
    # 🔧 실험 시작 전 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
    # 🔧 실험 완료 후 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
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


def tiny_imagenet_paper_experiment(batch_size: int = 128, epochs: int = 40):
    """
    논문용 Tiny ImageNet 실험 (5개 옵티마이저 비교)
    
    Args:
        batch_size: 배치 크기 (64, 128, 256 중 선택)
        epochs: 에포크 수 (기본 40)
    """
    print(f"📄 논문용 Tiny ImageNet 실험 (배치 사이즈: {batch_size}, 에포크: {epochs})")
    
    # 논문 실험용 최적 설정
    return run_experiment(
        dataset_type=3,  # Tiny ImageNet
        epochs=epochs,
        lr=0.0005,  # 논문에서 확인된 최적 학습률
        batch_size=batch_size,
        model_type='default',
        optimizers_to_test=None,  # 모든 5개 옵티마이저 테스트
        resume_training=False
    )


def hyperparameter_grid_search_experiment(dataset_type: Optional[int] = None, epochs: Optional[int] = None):
    """
    Learning rate와 epsilon 조합별 Adam vs AdamABS 비교 실험
    
    Args:
        dataset_type: 특정 데이터셋만 테스트 (None이면 모든 데이터셋)
        epochs: 에포크 수 (None이면 기본값 사용)
    
    Returns:
        dict: 모든 실험 결과
    """
    print("🚀 Hyperparameter Grid Search: Adam vs AdamABS 비교 실험")
    print("=" * 80)
    
    # 시각화 모드 안내 (실험 시작 시에만)
    set_visualization_mode(show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS, show_message=True)
    
    # 실험 설정
    datasets_to_test = [dataset_type] if dataset_type else [1, 2, 3]
    learning_rates = [0.0005, 0.001, 0.002]  # 기본값 0.001 중심
    epsilons = [1e-9, 1e-8, 1e-7]  # 기본값 1e-8 중심
    optimizers = ['Adam', 'AdamABS']
    
    dataset_names = {1: "MNIST", 2: "CIFAR-10", 3: "Tiny ImageNet"}
    
    # 전체 실험 수 계산
    total_experiments = len(datasets_to_test) * len(learning_rates) * len(epsilons) * len(optimizers)
    print(f"📋 실험 계획:")
    print(f"   데이터셋: {[dataset_names[d] for d in datasets_to_test]}")
    print(f"   Learning Rates: {learning_rates}")
    print(f"   Epsilon Values: {epsilons}")
    print(f"   옵티마이저: {optimizers}")
    print(f"   배치 사이즈: 128 (고정)")
    print(f"   총 실험 수: {total_experiments}개")
    print("=" * 80)
    
    # 모든 실험 결과 저장
    all_results = {}
    experiment_count = 0
    
    start_time = time.time()
    
    for dataset_idx, ds_type in enumerate(datasets_to_test):
        dataset_name = dataset_names[ds_type]
        print(f"\n🎯 {dataset_name} 데이터셋 실험 시작...")
        
        # 🔧 데이터셋 변경 시 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        all_results[dataset_name] = {}
        
        for lr_idx, lr in enumerate(learning_rates):
            for eps_idx, eps in enumerate(epsilons):
                combo_key = f"lr_{lr}_eps_{eps:.0e}"
                print(f"\n🔧 LR={lr}, Epsilon={eps:.0e} 실험...")
                
                # 🔧 하이퍼파라미터 조합 변경 시 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                all_results[dataset_name][combo_key] = {}
                
                # 해당 hyperparameter 조합으로 실험 실행
                try:
                    # epsilon 값을 전달하기 위해 custom experiment 실행
                    config = setup_experiment_config(ds_type, epochs, lr)
                    config['batch_size'] = 128  # 고정
                    
                    # 데이터 로더 생성
                    data_loader = get_dataset_loader(
                        dataset_type=ds_type,
                        batch_size=128,
                        num_workers=4 if torch.cuda.is_available() else 2
                    )
                    
                    # 옵티마이저 설정 (epsilon 포함)
                    optimizers_config = create_optimizers_config(lr, config['weight_decay'], eps)
                    optimizers_config = {name: optimizers_config[name] 
                                       for name in optimizers if name in optimizers_config}
                    
                    # 모델 팩토리
                    def model_factory():
                        return create_model(ds_type, config['model_type'])
                    
                    # 스케줄러 팩토리
                    def scheduler_factory(optimizer):
                        return create_standard_scheduler(
                            optimizer, 
                            scheduler_type=config['scheduler_type'], 
                            epochs=config['epochs']
                        )
                    
                    # 실험 실행
                    experiment = OptimizerExperiment(ds_type, config['model_type'], enable_checkpoints=True)
                    experiment_results = experiment.run_comparison_experiment(
                        optimizers_config=optimizers_config,
                        train_loader=data_loader.train_loader,
                        val_loader=data_loader.val_loader,
                        test_loader=data_loader.test_loader,
                        model_factory_fn=model_factory,
                        epochs=config['epochs'],
                        scheduler_factory_fn=scheduler_factory,
                        resume_from_checkpoint=False
                    )
                    
                    # epsilon 값을 옵티마이저 설정에 추가로 적용
                    if experiment_results:
                        # 결과에 hyperparameter 정보 추가
                        for opt_name in experiment_results.keys():
                            experiment_results[opt_name]['hyperparameters'] = {
                                'lr': lr,
                                'eps': eps,
                                'batch_size': 128
                            }
                        
                        all_results[dataset_name][combo_key] = experiment_results
                        experiment_count += len(optimizers)
                        
                        # 🔧 실험 완료 후 즉시 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # 개별 조합 결과 시각화 생성
                        print(f"📊 {combo_key} 조합 결과 시각화 생성 중...")
                        try:
                            from visualizer import ExperimentVisualizer
                            individual_visualizer = ExperimentVisualizer('./final_results/individual_hyperparameters')
                            combo_title = f"{dataset_name} (LR={lr}, Eps={eps:.0e})"
                            individual_viz_files = individual_visualizer.generate_all_visualizations(
                                experiment_results, 
                                combo_title
                            )
                            print(f"   ✅ {combo_key} 시각화 완료")
                        except Exception as viz_error:
                            print(f"   ⚠️ {combo_key} 시각화 실패: {viz_error}")
                        
                        # 진행 상황 출력
                        elapsed_time = time.time() - start_time
                        progress = experiment_count / total_experiments * 100
                        print(f"⏱️  진행률: {progress:.1f}% ({experiment_count}/{total_experiments})")
                        print(f"   경과 시간: {elapsed_time/60:.1f}분")
                        
                        if experiment_count < total_experiments:
                            remaining_time = elapsed_time * (total_experiments - experiment_count) / experiment_count
                            print(f"   예상 남은 시간: {remaining_time/60:.1f}분")
                        else:
                            print(f"🎯 모든 하이퍼파라미터 조합 실험 완료! 종합 시각화 준비 중...")
                    else:
                        print(f"❌ LR={lr}, Epsilon={eps:.0e} 실험 실패")
                        
                except Exception as e:
                    print(f"❌ LR={lr}, Epsilon={eps:.0e} 실험 중 오류: {e}")
                    continue
    
    total_time = time.time() - start_time
    
    # 결과 요약 및 시각화
    print("\n" + "=" * 80)
    print("🎉 모든 Hyperparameter Grid Search 실험 완료!")
    print("=" * 80)
    print(f"총 실험 시간: {total_time/3600:.2f}시간 ({total_time/60:.1f}분)")
    print(f"완료된 실험: {experiment_count}/{total_experiments}개")
    print(f"실험한 조합: {len(learning_rates)}개 LR × {len(epsilons)}개 Epsilon × {len(optimizers)}개 Optimizer")
    if len(datasets_to_test) > 1:
        print(f"테스트한 데이터셋: {len(datasets_to_test)}개 ({', '.join([dataset_names[d] for d in datasets_to_test])})")
    
    if experiment_count > 0:
        # Hyperparameter 전용 시각화 생성
        print(f"\n📊 Hyperparameter Grid Search 종합 시각화 생성 중...")
        
        try:
            from visualizer import HyperparameterVisualizer
            hyperparameter_visualizer = HyperparameterVisualizer('./final_results')
            viz_files = hyperparameter_visualizer.create_hyperparameter_analysis(all_results)
            
            print(f"\n📁 생성된 Hyperparameter 시각화 파일:")
            for viz_type, file_path in viz_files.items():
                print(f"   {viz_type}: {os.path.basename(file_path)}")
            
            print(f"\n✅ Hyperparameter Grid Search 종합 시각화 완료!")
            print(f"   저장 위치: {os.path.abspath('./final_results/hyperparameter_grid')}")
                
        except Exception as e:
            print(f"⚠️  Hyperparameter 시각화 생성 중 오류: {e}")
            print("📊 기본 시각화로 대체합니다...")
            
            # 기본 시각화 (각 하이퍼파라미터 조합별로)
            visualizer = ExperimentVisualizer('./final_results')
            
            for dataset_name, dataset_results in all_results.items():
                for combo_key, combo_results in dataset_results.items():
                    if combo_results:
                        # 하이퍼파라미터 정보를 파일명에 포함
                        lr_eps = combo_key.replace('_', '').replace('lr', 'LR').replace('eps', 'Eps')
                        save_name = f"{dataset_name.lower()}_{lr_eps}_comparison"
                        visualizer.generate_all_visualizations(
                            combo_results, 
                            f"{dataset_name} ({combo_key})"
                        )
            
            print(f"✅ 기본 시각화 완료! 각 하이퍼파라미터 조합별 차트 생성됨")
    else:
        print(f"\n⚠️  완료된 실험이 없어서 시각화를 생성하지 않습니다.")
    
    # 최종 요약 출력
    print(f"\n📈 Hyperparameter 조합별 성능 요약:")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for combo_key, combo_results in dataset_results.items():
            if combo_results:
                print(f"  {combo_key}:")
                
                # 🔧 타입 안전성 검증: combo_results가 딕셔너리인지 확인
                if isinstance(combo_results, dict):
                    for opt_name, opt_result in combo_results.items():
                        if isinstance(opt_result, dict):
                            best_acc = opt_result.get('best_val_acc', 0)
                            print(f"    {opt_name}: {best_acc:.2f}%")
                        else:
                            print(f"    {opt_name}: 결과 없음")
                else:
                    # 실험 실패 메시지 출력 (문자열 상태)
                    print(f"    상태: {combo_results}")
    
    return all_results


def batch_size_comparison_experiment(dataset_type: Optional[int] = None, epochs: Optional[int] = None):
    """
    배치 사이즈별 5개 옵티마이저 비교 실험
    
    Args:
        dataset_type: 특정 데이터셋만 테스트 (None이면 모든 데이터셋)
        epochs: 에포크 수 (None이면 기본값 사용)
    
    Returns:
        dict: 모든 실험 결과
    """
    print("🚀 배치 사이즈별 5개 옵티마이저 비교 실험")
    print("=" * 80)
    
    # 시각화 모드 안내 (실험 시작 시에만)
    set_visualization_mode(show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS, show_message=True)
    
    # 실험 설정
    datasets_to_test = [dataset_type] if dataset_type else [1, 2, 3]
    # 기본 배치 사이즈 (데이터셋별로 동적 조정)
    batch_sizes = [64, 128, 256]
    optimizers = ['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS']
    
    dataset_names = {1: "MNIST", 2: "CIFAR-10", 3: "Tiny ImageNet"}
    
    # 전체 실험 수 계산
    total_experiments = len(datasets_to_test) * len(batch_sizes) * len(optimizers)
    print(f"📋 실험 계획:")
    print(f"   데이터셋: {[dataset_names[d] for d in datasets_to_test]}")
    print(f"   배치 사이즈: {batch_sizes}")
    print(f"   옵티마이저: {optimizers}")
    print(f"   총 실험 수: {total_experiments}개")
    print("=" * 80)
    
    # 모든 실험 결과 저장
    all_results = {}
    experiment_count = 0
    
    start_time = time.time()
    
    for dataset_idx, ds_type in enumerate(datasets_to_test):
        dataset_name = dataset_names[ds_type]
        print(f"\n🎯 {dataset_name} 데이터셋 실험 시작...")
        
        # 🔧 데이터셋 변경 시 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   GPU 메모리 정리 완료")
        gc.collect()
        
        all_results[dataset_name] = {}
        
        # Tiny ImageNet의 경우 배치 사이즈 조정 (메모리 절약)
        if ds_type == 3:  # Tiny ImageNet
            current_batch_sizes = [64, 128, 256]  # 더 작은 배치 사이즈 사용
            print(f"   Tiny ImageNet: 메모리 절약을 위해 배치 사이즈 조정 {current_batch_sizes}")
        else:
            current_batch_sizes = batch_sizes  # [64, 128, 256]
            
        for batch_idx, batch_size in enumerate(current_batch_sizes):
            print(f"\n📦 배치 사이즈 {batch_size} 실험...")
            
            batch_key = f"batch_{batch_size}"
            all_results[dataset_name][batch_key] = {}
            
            # 🔧 배치 사이즈 변경 시 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 해당 배치 사이즈로 실험 실행
            try:
                # 🔧 메모리 사전 체크 (Tiny ImageNet의 경우)
                if torch.cuda.is_available() and ds_type == 3:
                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    reserved_memory = torch.cuda.memory_reserved() / (1024**3)  # GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    free_memory = total_memory - reserved_memory
                    
                    print(f"   GPU 메모리 상태: 전체={total_memory:.1f}GB, 예약={reserved_memory:.1f}GB, 여유={free_memory:.1f}GB")
                    
                    # 메모리 부족 예상 시 건너뛰기
                    if free_memory < 1.5 and batch_size > 64:  # 1.5GB 미만이고 큰 배치일 때
                        print(f"   ⚠️ 메모리 부족 예상 - 배치 사이즈 {batch_size} 건너뜀")
                        all_results[dataset_name][batch_key] = "메모리_부족_예방"
                        continue
                
                experiment_results = run_experiment(
                    dataset_type=ds_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    optimizers_to_test=optimizers,
                    resume_training=False
                )
                
                if experiment_results:
                    all_results[dataset_name][batch_key] = experiment_results
                    experiment_count += len(optimizers)
                    
                    # 🔧 성공 후 즉시 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 진행 상황 출력
                    elapsed_time = time.time() - start_time
                    total_planned = len(datasets_to_test) * len(current_batch_sizes) * len(optimizers)
                    progress = experiment_count / total_planned * 100
                    print(f"⏱️  진행률: {progress:.1f}% ({experiment_count}/{total_planned})")
                    print(f"   경과 시간: {elapsed_time/60:.1f}분")
                    
                    if experiment_count < total_planned:
                        remaining_time = elapsed_time * (total_planned - experiment_count) / experiment_count
                        print(f"   예상 남은 시간: {remaining_time/60:.1f}분")
                else:
                    print(f"❌ 배치 사이즈 {batch_size} 실험 실패 (메모리 부족 가능성)")
                    all_results[dataset_name][batch_key] = "실험_실패"
                    
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    print(f"❌ CUDA 메모리 부족: 배치 {batch_size} - {error_msg[:100]}...")
                    all_results[dataset_name][batch_key] = "CUDA_메모리_부족"
                    
                    # 🔧 강제 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 더 큰 배치 사이즈는 건너뛰기
                    remaining_batches = current_batch_sizes[batch_idx+1:]
                    if remaining_batches:
                        print(f"   남은 큰 배치 사이즈들 ({remaining_batches}) 건너뜀")
                        for skip_batch in remaining_batches:
                            skip_key = f"batch_{skip_batch}"
                            all_results[dataset_name][skip_key] = "메모리_부족_건너뜀"
                        break
                else:
                    print(f"❌ 기타 오류: {error_msg[:100]}...")
                    all_results[dataset_name][batch_key] = f"오류_{error_msg[:50]}"
                continue
            except Exception as e:
                print(f"❌ 예상치 못한 오류: {str(e)[:100]}...")
                all_results[dataset_name][batch_key] = f"예상치못한오류_{str(e)[:50]}"
                continue
    
    total_time = time.time() - start_time
    
    # 결과 요약 및 시각화
    print("\n" + "=" * 80)
    print("🎉 모든 배치 사이즈 실험 완료!")
    print("=" * 80)
    print(f"총 실험 시간: {total_time/3600:.2f}시간 ({total_time/60:.1f}분)")
    print(f"완료된 실험: {experiment_count}/{total_experiments}개")
    
    if experiment_count > 0:
        # 배치 사이즈 전용 시각화 생성
        print(f"\n📊 배치 사이즈 비교 시각화 생성 중...")
        from visualizer import BatchSizeVisualizer
        
        try:
            batch_visualizer = BatchSizeVisualizer('./results')
            viz_files = batch_visualizer.create_batch_size_analysis(all_results)
            
            print(f"\n📁 생성된 시각화 파일:")
            for viz_type, file_path in viz_files.items():
                print(f"   {viz_type}: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"⚠️  시각화 생성 중 오류: {e}")
            print("기본 시각화로 대체합니다...")
            
            # 기본 시각화 (각 데이터셋별로)
            visualizer = ExperimentVisualizer('./results')
            
            for dataset_name, dataset_results in all_results.items():
                for batch_key, batch_results in dataset_results.items():
                    if batch_results:
                        batch_size = batch_key.split('_')[1]
                        save_name = f"{dataset_name.lower()}_batch{batch_size}_comparison"
                        visualizer.generate_all_visualizations(
                            batch_results, 
                            f"{dataset_name} (Batch Size {batch_size})"
                        )
    
    # 최종 요약 출력
    print(f"\n📈 배치 사이즈별 성능 요약:")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for batch_key, batch_results in dataset_results.items():
            if batch_results:
                batch_size = batch_key.split('_')[1]
                print(f"  배치 사이즈 {batch_size}:")
                
                # 🔧 타입 안전성 검증: batch_results가 딕셔너리인지 확인
                if isinstance(batch_results, dict):
                    for opt_name, opt_result in batch_results.items():
                        if isinstance(opt_result, dict):
                            best_acc = opt_result.get('best_val_acc', 0)
                            print(f"    {opt_name}: {best_acc:.2f}%")
                        else:
                            print(f"    {opt_name}: 결과 없음")
                else:
                    # 실험 실패 메시지 출력 (문자열 상태)
                    print(f"    상태: {batch_results}")
    
    return all_results


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
    print("3. 전체 옵티마이저 비교 (5개: RMSProp, RMSPropABS, Adam, AdamW, AdamABS)")
    print("4. 논문용 Tiny ImageNet 실험 (5개 옵티마이저, 40 에포크)")
    print("5. 배치 사이즈별 비교 실험 (5개 옵티마이저, 배치 64/128/256)")
    print("6. 특정 배치 사이즈 실험 (배치 사이즈 선택)")
    print("7. Hyperparameter Grid Search (Adam vs AdamABS, LR & Epsilon 조합)")
    print("8. 사용자 정의 설정")
    
    while True:
        try:
            mode_choice = int(input("모드를 선택하세요 (1-8): "))
            if mode_choice in [1, 2, 3, 4, 5, 6, 7, 8]:
                break
            else:
                print("❌ 1~8 중에서 선택해주세요.")
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
        # 논문용 Tiny ImageNet 실험
        print("\n🚀 논문용 Tiny ImageNet 실험 설정:")
        
        # 배치 사이즈 선택
        print("배치 사이즈를 선택하세요:")
        print("1. 64")
        print("2. 128 (권장)")
        print("3. 256")
        
        while True:
            try:
                batch_choice = int(input("배치 사이즈 선택 (1-3): "))
                if batch_choice in [1, 2, 3]:
                    break
                else:
                    print("❌ 1, 2, 3 중에서 선택해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
        
        batch_sizes = {1: 64, 2: 128, 3: 256}
        selected_batch_size = batch_sizes[batch_choice]
        
        # 에포크 수 설정
        epochs_input = input(f"에포크 수 (기본 40, 엔터로 기본값 사용): ").strip()
        epochs = 40
        if epochs_input:
            try:
                epochs = int(epochs_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값 40을 사용합니다.")
        
        print(f"\n실험 설정:")
        print(f"   데이터셋: Tiny ImageNet")
        print(f"   배치 사이즈: {selected_batch_size}")
        print(f"   에포크: {epochs}")
        print(f"   학습률: 0.0005 (고정)")
        print(f"   Epsilon: 1e-8 (고정)")
        print(f"   옵티마이저: 5개 (RMSProp, RMSPropABS, Adam, AdamW, AdamABS)")
        
        confirm = input("\n실험을 시작하시겠습니까? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes', '예']:
            return tiny_imagenet_paper_experiment(selected_batch_size, epochs)
        else:
            print("❌ 실험이 취소되었습니다.")
            return None
    elif mode_choice == 5:
        # 배치 사이즈 비교 실험
        print("\n🚀 배치 사이즈 비교 실험 설정:")
        
        # 에포크 수 설정 (선택사항)
        epochs = None
        epochs_input = input("에포크 수 (기본값 사용하려면 엔터): ").strip()
        if epochs_input:
            try:
                epochs = int(epochs_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값을 사용합니다.")
        
        # 전체 데이터셋 vs 선택된 데이터셋
        all_datasets_input = input("모든 데이터셋에서 실험하시겠습니까? (Y/n): ").strip().lower()
        target_dataset = None if all_datasets_input in ['', 'y', 'yes', '예'] else dataset_choice
        
        if target_dataset:
            dataset_name = {1: "MNIST", 2: "CIFAR-10", 3: "Tiny ImageNet"}[target_dataset]
            print(f"선택된 데이터셋: {dataset_name}")
        else:
            print("선택된 데이터셋: 모든 데이터셋 (MNIST, CIFAR-10, Tiny ImageNet)")
        
        print("배치 사이즈: 64, 128, 256")
        print("옵티마이저: 5개 (RMSProp, RMSPropABS, Adam, AdamW, AdamABS)")
        
        confirm = input("\n실험을 시작하시겠습니까? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes', '예']:
            return batch_size_comparison_experiment(target_dataset, epochs)
        else:
            print("❌ 실험이 취소되었습니다.")
            return None
    elif mode_choice == 6:
        # 특정 배치 사이즈 실험
        print("\n🚀 특정 배치 사이즈 실험 설정:")
        
        # 배치 사이즈 선택
        print("배치 사이즈를 선택하세요:")
        print("1. 64")
        print("2. 128")
        print("3. 256")
        
        while True:
            try:
                batch_choice = int(input("배치 사이즈 선택 (1-3): "))
                if batch_choice in [1, 2, 3]:
                    break
                else:
                    print("❌ 1, 2, 3 중에서 선택해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
        
        batch_sizes_map = {1: 64, 2: 128, 3: 256}
        selected_batch = batch_sizes_map[batch_choice]
        
        # 에포크 수 설정
        epochs = None
        epochs_input = input("에포크 수 (기본값 사용하려면 엔터): ").strip()
        if epochs_input:
            try:
                epochs = int(epochs_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값을 사용합니다.")
        
        print(f"\n실험 설정:")
        print(f"   데이터셋: {['MNIST', 'CIFAR-10', 'Tiny ImageNet'][dataset_choice-1]}")
        print(f"   배치 사이즈: {selected_batch}")
        print(f"   옵티마이저: 5개 (RMSProp, RMSPropABS, Adam, AdamW, AdamABS)")
        
        confirm = input("\n실험을 시작하시겠습니까? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes', '예']:
            return run_experiment(dataset_choice, epochs=epochs, batch_size=selected_batch, resume_training=resume_training)
        else:
            print("❌ 실험이 취소되었습니다.")
            return None
    elif mode_choice == 7:
        # Hyperparameter Grid Search 실험
        print("\n🚀 Hyperparameter Grid Search 실험 설정:")
        
        # 에포크 수 설정 (선택사항)
        epochs = None
        epochs_input = input("에포크 수 (기본값 사용하려면 엔터): ").strip()
        if epochs_input:
            try:
                epochs = int(epochs_input)
            except ValueError:
                print("❌ 잘못된 입력. 기본값을 사용합니다.")
        
        # 전체 데이터셋 vs 선택된 데이터셋
        all_datasets_input = input("모든 데이터셋에서 실험하시겠습니까? (Y/n): ").strip().lower()
        target_dataset = None if all_datasets_input in ['', 'y', 'yes', '예'] else dataset_choice
        
        if target_dataset:
            dataset_name = {1: "MNIST", 2: "CIFAR-10", 3: "Tiny ImageNet"}[target_dataset]
            print(f"선택된 데이터셋: {dataset_name}")
        else:
            print("선택된 데이터셋: 모든 데이터셋 (MNIST, CIFAR-10, Tiny ImageNet)")
        
        print("Learning Rates: [0.0005, 0.001, 0.002]")
        print("Epsilon Values: [1e-9, 1e-8, 1e-7]")
        print("배치 사이즈: 128 (고정)")
        print("옵티마이저: Adam, AdamABS")
        
        confirm = input("\n실험을 시작하시겠습니까? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes', '예']:
            return hyperparameter_grid_search_experiment(target_dataset, epochs)
        else:
            print("❌ 실험이 취소되었습니다.")
            return None
    elif mode_choice == 8:
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
  python main_experiment.py --dataset 3 --epochs 15 --lr 0.001
  python main_experiment.py --batch-size-comparison --dataset 1 # MNIST 배치 사이즈 비교
  python main_experiment.py --batch-size-comparison # 모든 데이터셋 배치 사이즈 비교
  python main_experiment.py --hyperparameter-grid-search --dataset 1 # MNIST hyperparameter 그리드 서치
  python main_experiment.py --hyperparameter-grid-search # 모든 데이터셋 hyperparameter 그리드 서치
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
                       choices=['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS'],
                       help='테스트할 옵티마이저 (기본: 모두)')
    parser.add_argument('--quick', action='store_true',
                       help='빠른 테스트 모드 (3 에포크)')
    parser.add_argument('--adam-vs-adamabs', action='store_true',
                       help='Adam vs AdamABS만 비교')
    parser.add_argument('--batch-size-comparison', action='store_true',
                       help='배치 사이즈별 5개 옵티마이저 비교 (64, 128, 256)')
    parser.add_argument('--batch-size-option', type=int, choices=[64, 128, 256],
                       help='특정 배치 사이즈로만 실험 (64, 128, 256 중 선택)')
    parser.add_argument('--hyperparameter-grid-search', action='store_true',
                       help='Learning rate와 epsilon 조합별 Adam vs AdamABS 비교')
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
    
    # 특정 배치 사이즈 옵션
    if args.batch_size_option:
        if args.dataset is None:
            print("❌ --batch-size-option 사용 시 --dataset 옵션이 필요합니다.")
            return None
        return run_experiment(
            dataset_type=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size_option,
            resume_training=args.resume
        )
    
    # 배치 사이즈 비교 실험은 데이터셋 선택이 선택사항
    if args.batch_size_comparison:
        return batch_size_comparison_experiment(args.dataset, args.epochs)
    
    # Hyperparameter Grid Search 실험도 데이터셋 선택이 선택사항
    if args.hyperparameter_grid_search:
        return hyperparameter_grid_search_experiment(args.dataset, args.epochs)
    
    # 다른 실험들은 데이터셋이 필수
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