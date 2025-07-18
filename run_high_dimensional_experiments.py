"""
고차원 데이터셋 실험 실행 스크립트
Adam, AdamW, AdamAbs 비교를 위한 포괄적인 실험 자동화
"""

import os
import sys
import time
import torch
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 모듈 import
from adam_abs_optimizer import AdamAbs, create_optimizer
from high_dimensional_experiments import HighDimensionalExperiment, ResNet18, SimpleViT


def check_gpu_memory():
    """GPU 메모리 확인 및 CUDA 환경 설정"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        free_memory = total_memory - allocated_memory
        
        print(f"CUDA 환경 정보:")
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  총 메모리: {total_memory:.2f}GB")
        print(f"  사용 중: {allocated_memory:.2f}GB")
        print(f"  사용 가능: {free_memory:.2f}GB")
        
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Mixed precision 지원 확인
        if hasattr(torch.cuda, 'amp'):
            print(f"  Mixed Precision (AMP): 지원됨")
        else:
            print(f"  Mixed Precision (AMP): 지원되지 않음")
        
        # 메모리 최적화
        torch.cuda.empty_cache()
        
        if free_memory < 4.0:
            print("⚠️  경고: GPU 메모리가 부족할 수 있습니다. 배치 크기를 줄이는 것을 고려하세요.")
        
        return free_memory
    else:
        print("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return 0


def run_single_experiment(experiment_name, model_factory, train_loader, val_loader, test_loader, 
                         epochs=50, lr=0.001, weight_decay=1e-4, save_dir=None):
    """단일 실험 실행"""
    print(f"\n{'='*80}")
    print(f"실험 시작: {experiment_name}")
    print(f"{'='*80}")
    
    experiment = HighDimensionalExperiment()
    
    # 실험 실행
    results = experiment.run_experiment(
        model_factory, train_loader, val_loader, test_loader,
        experiment_name, epochs=epochs, lr=lr, weight_decay=weight_decay
    )
    
    # 결과 분석
    experiment.analyze_results(results, experiment_name)
    
    # 시각화
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_name.replace(' ', '_').replace('+', '_')}_results.png")
        experiment.plot_comprehensive_results(results, experiment_name, save_path)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f"{experiment_name.replace(' ', '_')}_{timestamp}.json")
        
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
        
        print(f"결과 저장 완료: {results_file}")
    
    return results


def run_cifar10_resnet_experiment(save_dir=None, epochs=50, batch_size=128):
    """CIFAR-10 + ResNet18 실험"""
    print("CIFAR-10 + ResNet18 실험 준비 중...")
    
    experiment = HighDimensionalExperiment()
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders(
        'cifar10', batch_size=batch_size, augmentation=True
    )
    
    def resnet18_factory():
        return ResNet18(num_classes=num_classes)
    
    return run_single_experiment(
        "CIFAR-10 + ResNet18", resnet18_factory, train_loader, val_loader, test_loader,
        epochs=epochs, lr=0.001, weight_decay=5e-4, save_dir=save_dir
    )


def run_cifar100_resnet_experiment(save_dir=None, epochs=50, batch_size=128):
    """CIFAR-100 + ResNet18 실험"""
    print("CIFAR-100 + ResNet18 실험 준비 중...")
    
    experiment = HighDimensionalExperiment()
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders(
        'cifar100', batch_size=batch_size, augmentation=True
    )
    
    def resnet18_factory():
        return ResNet18(num_classes=num_classes)
    
    return run_single_experiment(
        "CIFAR-100 + ResNet18", resnet18_factory, train_loader, val_loader, test_loader,
        epochs=epochs, lr=0.001, weight_decay=5e-4, save_dir=save_dir
    )


def run_synthetic_mlp_experiment(save_dir=None, epochs=30, batch_size=128):
    """고차원 합성 데이터 + 깊은 MLP 실험"""
    print("고차원 합성 데이터 + 깊은 MLP 실험 준비 중...")
    
    experiment = HighDimensionalExperiment()
    train_loader, val_loader, test_loader, n_features, n_classes = experiment.create_high_dim_synthetic_loader(
        n_samples=10000, n_features=2000, n_classes=20, batch_size=batch_size
    )
    
    def deep_mlp_factory():
        return experiment.create_deep_mlp(n_features, [1024, 512, 256, 128], n_classes, dropout=0.3)
    
    return run_single_experiment(
        "고차원 합성 데이터 + 깊은 MLP", deep_mlp_factory, train_loader, val_loader, test_loader,
        epochs=epochs, lr=0.001, weight_decay=1e-4, save_dir=save_dir
    )


def run_vit_experiment(save_dir=None, epochs=30, batch_size=64):
    """CIFAR-10 + Vision Transformer 실험"""
    print("CIFAR-10 + Vision Transformer 실험 준비 중...")
    
    experiment = HighDimensionalExperiment()
    train_loader, val_loader, test_loader, num_classes = experiment.create_cifar_data_loaders(
        'cifar10', batch_size=batch_size, augmentation=True
    )
    
    def vit_factory():
        return SimpleViT(image_size=32, patch_size=4, num_classes=num_classes, 
                        embed_dim=256, num_heads=8, num_layers=6, mlp_dim=512)
    
    return run_single_experiment(
        "CIFAR-10 + Vision Transformer", vit_factory, train_loader, val_loader, test_loader,
        epochs=epochs, lr=0.0005, weight_decay=1e-4, save_dir=save_dir
    )


def run_quick_test(save_dir=None):
    """빠른 테스트 실험 (모든 모델을 짧은 에포크로 실행)"""
    print("빠른 테스트 실험 실행 중...")
    
    experiments = [
        ("CIFAR-10 + ResNet18", run_cifar10_resnet_experiment, {"epochs": 5, "batch_size": 64}),
        ("고차원 합성 데이터 + 깊은 MLP", run_synthetic_mlp_experiment, {"epochs": 10, "batch_size": 64}),
    ]
    
    results = {}
    for name, func, kwargs in experiments:
        print(f"\n빠른 테스트: {name}")
        results[name] = func(save_dir=save_dir, **kwargs)
    
    return results


def compare_all_experiments(all_results, save_dir=None):
    """모든 실험 결과 비교"""
    print(f"\n{'='*80}")
    print("모든 실험 결과 종합 비교")
    print(f"{'='*80}")
    
    # 실험별 최고 성능 정리
    summary = {}
    for exp_name, results in all_results.items():
        summary[exp_name] = {}
        for opt_name, result in results.items():
            summary[exp_name][opt_name] = {
                'test_acc': result['final_test_acc'],
                'val_acc': result['best_val_acc'],
                'time': result['total_time'],
                'params': result['total_params']
            }
    
    # 최고 성능 실험 찾기
    print("\n1. 실험별 최고 성능:")
    for exp_name, exp_results in summary.items():
        best_opt = max(exp_results.keys(), key=lambda x: exp_results[x]['test_acc'])
        best_acc = exp_results[best_opt]['test_acc']
        print(f"   {exp_name:30s}: {best_opt:8s} ({best_acc:.2f}%)")
    
    # AdamAbs vs Adam 비교
    print(f"\n2. AdamAbs vs Adam 성능 비교:")
    for exp_name, exp_results in summary.items():
        if 'Adam' in exp_results and 'AdamAbs' in exp_results:
            adam_acc = exp_results['Adam']['test_acc']
            adamabs_acc = exp_results['AdamAbs']['test_acc']
            improvement = (adamabs_acc - adam_acc) / adam_acc * 100
            
            adam_time = exp_results['Adam']['time']
            adamabs_time = exp_results['AdamAbs']['time']
            time_diff = (adamabs_time - adam_time) / adam_time * 100
            
            print(f"   {exp_name:30s}: 정확도 {improvement:+6.2f}%, 시간 {time_diff:+6.2f}%")
    
    # 종합 시각화
    if save_dir:
        create_comprehensive_comparison_plot(summary, save_dir)
    
    # 종합 결과 저장
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(save_dir, f"comprehensive_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n종합 결과 저장: {summary_file}")
    
    return summary


def create_comprehensive_comparison_plot(summary, save_dir):
    """종합 비교 플롯 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    exp_names = list(summary.keys())
    optimizers = ['Adam', 'AdamW', 'AdamAbs']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. 테스트 정확도 비교
    ax = axes[0, 0]
    x = np.arange(len(exp_names))
    width = 0.25
    
    for i, opt in enumerate(optimizers):
        accs = []
        for exp_name in exp_names:
            if opt in summary[exp_name]:
                accs.append(summary[exp_name][opt]['test_acc'])
            else:
                accs.append(0)
        
        bars = ax.bar(x + i*width, accs, width, label=opt, color=colors[i], alpha=0.8)
        
        # 값 표시
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Test Accuracy Comparison Across Experiments')
    ax.set_xlabel('Experiments')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([name.replace(' + ', '\\n') for name in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 훈련 시간 비교
    ax = axes[0, 1]
    
    for i, opt in enumerate(optimizers):
        times = []
        for exp_name in exp_names:
            if opt in summary[exp_name]:
                times.append(summary[exp_name][opt]['time'])
            else:
                times.append(0)
        
        bars = ax.bar(x + i*width, times, width, label=opt, color=colors[i], alpha=0.8)
        
        # 값 표시
        for bar, time_val in zip(bars, times):
            if time_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{time_val:.0f}s', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Training Time Comparison Across Experiments')
    ax.set_xlabel('Experiments')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([name.replace(' + ', '\\n') for name in exp_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. AdamAbs vs Adam 개선율
    ax = axes[1, 0]
    improvements = []
    exp_labels = []
    
    for exp_name in exp_names:
        if 'Adam' in summary[exp_name] and 'AdamAbs' in summary[exp_name]:
            adam_acc = summary[exp_name]['Adam']['test_acc']
            adamabs_acc = summary[exp_name]['AdamAbs']['test_acc']
            improvement = (adamabs_acc - adam_acc) / adam_acc * 100
            improvements.append(improvement)
            exp_labels.append(exp_name.replace(' + ', '\\n'))
    
    bars = ax.bar(exp_labels, improvements, color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
    
    # 값 표시
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if imp > 0 else -0.3),
               f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    ax.set_title('AdamAbs vs Adam: Performance Improvement')
    ax.set_xlabel('Experiments')
    ax.set_ylabel('Improvement (%)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. 모델 복잡도 vs 성능
    ax = axes[1, 1]
    
    for i, opt in enumerate(optimizers):
        params = []
        accs = []
        for exp_name in exp_names:
            if opt in summary[exp_name]:
                params.append(summary[exp_name][opt]['params'] / 1e6)  # 백만 단위
                accs.append(summary[exp_name][opt]['test_acc'])
        
        if params and accs:
            ax.scatter(params, accs, label=opt, color=colors[i], alpha=0.7, s=100)
    
    ax.set_title('Model Complexity vs Performance')
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='고차원 데이터셋 실험 실행')
    parser.add_argument('--experiment', type=str, default='all', 
                       choices=['all', 'cifar10', 'cifar100', 'synthetic', 'vit', 'quick'],
                       help='실행할 실험 선택')
    parser.add_argument('--epochs', type=int, default=50, help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--save_dir', type=str, default='/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--no_cuda', action='store_true', help='CUDA 사용 안 함')
    
    args = parser.parse_args()
    
    # CUDA 설정
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("고차원 데이터셋을 활용한 Adam, AdamW, AdamAbs 비교 실험")
    print("="*80)
    print(f"실험 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"실험 종류: {args.experiment}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"결과 저장 경로: {args.save_dir}")
    
    # GPU 메모리 확인
    free_memory = check_gpu_memory()
    
    # 배치 크기 자동 조정 (GPU 메모리 기반)
    if torch.cuda.is_available():
        if free_memory < 4.0 and args.batch_size > 32:
            print("⚠️  GPU 메모리 부족으로 배치 크기를 32로 조정합니다.")
            args.batch_size = 32
        elif free_memory < 6.0 and args.batch_size > 64:
            print("⚠️  GPU 메모리 부족으로 배치 크기를 64로 조정합니다.")
            args.batch_size = 64
        elif free_memory < 8.0 and args.batch_size > 128:
            print("⚠️  GPU 메모리 부족으로 배치 크기를 128로 조정합니다.")
            args.batch_size = 128
    
    # 결과 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 실험 실행
    start_time = time.time()
    all_results = {}
    
    try:
        if args.experiment == 'all':
            print("\n모든 실험 실행...")
            all_results['CIFAR-10 + ResNet18'] = run_cifar10_resnet_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            all_results['CIFAR-100 + ResNet18'] = run_cifar100_resnet_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            all_results['고차원 합성 데이터 + 깊은 MLP'] = run_synthetic_mlp_experiment(
                args.save_dir, args.epochs // 2, args.batch_size
            )
            all_results['CIFAR-10 + Vision Transformer'] = run_vit_experiment(
                args.save_dir, args.epochs, args.batch_size // 2
            )
            
        elif args.experiment == 'cifar10':
            all_results['CIFAR-10 + ResNet18'] = run_cifar10_resnet_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            
        elif args.experiment == 'cifar100':
            all_results['CIFAR-100 + ResNet18'] = run_cifar100_resnet_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            
        elif args.experiment == 'synthetic':
            all_results['고차원 합성 데이터 + 깊은 MLP'] = run_synthetic_mlp_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            
        elif args.experiment == 'vit':
            all_results['CIFAR-10 + Vision Transformer'] = run_vit_experiment(
                args.save_dir, args.epochs, args.batch_size
            )
            
        elif args.experiment == 'quick':
            all_results = run_quick_test(args.save_dir)
        
        # 종합 비교 (2개 이상의 실험이 있을 때)
        if len(all_results) > 1:
            compare_all_experiments(all_results, args.save_dir)
        
        # 실험 완료
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"모든 실험 완료!")
        print(f"총 소요 시간: {total_time:.2f}초 ({total_time/3600:.2f}시간)")
        print(f"결과 저장 경로: {args.save_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()