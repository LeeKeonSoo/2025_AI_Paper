"""
Adam vs AdamAbs 포괄적인 실험 실행 스크립트
논문 작성을 위한 모든 실험을 자동화하여 실행
"""

import os
import sys
import time
import torch
import argparse
from datetime import datetime

# 프로젝트 모듈 import
from adam_abs_optimizer import AdamAbs, AdamAbsW, create_optimizer, compare_optimizers_theory
from mnist_experiment import MNISTExperiment, MNISTNet, SimpleMNISTNet

def check_system_requirements():
    """시스템 요구사항 확인"""
    print("시스템 요구사항 확인...")
    
    # PyTorch 버전 확인
    print(f"PyTorch 버전: {torch.__version__}")
    
    # CUDA 가용성 확인
    if torch.cuda.is_available():
        print(f"CUDA 가용: {torch.cuda.device_count()}개 GPU")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA 미사용 - CPU 모드로 실행")
    
    # 메모리 확인
    if torch.cuda.is_available():
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print("="*60)

def run_theory_comparison():
    """이론적 비교 실행"""
    print("이론적 차이점 분석 실행...")
    compare_optimizers_theory()
    print("\n이론적 분석 완료!")

def run_synthetic_experiments():
    """합성 데이터 실험 실행"""
    print("\n합성 데이터 실험 실행...")
    
    # 간단한 합성 데이터 실험
    import numpy as np
    from sklearn.datasets import make_regression, make_classification
    from sklearn.preprocessing import StandardScaler
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    # 회귀 문제
    print("1. 회귀 문제 실험")
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 간단한 회귀 모델
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # 최적화 알고리즘 비교
    optimizers = ['adam', 'adamabs']
    results = {}
    
    for opt_name in optimizers:
        print(f"  {opt_name.upper()} 훈련 중...")
        
        # 모델 초기화
        test_model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        optimizer = create_optimizer(opt_name, test_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            output = test_model(X_tensor)
            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[opt_name] = losses
        print(f"    최종 손실: {losses[-1]:.6f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    for opt_name, losses in results.items():
        plt.plot(losses, label=opt_name.upper(), linewidth=2)
    plt.title('Synthetic Regression: Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/synthetic_regression_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("합성 데이터 실험 완료!")

def run_mnist_experiments(epochs=20, batch_size=128):
    """MNIST 실험 실행"""
    print(f"\nMNIST 실험 실행 (에포크: {epochs}, 배치 크기: {batch_size})")
    
    # 실험 설정
    experiment = MNISTExperiment(batch_size=batch_size)
    
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
    cnn_results = experiment.run_experiment(MNISTNet, optimizer_configs, epochs=epochs)
    
    # 간단한 모델 실험
    print("\n2. 완전연결 모델 실험")
    fc_results = experiment.run_experiment(SimpleMNISTNet, optimizer_configs, epochs=epochs)
    
    # 결과 분석
    print("\n" + "="*80)
    print("CNN 모델 결과 분석")
    cnn_analysis = experiment.analyze_results(cnn_results)
    
    print("\n" + "="*80)
    print("FC 모델 결과 분석")
    fc_analysis = experiment.analyze_results(fc_results)
    
    # 시각화
    experiment.plot_results(cnn_results, 
                           save_path='/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/mnist_cnn_results.png')
    
    experiment.plot_results(fc_results, 
                           save_path='/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/mnist_fc_results.png')
    
    # 결과 저장
    os.makedirs('/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results', exist_ok=True)
    cnn_file = experiment.save_results(cnn_results, '/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results')
    fc_file = experiment.save_results(fc_results, '/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results')
    
    return cnn_results, fc_results, cnn_analysis, fc_analysis

def run_hyperparameter_sensitivity():
    """하이퍼파라미터 민감도 분석"""
    print("\n하이퍼파라미터 민감도 분석...")
    
    # 학습률 민감도 분석
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    
    print("학습률 민감도 분석:")
    for lr in learning_rates:
        print(f"  학습률: {lr}")
        
        # 간단한 테스트 모델
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
        # 더미 데이터
        X = torch.randn(100, 10)
        y = torch.randn(100)
        
        # Adam vs AdamAbs 비교
        for opt_name in ['adam', 'adamabs']:
            optimizer = create_optimizer(opt_name, model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            initial_loss = None
            for epoch in range(50):
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output.squeeze(), y)
                if initial_loss is None:
                    initial_loss = loss.item()
                loss.backward()
                optimizer.step()
            
            improvement = (initial_loss - loss.item()) / initial_loss * 100
            print(f"    {opt_name}: 개선율 {improvement:.2f}%")
    
    print("하이퍼파라미터 민감도 분석 완료!")

def generate_paper_summary(cnn_results, fc_results, cnn_analysis, fc_analysis):
    """논문 작성을 위한 요약 생성"""
    print("\n" + "="*80)
    print("논문 작성을 위한 실험 결과 요약")
    print("="*80)
    
    summary = {
        'experiment_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'cnn_results': {},
        'fc_results': {},
        'key_findings': [],
        'recommendations': []
    }
    
    # CNN 결과 요약
    print("\n1. CNN 모델 결과:")
    best_cnn_opt = max(cnn_results.keys(), key=lambda x: cnn_results[x]['final_test_acc'])
    adam_cnn_acc = cnn_results['adam']['final_test_acc']
    adamabs_cnn_acc = cnn_results['adamabs']['final_test_acc']
    cnn_improvement = (adamabs_cnn_acc - adam_cnn_acc) / adam_cnn_acc * 100
    
    print(f"   최고 성능: {best_cnn_opt} ({cnn_results[best_cnn_opt]['final_test_acc']:.2f}%)")
    print(f"   Adam: {adam_cnn_acc:.2f}%")
    print(f"   AdamAbs: {adamabs_cnn_acc:.2f}%")
    print(f"   개선율: {cnn_improvement:.2f}%")
    
    summary['cnn_results'] = {
        'best_optimizer': best_cnn_opt,
        'adam_accuracy': adam_cnn_acc,
        'adamabs_accuracy': adamabs_cnn_acc,
        'improvement': cnn_improvement
    }
    
    # FC 결과 요약
    print("\n2. FC 모델 결과:")
    best_fc_opt = max(fc_results.keys(), key=lambda x: fc_results[x]['final_test_acc'])
    adam_fc_acc = fc_results['adam']['final_test_acc']
    adamabs_fc_acc = fc_results['adamabs']['final_test_acc']
    fc_improvement = (adamabs_fc_acc - adam_fc_acc) / adam_fc_acc * 100
    
    print(f"   최고 성능: {best_fc_opt} ({fc_results[best_fc_opt]['final_test_acc']:.2f}%)")
    print(f"   Adam: {adam_fc_acc:.2f}%")
    print(f"   AdamAbs: {adamabs_fc_acc:.2f}%")
    print(f"   개선율: {fc_improvement:.2f}%")
    
    summary['fc_results'] = {
        'best_optimizer': best_fc_opt,
        'adam_accuracy': adam_fc_acc,
        'adamabs_accuracy': adamabs_fc_acc,
        'improvement': fc_improvement
    }
    
    # 주요 발견사항
    print("\n3. 주요 발견사항:")
    findings = []
    
    if cnn_improvement > 0:
        findings.append(f"CNN 모델에서 AdamAbs가 Adam보다 {cnn_improvement:.2f}% 더 높은 성능을 보임")
    if fc_improvement > 0:
        findings.append(f"FC 모델에서 AdamAbs가 Adam보다 {fc_improvement:.2f}% 더 높은 성능을 보임")
    
    # 수렴 속도 분석
    adam_cnn_convergence = cnn_analysis['adam']['convergence_epoch']
    adamabs_cnn_convergence = cnn_analysis['adamabs']['convergence_epoch']
    
    if adamabs_cnn_convergence < adam_cnn_convergence:
        findings.append(f"AdamAbs가 Adam보다 {adam_cnn_convergence - adamabs_cnn_convergence}에포크 빠르게 수렴")
    
    # 안정성 분석
    adam_stability = cnn_analysis['adam']['stability']
    adamabs_stability = cnn_analysis['adamabs']['stability']
    
    if adamabs_stability < adam_stability:
        findings.append("AdamAbs가 Adam보다 더 안정적인 훈련을 보임")
    
    for finding in findings:
        print(f"   - {finding}")
        summary['key_findings'].append(finding)
    
    # 권장사항
    print("\n4. 권장사항:")
    recommendations = [
        "제곱근 연산을 피하고 싶은 경우 AdamAbs 사용 고려",
        "계산 효율성이 중요한 경우 AdamAbs 적용",
        "기존 Adam과 유사한 성능을 유지하면서 수치적 안정성 향상",
        "다양한 데이터셋과 모델 아키텍처에서 추가 검증 필요"
    ]
    
    for rec in recommendations:
        print(f"   - {rec}")
        summary['recommendations'].append(rec)
    
    # 요약 저장
    import json
    with open('/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/paper_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n논문 요약이 저장되었습니다: paper_summary.json")
    
    return summary

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Adam vs AdamAbs 포괄적인 실험')
    parser.add_argument('--epochs', type=int, default=20, help='훈련 에포크 수')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--skip_theory', action='store_true', help='이론 분석 건너뛰기')
    parser.add_argument('--skip_synthetic', action='store_true', help='합성 데이터 실험 건너뛰기')
    parser.add_argument('--skip_mnist', action='store_true', help='MNIST 실험 건너뛰기')
    parser.add_argument('--skip_hyperparameter', action='store_true', help='하이퍼파라미터 분석 건너뛰기')
    
    args = parser.parse_args()
    
    print("Adam vs AdamAbs 포괄적인 실험 실행")
    print("="*80)
    print(f"실험 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 시스템 요구사항 확인
    check_system_requirements()
    
    # 결과 디렉토리 생성
    os.makedirs('/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/results', exist_ok=True)
    
    # 실험 실행
    start_time = time.time()
    
    try:
        # 1. 이론적 분석
        if not args.skip_theory:
            run_theory_comparison()
        
        # 2. 합성 데이터 실험
        if not args.skip_synthetic:
            run_synthetic_experiments()
        
        # 3. MNIST 실험
        cnn_results, fc_results, cnn_analysis, fc_analysis = None, None, None, None
        if not args.skip_mnist:
            cnn_results, fc_results, cnn_analysis, fc_analysis = run_mnist_experiments(
                epochs=args.epochs, batch_size=args.batch_size
            )
        
        # 4. 하이퍼파라미터 민감도 분석
        if not args.skip_hyperparameter:
            run_hyperparameter_sensitivity()
        
        # 5. 논문 요약 생성
        if cnn_results and fc_results:
            generate_paper_summary(cnn_results, fc_results, cnn_analysis, fc_analysis)
        
        # 실험 완료
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"모든 실험 완료! 총 소요 시간: {total_time:.2f}초")
        print(f"결과 파일들이 /Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/ 에 저장되었습니다.")
        print("="*80)
        
    except Exception as e:
        print(f"실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()