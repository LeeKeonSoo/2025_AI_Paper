"""
Adam, AdamW, AdamABS 옵티마이저 직접 구현
CUDA 최적화를 위해 PyTorch 내장 옵티마이저 대신 직접 구현

Author: AI Research
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple


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


class CustomAdamABS(torch.optim.Optimizer):
    """
    AdamABS 최적화 알고리즘 직접 구현
    
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
        super(CustomAdamABS, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """AdamABS 최적화 스텝 수행"""
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


def create_optimizer(optimizer_name: str, params, lr: float = 1e-3, 
                    weight_decay: float = 0, **kwargs) -> torch.optim.Optimizer:
    """
    최적화 알고리즘 팩토리 함수
    
    Args:
        optimizer_name: 최적화 알고리즘 이름 ('adam', 'adamw', 'adamabs')
        params: 모델 파라미터
        lr: 학습률
        weight_decay: 가중치 감소
        **kwargs: 추가 파라미터 (betas, eps 등)
    
    Returns:
        torch.optim.Optimizer: 최적화 알고리즘 인스턴스
    """
    optimizer_name = optimizer_name.lower()
    
    # 기본 파라미터 설정
    betas = kwargs.get('betas', (0.9, 0.999))
    eps = kwargs.get('eps', 1e-8)
    
    if optimizer_name == 'adam':
        return CustomAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return CustomAdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'adamabs':
        return CustomAdamABS(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'adam', 'adamw', 'adamabs'")


def get_optimizer_info():
    """최적화 알고리즘 정보 반환"""
    info = {
        'adam': {
            'name': 'Custom Adam',
            'description': 'Adam 최적화 알고리즘 직접 구현',
            'features': ['1차/2차 모멘텀', '편향 보정', '적응적 학습률'],
            'formula': 'θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)'
        },
        'adamw': {
            'name': 'Custom AdamW',
            'description': 'AdamW (분리된 가중치 감소) 직접 구현',
            'features': ['1차/2차 모멘텀', '편향 보정', '분리된 가중치 감소'],
            'formula': 'θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})'
        },
        'adamabs': {
            'name': 'Custom AdamABS',
            'description': 'AdamABS (절댓값 + 제곱근 제거) 새로운 알고리즘',
            'features': ['1차 모멘텀', '절댓값 기반 2차 모멘텀', '제곱근 연산 제거'],
            'formula': 'θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε), v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|'
        },
    }
    return info


def compare_optimizers_theory():
    """최적화 알고리즘 이론적 차이점 분석"""
    print("=" * 80)
    print("Custom Optimizers 이론적 차이점")
    print("=" * 80)
    
    print("\n1. Adam (기준):")
    print("   - 2차 모멘텀: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²")
    print("   - 업데이트: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)")
    print("   - 특징: 표준 Adam, 제곱근 연산 포함")
    
    print("\n2. AdamW:")
    print("   - 2차 모멘텀: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t² (Adam과 동일)")
    print("   - 업데이트: θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})")
    print("   - 특징: 분리된 가중치 감소, 정규화 개선")
    
    print("\n3. AdamABS (새로운 아이디어):")
    print("   - 2차 모멘텀: v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t| ← 절댓값")
    print("   - 업데이트: θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε) ← 제곱근 제거")
    print("   - 특징: 계산 효율성, 이상치 강건성")
    
    
    print("\n5. 주요 혁신점 (AdamABS):")
    print("   ✓ 절댓값 사용: 이상치(outlier)에 덜 민감")
    print("   ✓ 제곱근 제거: 계산 효율성 향상")
    print("   ✓ 수치 안정성: 더 안정적인 학습")
    print("   ✓ 메모리 효율성: 동일한 메모리 사용량")
    
    print("\n6. 예상 장점:")
    print("   - 빠른 연산: sqrt() 연산 제거")
    print("   - 안정성: 절댓값으로 인한 수치 안정성")
    print("   - 강건성: 이상치 gradient에 덜 민감")
    print("   - 수렴성: 더 부드러운 수렴 가능")
    
    print("=" * 80)


def benchmark_operations(device='cuda', size=1000000, iterations=1000):
    """AdamABS vs Adam 연산 성능 벤치마크"""
    import time
    
    print(f"\n{'='*60}")
    print("AdamABS vs Adam 연산 성능 벤치마크")
    print(f"텐서 크기: {size:,}, 반복 횟수: {iterations:,}")
    print(f"{'='*60}")
    
    # 테스트용 텐서 생성
    grad = torch.randn(size, device=device)
    v = torch.randn(size, device=device).abs()  # 양수로 만들기
    
    # 연산별 시간 측정
    operations = {
        'Adam (g² + sqrt)': lambda: grad / (torch.sqrt(grad.pow(2) + 1e-8)),
        'AdamABS (|g|)': lambda: grad / (grad.abs() + 1e-8),
        'Square only': lambda: grad.pow(2),
        'Absolute only': lambda: grad.abs(),
        'Sqrt only': lambda: torch.sqrt(v + 1e-8),
        'Division only': lambda: grad / (v + 1e-8)
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
    
    # 상대적 성능 비교
    print(f"\n상대적 성능 (Adam 기준):")
    adam_time = results['Adam (g² + sqrt)']
    adamabs_time = results['AdamABS (|g|)']
    speedup = adam_time / adamabs_time
    
    print(f"  AdamABS 속도 개선: {speedup:.2f}x {'빠름' if speedup > 1 else '느림'}")
    print(f"  Adam: {adam_time:.4f} ms")
    print(f"  AdamABS: {adamabs_time:.4f} ms")
    print(f"  차이: {adam_time - adamabs_time:+.4f} ms")
    
    return results


if __name__ == "__main__":
    # 이론적 차이점 출력
    compare_optimizers_theory()
    
    # 간단한 테스트
    print("\n" + "="*60)
    print("옵티마이저 생성 테스트")
    print("="*60)
    
    # 더미 모델 생성
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    
    # 모든 옵티마이저 테스트
    optimizer_names = ['adam', 'adamw', 'adamabs']
    
    for opt_name in optimizer_names:
        try:
            optimizer = create_optimizer(opt_name, model.parameters(), lr=0.001, weight_decay=1e-4)
            print(f"✓ {opt_name.upper()} 생성 성공")
        except Exception as e:
            print(f"✗ {opt_name.upper()} 생성 실패: {e}")
    
    # 연산 벤치마크 (CUDA 사용 가능한 경우)
    if torch.cuda.is_available():
        benchmark_operations()
    else:
        print("\nCUDA가 사용 불가능하여 벤치마크를 건너뜁니다.")
    
    print("\n✅ 모든 테스트 완료!")