"""
AdamAbs: Adam 변형 최적화 알고리즘
제곱 대신 절댓값 사용, 제곱근 대신 나눗셈 사용

논문 실험을 위한 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple


class AdamAbs(torch.optim.Optimizer):
    """
    Adam 변형 최적화 알고리즘
    
    기존 Adam:
    - v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    - θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    
    AdamAbs:
    - v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|
    - θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)
    
    Parameters:
        params: 최적화할 파라미터
        lr: 학습률 (default: 1e-3)
        betas: Adam의 베타 파라미터 (default: (0.9, 0.999))
        eps: 수치 안정성을 위한 엡실론 (default: 1e-8)
        weight_decay: 가중치 감소 (default: 0)
        amsgrad: AMSGrad 변형 사용 여부 (default: False)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=False):
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
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamAbs, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(AdamAbs, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def step(self, closure=None):
        """한 번의 최적화 스텝 수행"""
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
                    # 1차 모멘텀 (평균)
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # 2차 모멘텀 (절댓값 기반)
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        # AMSGrad를 위한 최대값 추적
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 1차 모멘텀 업데이트 (기존 Adam과 동일)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2차 모멘텀 업데이트 (절댓값 사용)
                exp_avg_sq.mul_(beta2).add_(grad.abs(), alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 편향 보정된 추정값들
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                if group['amsgrad']:
                    # AMSGrad: 최대값 추적
                    torch.maximum(max_exp_avg_sq, bias_corrected_exp_avg_sq, out=max_exp_avg_sq)
                    denominator = max_exp_avg_sq + group['eps']
                else:
                    denominator = bias_corrected_exp_avg_sq + group['eps']
                
                # 파라미터 업데이트 (제곱근 없이 바로 나눗셈)
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-group['lr'])
        
        return loss
    
    def get_lr_effective(self) -> List[float]:
        """각 파라미터 그룹의 효과적인 학습률 계산"""
        lr_effective = []
        
        for group in self.param_groups:
            group_lr_effective = []
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if 'exp_avg_sq' in state:
                        bias_correction2 = 1 - group['betas'][1] ** state['step']
                        bias_corrected_exp_avg_sq = state['exp_avg_sq'] / bias_correction2
                        denominator = bias_corrected_exp_avg_sq + group['eps']
                        lr_eff = group['lr'] / denominator.mean().item()
                        group_lr_effective.append(lr_eff)
            
            if group_lr_effective:
                lr_effective.append(sum(group_lr_effective) / len(group_lr_effective))
            else:
                lr_effective.append(group['lr'])
        
        return lr_effective


class AdamAbsW(AdamAbs):
    """
    AdamAbs with decoupled weight decay (AdamW style)
    """
    
    def step(self, closure=None):
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
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
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
                
                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, bias_corrected_exp_avg_sq, out=max_exp_avg_sq)
                    denominator = max_exp_avg_sq + group['eps']
                else:
                    denominator = bias_corrected_exp_avg_sq + group['eps']
                
                # 파라미터 업데이트 (AdamW 스타일의 분리된 weight decay)
                p.data.add_(bias_corrected_exp_avg / denominator, alpha=-group['lr'])
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


def create_optimizer(optimizer_name: str, model_params, lr: float = 1e-3, 
                    weight_decay: float = 0, **kwargs) -> torch.optim.Optimizer:
    """
    최적화 알고리즘 팩토리 함수
    
    Args:
        optimizer_name: 최적화 알고리즘 이름
        model_params: 모델 파라미터
        lr: 학습률
        weight_decay: 가중치 감소
        **kwargs: 추가 파라미터
    
    Returns:
        torch.optim.Optimizer: 최적화 알고리즘 인스턴스
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamabs':
        return AdamAbs(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamabsw':
        return AdamAbsW(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay, 
                        momentum=kwargs.get('momentum', 0.9), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def compare_optimizers_theory():
    """
    이론적 차이점 분석
    """
    print("="*80)
    print("Adam vs AdamAbs 이론적 차이점")
    print("="*80)
    
    print("\n1. 2차 모멘텀 업데이트:")
    print("   Adam:    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²")
    print("   AdamAbs: v_t = β₂ * v_{t-1} + (1 - β₂) * |g_t|")
    
    print("\n2. 파라미터 업데이트:")
    print("   Adam:    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)")
    print("   AdamAbs: θ_t = θ_{t-1} - α * m̂_t / (v̂_t + ε)")
    
    print("\n3. 주요 차이점:")
    print("   - 제곱 vs 절댓값: AdamAbs는 gradient의 절댓값을 사용하여 이상치에 덜 민감")
    print("   - 제곱근 vs 나눗셈: AdamAbs는 제곱근 연산을 피하여 계산 효율성 향상")
    print("   - 적응적 학습률: AdamAbs는 더 안정적인 학습률 조정 가능")
    
    print("\n4. 예상 장점:")
    print("   - 계산 효율성: 제곱근 연산 없음")
    print("   - 수치 안정성: 절댓값 사용으로 안정적")
    print("   - 이상치 강건성: 제곱 대신 절댓값으로 이상치 영향 감소")
    print("   - 메모리 효율성: 동일한 메모리 사용량")
    
    print("\n5. 예상 단점:")
    print("   - 이론적 수렴성: 기존 Adam의 수렴성 보장 없음")
    print("   - 하이퍼파라미터 민감성: 새로운 조정 필요 가능")
    print("   - 실험적 검증: 충분한 실험 데이터 필요")
    
    print("="*80)


if __name__ == "__main__":
    compare_optimizers_theory()
    
    # 간단한 테스트
    print("\n테스트: 간단한 모델로 최적화 알고리즘 비교")
    
    # 더미 모델 생성
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # 최적화 알고리즘 테스트
    optimizers = ['adam', 'adamabs', 'adamabsw']
    
    for opt_name in optimizers:
        try:
            optimizer = create_optimizer(opt_name, model.parameters())
            print(f"✓ {opt_name.upper()} 최적화 알고리즘 생성 성공")
        except Exception as e:
            print(f"✗ {opt_name.upper()} 최적화 알고리즘 생성 실패: {e}")
    
    print("\n모든 테스트 완료!")