"""
CUDA 최적화된 AdamAbs 구현
간단한 버전 (Mixed Precision 제거)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class OptimizedAdamAbs(torch.optim.Optimizer):
    """간단한 AdamAbs 옵티마이저 (오류 수정됨)"""
    
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
        super(OptimizedAdamAbs, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """AdamAbs 스텝 실행"""
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
                
                # 파라미터 업데이트 (제곱근 없이 바로 나눗셈)
                denominator = bias_corrected_exp_avg_sq + group['eps']
                
                # in-place 연산 대신 일반 연산 사용
                p.data = p.data - group['lr'] * bias_corrected_exp_avg / denominator
        
        return loss




if __name__ == "__main__":
    print("OptimizedAdamAbs 준비 완료")