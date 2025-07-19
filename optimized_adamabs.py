"""
CUDA 최적화된 AdamAbs 구현
JIT 컴파일과 융합 커널을 사용한 고성능 버전
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import math


@torch.jit.script
def adamabs_single_tensor_kernel(param: torch.Tensor,
                                grad: torch.Tensor,
                                exp_avg: torch.Tensor,
                                exp_avg_sq: torch.Tensor,
                                beta1: float,
                                beta2: float,
                                lr: float,
                                eps: float,
                                step: int,
                                weight_decay: float = 0.0):
    """단일 텐서에 대한 AdamAbs 커널 (JIT 컴파일됨)"""
    
    # Weight decay 적용
    if weight_decay != 0.0:
        grad = grad.add(param, alpha=weight_decay)
    
    # Bias correction 계산
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    
    # 1차 모멘텀 업데이트 (기존 Adam과 동일)
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    
    # 2차 모멘텀 업데이트 (절댓값 사용)
    exp_avg_sq.mul_(beta2).add_(grad.abs(), alpha=1.0 - beta2)
    
    # 편향 보정
    corrected_exp_avg = exp_avg.div(bias_correction1)
    corrected_exp_avg_sq = exp_avg_sq.div(bias_correction2)
    
    # 파라미터 업데이트 (제곱근 없음)
    denom = corrected_exp_avg_sq.add_(eps)
    step_size = lr
    
    param.addcdiv_(corrected_exp_avg, denom, value=-step_size)


@torch.jit.script
def adamabs_fused_kernel(params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        exp_avgs: List[torch.Tensor],
                        exp_avg_sqs: List[torch.Tensor],
                        beta1: float,
                        beta2: float,
                        lr: float,
                        eps: float,
                        step: int,
                        weight_decay: float = 0.0):
    """다중 텐서 융합 커널 (모든 파라미터를 한 번에 처리)"""
    
    for i in range(len(params)):
        adamabs_single_tensor_kernel(
            params[i], grads[i], exp_avgs[i], exp_avg_sqs[i],
            beta1, beta2, lr, eps, step, weight_decay
        )


class OptimizedAdamAbs(torch.optim.Optimizer):
    """CUDA 최적화된 AdamAbs 옵티마이저"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=False, fused=True):
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
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       amsgrad=amsgrad, fused=fused)
        super(OptimizedAdamAbs, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """최적화된 스텝 실행"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            
            # 그래디언트가 있는 파라미터들 수집
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    
                    state = self.state[p]
                    
                    # State 초기화
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    
                    state['step'] += 1
                    state_steps.append(state['step'])
            
            # 융합 커널 사용 (CUDA에서 실행될 때만)
            if group['fused'] and params_with_grad and torch.cuda.is_available():
                # 모든 파라미터가 같은 스텝에 있다고 가정
                step = state_steps[0] if state_steps else 1
                
                # JIT 컴파일된 융합 커널 실행
                adamabs_fused_kernel(
                    params_with_grad, grads, exp_avgs, exp_avg_sqs,
                    beta1, beta2, group['lr'], group['eps'], 
                    step, group['weight_decay']
                )
            else:
                # 개별 처리 (fallback)
                for i, param in enumerate(params_with_grad):
                    step = state_steps[i]
                    adamabs_single_tensor_kernel(
                        param, grads[i], exp_avgs[i], exp_avg_sqs[i],
                        beta1, beta2, group['lr'], group['eps'],
                        step, group['weight_decay']
                    )
        
        return loss


class AdamAbsApex(torch.optim.Optimizer):
    """Apex 스타일의 AdamAbs (Multi-tensor 지원)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamAbsApex, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Multi-tensor 최적화"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 파라미터 그룹별로 처리
        for group in self.param_groups:
            # 같은 디바이스의 텐서들을 그룹화
            device_params = {}
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                device = p.device
                if device not in device_params:
                    device_params[device] = {
                        'params': [], 'grads': [], 'exp_avgs': [], 'exp_avg_sqs': [], 'steps': []
                    }
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                device_params[device]['params'].append(p)
                device_params[device]['grads'].append(p.grad)
                device_params[device]['exp_avgs'].append(state['exp_avg'])
                device_params[device]['exp_avg_sqs'].append(state['exp_avg_sq'])
                device_params[device]['steps'].append(state['step'])
            
            # 디바이스별로 융합 연산 실행
            for device, tensors in device_params.items():
                if tensors['params']:
                    with torch.cuda.device(device):
                        # 평균 스텝 사용 (모든 파라미터가 동일한 스텝이라고 가정)
                        avg_step = int(sum(tensors['steps']) / len(tensors['steps']))
                        
                        adamabs_fused_kernel(
                            tensors['params'], tensors['grads'], 
                            tensors['exp_avgs'], tensors['exp_avg_sqs'],
                            group['betas'][0], group['betas'][1],
                            group['lr'], group['eps'], avg_step, group['weight_decay']
                        )
        
        return loss


def benchmark_optimizers():
    """최적화 성능 벤치마크"""
    print("AdamAbs 최적화 버전들 성능 벤치마크")
    print("=" * 50)
    
    # 테스트 모델 생성 (ResNet18과 유사한 크기)
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).cuda()
    
    # 더미 데이터 (CIFAR-10 크기)
    data = torch.randn(256, 3, 32, 32).cuda()
    target = torch.randint(0, 10, (256,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    optimizers_to_test = {
        'Original Adam': torch.optim.Adam(model.parameters(), lr=0.001),  # 비교용
        'Optimized AdamAbs': OptimizedAdamAbs(model.parameters(), lr=0.001, fused=True),
        'Apex Style AdamAbs': AdamAbsApex(model.parameters(), lr=0.001)
    }
    
    # 벤치마크 실행
    for opt_name, optimizer in optimizers_to_test.items():
        print(f"\n{opt_name} 벤치마크:")
        
        # 워밍업
        for _ in range(10):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 실제 측정
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(100):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"  100 스텝 소요시간: {elapsed_time:.2f}ms")
        print(f"  스텝당 평균시간: {elapsed_time/100:.3f}ms")


if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_optimizers()
    else:
        print("CUDA가 필요합니다.")