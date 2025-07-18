"""
CUDA 환경에서의 Adam, AdamW, AdamAbs 성능 벤치마크
GPU 메모리 사용량, 훈련 속도, Mixed Precision 효과 등을 측정
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from adam_abs_optimizer import AdamAbs, create_optimizer
from high_dimensional_experiments import ResNet18, SimpleViT


class CUDABenchmark:
    """CUDA 환경에서의 최적화 알고리즘 벤치마크"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            print(f"CUDA 벤치마크 환경:")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Mixed precision 지원 확인
            self.use_amp = hasattr(torch.cuda, 'amp')
            print(f"  Mixed Precision: {'Available' if self.use_amp else 'Not Available'}")
        else:
            print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            self.use_amp = False
    
    def create_synthetic_data(self, batch_size=128, seq_length=1000, input_dim=512):
        """벤치마크용 합성 데이터 생성"""
        X = torch.randn(batch_size, seq_length, input_dim, device=self.device)
        y = torch.randint(0, 10, (batch_size,), device=self.device)
        return X, y
    
    def create_cnn_data(self, batch_size=128, channels=3, height=224, width=224):
        """CNN 벤치마크용 데이터 생성"""
        X = torch.randn(batch_size, channels, height, width, device=self.device)
        y = torch.randint(0, 1000, (batch_size,), device=self.device)
        return X, y
    
    def benchmark_optimizer_speed(self, model, optimizer_name, data_loader, epochs=10, use_amp=False):
        """최적화 알고리즘 속도 벤치마크"""
        model = model.to(self.device)
        optimizer = create_optimizer(optimizer_name, model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 워밍업
        model.train()
        for _ in range(3):
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 5:  # 워밍업은 5배치만
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                if use_amp and self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    scaler = torch.cuda.amp.GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
        
        # 실제 벤치마크
        torch.cuda.synchronize()
        start_time = time.time()
        
        memory_usage = []
        training_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                if use_amp and self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    scaler = torch.cuda.amp.GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                # GPU 메모리 사용량 기록
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'avg_epoch_time': np.mean(training_times),
            'max_memory_usage': max(memory_usage) if memory_usage else 0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'training_times': training_times,
            'memory_usage': memory_usage
        }
    
    def benchmark_model_sizes(self, model_configs, data_config, epochs=5):
        """다양한 모델 크기에서의 벤치마크"""
        results = {}
        
        for model_name, model_factory in model_configs.items():
            print(f"\n벤치마킹 모델: {model_name}")
            
            # 데이터 생성
            if data_config['type'] == 'cnn':
                data_loader = self.create_cnn_dataloader(**data_config['params'])
            else:
                data_loader = self.create_synthetic_dataloader(**data_config['params'])
            
            model_results = {}\n            
            for opt_name in ['adam', 'adamw', 'adamabs']:
                print(f"  최적화 알고리즘: {opt_name}")
                
                # 일반 정밀도
                model = model_factory()
                fp32_results = self.benchmark_optimizer_speed(
                    model, opt_name, data_loader, epochs=epochs, use_amp=False
                )\n                
                # Mixed precision (가능한 경우)
                if self.use_amp:
                    model = model_factory()
                    fp16_results = self.benchmark_optimizer_speed(
                        model, opt_name, data_loader, epochs=epochs, use_amp=True
                    )
                else:
                    fp16_results = None
                
                model_results[opt_name] = {
                    'fp32': fp32_results,
                    'fp16': fp16_results
                }
                
                # 메모리 정리
                del model
                torch.cuda.empty_cache()
            
            results[model_name] = model_results
        
        return results
    
    def create_cnn_dataloader(self, batch_size=128, num_batches=50, **kwargs):
        """CNN 데이터 로더 생성"""
        data_list = []
        for _ in range(num_batches):
            data_list.append(self.create_cnn_data(batch_size, **kwargs))
        return data_list
    
    def create_synthetic_dataloader(self, batch_size=128, num_batches=50, **kwargs):
        """합성 데이터 로더 생성"""
        data_list = []
        for _ in range(num_batches):
            data_list.append(self.create_synthetic_data(batch_size, **kwargs))
        return data_list
    
    def plot_benchmark_results(self, results, save_path=None):
        """벤치마크 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 훈련 시간 비교 (FP32)
        ax = axes[0, 0]
        models = list(results.keys())
        optimizers = ['adam', 'adamw', 'adamabs']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, opt in enumerate(optimizers):
            times = [results[model][opt]['fp32']['avg_epoch_time'] for model in models]
            bars = ax.bar(x + i*width, times, width, label=opt.upper())
            
            # 값 표시
            for bar, time_val in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Training Time Comparison (FP32)')
        ax.set_xlabel('Model')
        ax.set_ylabel('Time per Epoch (s)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 메모리 사용량 비교 (FP32)
        ax = axes[0, 1]
        
        for i, opt in enumerate(optimizers):
            memory = [results[model][opt]['fp32']['max_memory_usage'] for model in models]
            bars = ax.bar(x + i*width, memory, width, label=opt.upper())
            
            # 값 표시
            for bar, mem_val in zip(bars, memory):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{mem_val:.1f}GB', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Memory Usage Comparison (FP32)')
        ax.set_xlabel('Model')
        ax.set_ylabel('Max Memory Usage (GB)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. FP32 vs FP16 속도 비교 (첫 번째 모델)
        if self.use_amp:
            ax = axes[0, 2]
            first_model = models[0]
            
            fp32_times = [results[first_model][opt]['fp32']['avg_epoch_time'] for opt in optimizers]
            fp16_times = [results[first_model][opt]['fp16']['avg_epoch_time'] for opt in optimizers]
            
            x_opt = np.arange(len(optimizers))
            width = 0.35
            
            bars1 = ax.bar(x_opt - width/2, fp32_times, width, label='FP32', alpha=0.8)
            bars2 = ax.bar(x_opt + width/2, fp16_times, width, label='FP16', alpha=0.8)
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'FP32 vs FP16 Speed Comparison ({first_model})')
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Time per Epoch (s)')
            ax.set_xticks(x_opt)
            ax.set_xticklabels([opt.upper() for opt in optimizers])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. 메모리 사용량 시계열 (첫 번째 모델, Adam)
        ax = axes[1, 0]
        first_model = models[0]
        
        for opt in optimizers:
            memory_usage = results[first_model][opt]['fp32']['memory_usage']
            if memory_usage:
                ax.plot(memory_usage, label=opt.upper(), linewidth=2)
        
        ax.set_title(f'Memory Usage Over Time ({first_model})')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Memory Usage (GB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 효율성 비교 (속도/메모리)
        ax = axes[1, 1]
        
        for i, opt in enumerate(optimizers):
            efficiency = []
            for model in models:
                time_val = results[model][opt]['fp32']['avg_epoch_time']
                memory_val = results[model][opt]['fp32']['max_memory_usage']
                eff = 1 / (time_val * memory_val) if memory_val > 0 else 0
                efficiency.append(eff)
            
            bars = ax.bar(x + i*width, efficiency, width, label=opt.upper())
            
            # 값 표시
            for bar, eff_val in zip(bars, efficiency):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{eff_val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Training Efficiency (1 / (Time × Memory))')
        ax.set_xlabel('Model')
        ax.set_ylabel('Efficiency Score')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 상대적 성능 개선 (AdamAbs vs Adam)
        ax = axes[1, 2]
        
        speed_improvements = []
        memory_improvements = []
        
        for model in models:
            # 속도 개선
            adam_time = results[model]['adam']['fp32']['avg_epoch_time']
            adamabs_time = results[model]['adamabs']['fp32']['avg_epoch_time']
            speed_imp = (adam_time - adamabs_time) / adam_time * 100
            speed_improvements.append(speed_imp)
            
            # 메모리 개선
            adam_mem = results[model]['adam']['fp32']['max_memory_usage']
            adamabs_mem = results[model]['adamabs']['fp32']['max_memory_usage']
            mem_imp = (adam_mem - adamabs_mem) / adam_mem * 100 if adam_mem > 0 else 0
            memory_improvements.append(mem_imp)
        
        x_imp = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x_imp - width/2, speed_improvements, width, label='Speed', alpha=0.8)
        bars2 = ax.bar(x_imp + width/2, memory_improvements, width, label='Memory', alpha=0.8)
        
        # 값 표시
        for bars, improvements in [(bars1, speed_improvements), (bars2, memory_improvements)]:
            for bar, imp in zip(bars, improvements):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{imp:+.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('AdamAbs vs Adam: Performance Improvement')
        ax.set_xlabel('Model')
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x_imp)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_benchmark(self, save_dir='./results'):
        """포괄적인 벤치마크 실행"""
        print("CUDA 환경에서의 포괄적인 벤치마크 실행")
        print("="*60)
        
        # 모델 설정
        model_configs = {
            'ResNet18_CIFAR': lambda: ResNet18(num_classes=10),
            'ResNet18_ImageNet': lambda: ResNet18(num_classes=1000),
            'SimpleViT': lambda: SimpleViT(image_size=224, patch_size=16, num_classes=1000, 
                                          embed_dim=512, num_heads=8, num_layers=6)
        }
        
        # 데이터 설정
        data_configs = {
            'ResNet18_CIFAR': {
                'type': 'cnn',
                'params': {'batch_size': 128, 'num_batches': 30, 'channels': 3, 'height': 32, 'width': 32}
            },
            'ResNet18_ImageNet': {
                'type': 'cnn',
                'params': {'batch_size': 64, 'num_batches': 20, 'channels': 3, 'height': 224, 'width': 224}
            },
            'SimpleViT': {
                'type': 'cnn',
                'params': {'batch_size': 32, 'num_batches': 15, 'channels': 3, 'height': 224, 'width': 224}
            }
        }
        
        # 벤치마크 실행
        all_results = {}
        
        for model_name in model_configs.keys():
            print(f"\n벤치마킹 모델: {model_name}")
            
            model_config = {model_name: model_configs[model_name]}
            data_config = data_configs[model_name]
            
            results = self.benchmark_model_sizes(model_config, data_config, epochs=5)
            all_results.update(results)
        
        # 결과 시각화
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'cuda_benchmark_results.png')
        self.plot_benchmark_results(all_results, save_path)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'cuda_benchmark_{timestamp}.json')
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for model_name, model_results in all_results.items():
            serializable_results[model_name] = {}
            for opt_name, opt_results in model_results.items():
                serializable_results[model_name][opt_name] = {}
                for precision, precision_results in opt_results.items():
                    if precision_results:
                        serializable_results[model_name][opt_name][precision] = {
                            'total_time': precision_results['total_time'],
                            'avg_epoch_time': precision_results['avg_epoch_time'],
                            'max_memory_usage': precision_results['max_memory_usage'],
                            'avg_memory_usage': precision_results['avg_memory_usage']
                        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n벤치마크 결과 저장: {results_file}")
        print(f"시각화 결과 저장: {save_path}")
        
        return all_results


def main():
    """메인 함수"""
    print("CUDA 최적화 벤치마크 실행")
    print("="*60)
    
    # 벤치마크 실행
    benchmark = CUDABenchmark()
    
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. 이 벤치마크는 GPU가 필요합니다.")
        return
    
    # 결과 저장 디렉토리 설정
    save_dir = '/Users/leekeonsoo/Desktop/Code/Python/2025_AI_Paper/benchmark_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 포괄적인 벤치마크 실행
    results = benchmark.run_comprehensive_benchmark(save_dir)
    
    print("\n벤치마크 완료!")
    print(f"결과는 {save_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()