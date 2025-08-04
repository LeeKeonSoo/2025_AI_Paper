"""
시각화 및 결과 저장 모듈
실험 결과를 그래프로 시각화하고 results 폴더에 저장

Author: AI Research
Date: 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (가능한 경우)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 시각화 모드 설정 (main_experiment.py에서 제어)
_SHOW_PLOTS = True  # 기본값: 그래프 창 표시
_SAVE_PLOTS = True  # 기본값: 파일 저장

def set_visualization_mode(show_plots: bool = True, save_plots: bool = True, show_message: bool = False):
    """
    시각화 모드 설정
    
    Args:
        show_plots: True면 그래프 창 표시, False면 자동 저장만
        save_plots: True면 파일로 저장, False면 저장 안함
        show_message: True면 설정 메시지 표시, False면 조용히 설정
    """
    global _SHOW_PLOTS, _SAVE_PLOTS
    _SHOW_PLOTS = show_plots
    _SAVE_PLOTS = save_plots
    
    if show_plots:
        plt.ion()   # 인터랙티브 모드 켜기
        if show_message:
            print("📊 시각화 모드: 그래프 창 표시 (X 버튼으로 닫아야 진행)")
    else:
        plt.ioff()  # 인터랙티브 모드 끄기
        if show_message:
            print("📊 시각화 모드: 자동 저장 (창 표시 안함)")
    
    if show_message:
        if save_plots:
            print("💾 저장 모드: 파일로 저장")
        else:
            print("🚫 저장 모드: 저장 안함")


def _handle_plot_display_and_save(save_path: Optional[str], message: str):
    """
    플롯 표시 및 저장을 전역 설정에 따라 처리
    
    Args:
        save_path: 저장할 파일 경로 (None이면 저장 안함)
        message: 저장 완료 메시지
    """
    global _SHOW_PLOTS, _SAVE_PLOTS
    
    if _SAVE_PLOTS and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(message)
    
    if _SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # 메모리 절약


class ExperimentVisualizer:
    """실험 결과 시각화 클래스"""
    
    def __init__(self, results_dir: str = './results'):
        """
        Args:
            results_dir: 결과 저장 디렉토리
        """
        self.results_dir = results_dir
        self.colors = {
            'Adam': '#1f77b4',
            'AdamW': '#ff7f0e', 
            'AdamABS': '#2ca02c'
        }
        
        # 결과 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"📊 ExperimentVisualizer 초기화")
        print(f"   결과 저장 경로: {os.path.abspath(results_dir)}")
    
    def plot_training_curves(self, experiment_results: Dict[str, Any], 
                           dataset_name: str, save_name: Optional[str] = None) -> Optional[str]:
        """
        훈련 곡선 시각화
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
            save_name: 저장할 파일명 (없으면 자동 생성)
        
        Returns:
            str: 저장된 파일 경로
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} Optimizer Comparison - Training Curves', 
                    fontsize=16, fontweight='bold')
        
        # 1. 훈련 손실
        ax = axes[0, 0]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax.plot(history['train_loss'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. 검증 손실
        ax = axes[0, 1]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax.plot(history['val_loss'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. 훈련 정확도
        ax = axes[0, 2]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax.plot(history['train_acc'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 검증 정확도
        ax = axes[1, 0]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax.plot(history['val_acc'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 학습률 변화
        ax = axes[1, 1]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            if 'lr_history' in history and history['lr_history']:
                ax.plot(history['lr_history'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 6. 에포크 시간
        ax = axes[1, 2]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            if 'epoch_times' in history:
                ax.plot(history['epoch_times'], label=opt_name, color=color, linewidth=2)
        
        ax.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_training_curves_{timestamp}.png'
        
        save_path = os.path.join(self.results_dir, save_name) if _SAVE_PLOTS else None
        _handle_plot_display_and_save(save_path, f"📈 훈련 곡선 저장: {save_path}")
        
        return save_path
    
    def plot_performance_comparison(self, experiment_results: Dict[str, Any],
                                  dataset_name: str, save_name: Optional[str] = None) -> str:
        """
        성능 비교 차트
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름  
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dataset_name} Optimizer Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        opt_names = list(experiment_results.keys())
        colors_list = [self.colors.get(name, 'gray') for name in opt_names]
        
        # 1. 최고 검증 정확도
        ax = axes[0, 0]
        val_accs = [experiment_results[name]['best_val_acc'] for name in opt_names]
        
        bars = ax.bar(opt_names, val_accs, color=colors_list, alpha=0.8)
        ax.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(max(0, min(val_accs) - 5), min(100, max(val_accs) + 5))
        
        # 막대 위에 값 표시
        for bar, acc in zip(bars, val_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. 최종 테스트 정확도 (있는 경우)
        ax = axes[0, 1]
        test_accs = []
        for name in opt_names:
            if 'test_results' in experiment_results[name]:
                test_accs.append(experiment_results[name]['test_results']['accuracy'])
            else:
                test_accs.append(experiment_results[name]['final_val_acc'])
        
        bars = ax.bar(opt_names, test_accs, color=colors_list, alpha=0.8)
        ax.set_title('Final Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(max(0, min(test_accs) - 5), min(100, max(test_accs) + 5))
        
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 3. 총 훈련 시간
        ax = axes[1, 0]
        train_times = [experiment_results[name]['total_training_time'] for name in opt_names]
        
        bars = ax.bar(opt_names, train_times, color=colors_list, alpha=0.8)
        ax.set_title('Total Training Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        for bar, time_val in zip(bars, train_times):
            # 시분초 형식으로 표시
            if time_val >= 3600:  # 1시간 이상
                time_str = f'{int(time_val//3600)}h{int((time_val%3600)//60)}m'
            elif time_val >= 60:  # 1분 이상
                time_str = f'{int(time_val//60)}m{int(time_val%60)}s'
            else:
                time_str = f'{time_val:.1f}s'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.01,
                   time_str, ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 4. 평균 에포크 시간
        ax = axes[1, 1]
        epoch_times = [experiment_results[name]['avg_epoch_time'] for name in opt_names]
        
        bars = ax.bar(opt_names, epoch_times, color=colors_list, alpha=0.8)
        ax.set_title('Average Epoch Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        for bar, time_val in zip(bars, epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(epoch_times)*0.01,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_performance_comparison_{timestamp}.png'
        
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 성능 비교 차트 저장: {save_path}")
        return save_path
    
    def plot_optimizer_analysis(self, experiment_results: Dict[str, Any],
                              dataset_name: str, save_name: Optional[str] = None) -> str:
        """
        옵티마이저 분석 차트 (수렴성, 안정성 등)
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dataset_name} Optimizer Analysis', 
                    fontsize=16, fontweight='bold')
        
        opt_names = list(experiment_results.keys())
        colors_list = [self.colors.get(name, 'gray') for name in opt_names]
        
        # 1. 수렴 분석 (최고 성능 달성 에포크)
        ax = axes[0, 0]
        convergence_epochs = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            best_epoch = np.argmax(val_acc_history) + 1
            convergence_epochs.append(best_epoch)
        
        bars = ax.bar(opt_names, convergence_epochs, color=colors_list, alpha=0.8)
        ax.set_title('Best Performance Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Epoch')
        
        for bar, epoch in zip(bars, convergence_epochs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{epoch}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 2. 안정성 분석 (마지막 10 에포크 검증 정확도 표준편차)
        ax = axes[0, 1]
        stability_scores = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            if len(val_acc_history) >= 10:
                stability = np.std(val_acc_history[-10:])
            else:
                stability = np.std(val_acc_history)
            stability_scores.append(stability)
        
        bars = ax.bar(opt_names, stability_scores, color=colors_list, alpha=0.8)
        ax.set_title('Training Stability (Last 10 Epochs Std)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Standard Deviation (%)')
        
        for bar, std in zip(bars, stability_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_scores)*0.01,
                   f'{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 3. 효율성 분석 (성능 대비 시간)
        ax = axes[1, 0]
        efficiency_scores = []
        for name in opt_names:
            best_acc = experiment_results[name]['best_val_acc']
            total_time = experiment_results[name]['total_training_time']
            efficiency = best_acc / (total_time / 60)  # 분당 정확도
            efficiency_scores.append(efficiency)
        
        bars = ax.bar(opt_names, efficiency_scores, color=colors_list, alpha=0.8)
        ax.set_title('Efficiency (Accuracy per Minute)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy % / Minute')
        
        for bar, eff in zip(bars, efficiency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_scores)*0.01,
                   f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 4. 학습 진행도 (에포크별 개선도)
        ax = axes[1, 1]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            val_acc = history['val_acc']
            if len(val_acc) > 1:
                # 이동평균을 통한 부드러운 곡선
                window_size = max(1, len(val_acc) // 10)
                smoothed_acc = pd.Series(val_acc).rolling(window=window_size, center=True).mean()
                color = self.colors.get(opt_name, 'black')
                ax.plot(smoothed_acc, label=f'{opt_name} (smoothed)', 
                       color=color, linewidth=2, alpha=0.8)
        
        ax.set_title('Learning Progress (Smoothed)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_optimizer_analysis_{timestamp}.png'
        
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔍 옵티마이저 분석 저장: {save_path}")
        return save_path
    
    def create_summary_report(self, experiment_results: Dict[str, Any],
                            dataset_name: str, save_name: Optional[str] = None) -> str:
        """
        종합 요약 보고서 생성
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 전체 제목
        fig.suptitle(f'{dataset_name} Optimizer Comparison - Comprehensive Report', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 그리드 레이아웃 설정 (4x4)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        opt_names = list(experiment_results.keys())
        colors_list = [self.colors.get(name, 'gray') for name in opt_names]
        
        # 1. 훈련/검증 정확도 (상위 좌측 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax1.plot(history['train_acc'], '--', color=color, alpha=0.7, linewidth=1.5, label=f'{opt_name} Train')
            ax1.plot(history['val_acc'], '-', color=color, linewidth=2, label=f'{opt_name} Val')
        
        ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 손실 곡선 (상위 우측 2x2)  
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            color = self.colors.get(opt_name, 'black')
            ax2.plot(history['train_loss'], '--', color=color, alpha=0.7, linewidth=1.5, label=f'{opt_name} Train')
            ax2.plot(history['val_loss'], '-', color=color, linewidth=2, label=f'{opt_name} Val')
        
        ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. 최종 성능 비교
        ax3 = fig.add_subplot(gs[2, 0])
        val_accs = [experiment_results[name]['best_val_acc'] for name in opt_names]
        bars = ax3.bar(opt_names, val_accs, color=colors_list, alpha=0.8)
        ax3.set_title('Best Validation Accuracy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        for bar, acc in zip(bars, val_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. 훈련 시간 비교
        ax4 = fig.add_subplot(gs[2, 1])
        train_times = [experiment_results[name]['total_training_time'] for name in opt_names]
        bars = ax4.bar(opt_names, train_times, color=colors_list, alpha=0.8)
        ax4.set_title('Training Time', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time (s)')
        for bar, time_val in zip(bars, train_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.01,
                    f'{time_val:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 5. 수렴 속도
        ax5 = fig.add_subplot(gs[2, 2])
        convergence_epochs = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            best_epoch = np.argmax(val_acc_history) + 1
            convergence_epochs.append(best_epoch)
        
        bars = ax5.bar(opt_names, convergence_epochs, color=colors_list, alpha=0.8)
        ax5.set_title('Convergence Speed', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Best Epoch')
        for bar, epoch in zip(bars, convergence_epochs):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{epoch}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # 6. 안정성 점수
        ax6 = fig.add_subplot(gs[2, 3])
        stability_scores = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            if len(val_acc_history) >= 10:
                stability = np.std(val_acc_history[-10:])
            else:
                stability = np.std(val_acc_history)
            stability_scores.append(stability)
        
        bars = ax6.bar(opt_names, stability_scores, color=colors_list, alpha=0.8)
        ax6.set_title('Stability Score', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Std Dev (%)')
        for bar, std in zip(bars, stability_scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_scores)*0.01,
                    f'{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        # 7. 요약 테이블 (하단)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # 요약 데이터 준비
        summary_data = []
        for opt_name in opt_names:
            results = experiment_results[opt_name]
            test_acc = results.get('test_results', {}).get('accuracy', results['final_val_acc'])
            
            summary_data.append([
                opt_name,
                f"{results['best_val_acc']:.2f}%",
                f"{test_acc:.2f}%", 
                f"{results['total_training_time']:.1f}s",
                f"{results['avg_epoch_time']:.1f}s",
                f"{convergence_epochs[opt_names.index(opt_name)]}",
                f"{stability_scores[opt_names.index(opt_name)]:.3f}"
            ])
        
        # 테이블 생성
        table = ax7.table(cellText=summary_data,
                         colLabels=['Optimizer', 'Best Val Acc', 'Test Acc', 'Total Time', 
                                   'Avg Epoch Time', 'Best Epoch', 'Stability'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 테이블 스타일링
        for i in range(len(opt_names) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # 헤더
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:  # 데이터 행
                    cell.set_facecolor('#f1f1f2')
                    if j == 0:  # 옵티마이저 이름
                        cell.set_facecolor(colors_list[i-1])
                        cell.set_text_props(weight='bold', color='white')
        
        ax7.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_comprehensive_report_{timestamp}.png'
        
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📋 종합 보고서 저장: {save_path}")
        return save_path
    
    def save_results_json(self, experiment_results: Dict[str, Any], 
                         dataset_name: str, save_name: Optional[str] = None) -> str:
        """
        실험 결과를 JSON 파일로 저장
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        
        for opt_name, results in experiment_results.items():
            serializable_results[opt_name] = {
                'optimizer_name': opt_name,
                'best_val_acc': results['best_val_acc'],
                'final_train_acc': results['final_train_acc'],
                'final_val_acc': results['final_val_acc'],
                'total_training_time': results['total_training_time'],
                'avg_epoch_time': results['avg_epoch_time'],
                'total_epochs_trained': results['total_epochs_trained'],
                'optimizer_config': results.get('optimizer_config', {}),
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # 테스트 결과 추가 (있는 경우)
            if 'test_results' in results:
                serializable_results[opt_name]['test_accuracy'] = results['test_results']['accuracy']
                serializable_results[opt_name]['test_loss'] = results['test_results']['loss']
            
            # 수렴 분석
            val_acc_history = results['training_history']['val_acc']
            if val_acc_history:
                best_epoch = np.argmax(val_acc_history) + 1
                final_stability = np.std(val_acc_history[-min(10, len(val_acc_history)):])
                
                serializable_results[opt_name]['convergence_epoch'] = int(best_epoch)
                serializable_results[opt_name]['final_stability'] = float(final_stability)
        
        # 실험 메타데이터 추가
        metadata = {
            'experiment_info': {
                'dataset': dataset_name,
                'total_optimizers': len(experiment_results),
                'optimizers_tested': list(experiment_results.keys()),
                'experiment_date': datetime.now().isoformat(),
                'best_optimizer': max(experiment_results.keys(), 
                                    key=lambda x: experiment_results[x]['best_val_acc']),
                'best_accuracy': max(results['best_val_acc'] for results in experiment_results.values())
            }
        }
        
        final_data = {
            'metadata': metadata,
            'results': serializable_results
        }
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_experiment_results_{timestamp}.json'
        
        save_path = os.path.join(self.results_dir, save_name)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 실험 결과 JSON 저장: {save_path}")
        return save_path
    
    def generate_all_visualizations(self, experiment_results: Dict[str, Any], 
                                  dataset_name: str) -> Dict[str, str]:
        """
        모든 시각화 생성 및 저장
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
        
        Returns:
            Dict[str, str]: 생성된 파일들의 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{dataset_name.lower()}_{timestamp}"
        
        print(f"\n📊 {dataset_name} 실험 결과 시각화 생성 중...")
        print("="*60)
        
        saved_files = {}
        
        # 1. 훈련 곡선
        saved_files['training_curves'] = self.plot_training_curves(
            experiment_results, dataset_name, f"{base_name}_training_curves.png"
        )
        
        # 2. 성능 비교
        saved_files['performance_comparison'] = self.plot_performance_comparison(
            experiment_results, dataset_name, f"{base_name}_performance.png"
        )
        
        # 3. 옵티마이저 분석
        saved_files['optimizer_analysis'] = self.plot_optimizer_analysis(
            experiment_results, dataset_name, f"{base_name}_analysis.png"
        )
        
        # 4. 종합 보고서
        saved_files['comprehensive_report'] = self.create_summary_report(
            experiment_results, dataset_name, f"{base_name}_report.png"
        )
        
        # 5. JSON 결과
        saved_files['json_results'] = self.save_results_json(
            experiment_results, dataset_name, f"{base_name}_results.json"
        )
        
        print("="*60)
        print(f"✅ 모든 시각화 완료! 총 {len(saved_files)}개 파일 생성")
        print(f"📂 저장 경로: {os.path.abspath(self.results_dir)}")
        
        return saved_files


class AdamABSAnalyzer:
    """AdamABS 논문용 특화 분석 클래스"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        self.colors = {
            'Adam': '#1f77b4',
            'AdamW': '#ff7f0e', 
            'AdamABS': '#2ca02c'
        }
        os.makedirs(results_dir, exist_ok=True)
        
        # 논문용 스타일 설정
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.titlesize': 15,
            'lines.linewidth': 2.5,
            'grid.alpha': 0.3
        })
    
    def plot_efficiency_comparison(self, experiment_results: Dict[str, Any], 
                                 dataset_name: str) -> str:
        """계산 효율성 비교 (AdamABS의 핵심 장점)"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'AdamABS Efficiency Analysis - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        opt_names = list(experiment_results.keys())
        colors_list = [self.colors.get(name, 'gray') for name in opt_names]
        
        # 1. 정확도 vs 훈련시간
        ax = axes[0, 0]
        for i, opt_name in enumerate(opt_names):
            results = experiment_results[opt_name]
            acc = results['best_val_acc']
            time = results['total_training_time']
            ax.scatter(time, acc, s=200, c=colors_list[i], alpha=0.8, 
                      label=opt_name, edgecolors='black', linewidth=1)
            ax.annotate(f'{opt_name}\n{acc:.1f}%', (time, acc), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Total Training Time (seconds)')
        ax.set_ylabel('Best Validation Accuracy (%)')
        ax.set_title('Accuracy vs Training Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 에포크당 시간 비교
        ax = axes[0, 1]
        epoch_times = [experiment_results[name]['avg_epoch_time'] for name in opt_names]
        bars = ax.bar(opt_names, epoch_times, color=colors_list, alpha=0.8)
        
        # Adam 대비 개선도 계산
        if 'Adam' in opt_names:
            adam_time = experiment_results['Adam']['avg_epoch_time']
            for i, (bar, time_val, opt_name) in enumerate(zip(bars, epoch_times, opt_names)):
                improvement = (adam_time - time_val) / adam_time * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{time_val:.1f}s\n({improvement:+.1f}%)', 
                       ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Average Epoch Time', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 3. 수렴 효율성 (목표 정확도 도달 시간)
        ax = axes[1, 0]
        target_acc = 50.0  # 50% 목표
        convergence_times = []
        
        for opt_name in opt_names:
            results = experiment_results[opt_name]
            val_acc_history = results['training_history']['val_acc']
            epoch_times = results['training_history'].get('epoch_times', 
                                                        [results['avg_epoch_time']] * len(val_acc_history))
            
            # 목표 정확도 달성 시점 찾기
            converged_epoch = None
            for epoch, acc in enumerate(val_acc_history):
                if acc >= target_acc:
                    converged_epoch = epoch
                    break
            
            if converged_epoch is not None:
                conv_time = sum(epoch_times[:converged_epoch+1])
                convergence_times.append(conv_time)
            else:
                convergence_times.append(sum(epoch_times))  # 전체 시간
        
        bars = ax.bar(opt_names, convergence_times, color=colors_list, alpha=0.8)
        ax.set_title(f'Time to Reach {target_acc}% Accuracy', fontweight='bold')
        ax.set_ylabel('Time (seconds)')
        
        for bar, time_val in zip(bars, convergence_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(convergence_times)*0.01,
                   f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 4. 종합 효율성 점수
        ax = axes[1, 1]
        efficiency_scores = []
        
        for opt_name in opt_names:
            results = experiment_results[opt_name]
            # 효율성 = 정확도 / (시간 * 복잡도_가중치)
            acc = results['best_val_acc']
            time = results['total_training_time']
            
            # AdamABS는 sqrt 연산이 없어서 복잡도 가중치 0.9
            complexity_weight = 0.9 if 'ABS' in opt_name else 1.0
            efficiency = acc / (time * complexity_weight)
            efficiency_scores.append(efficiency)
        
        bars = ax.bar(opt_names, efficiency_scores, color=colors_list, alpha=0.8)
        ax.set_title('Overall Efficiency Score\n(Accuracy / Weighted Time)', fontweight='bold')
        ax.set_ylabel('Efficiency Score')
        
        for bar, score in zip(bars, efficiency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_scores)*0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, 
                               f'{dataset_name.lower()}_adamabs_efficiency_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⚡ AdamABS 효율성 분석 저장: {save_path}")
        return save_path
    
    def plot_convergence_analysis(self, experiment_results: Dict[str, Any], 
                                dataset_name: str) -> str:
        """수렴성 분석 (AdamABS vs Adam/AdamW)"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Convergence Analysis: AdamABS vs Others - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 학습 곡선 비교 (smoothed)
        ax = axes[0, 0]
        for opt_name, results in experiment_results.items():
            history = results['training_history']
            val_acc = history['val_acc']
            
            # 이동평균으로 부드럽게
            window = max(3, len(val_acc) // 10)
            if len(val_acc) > window:
                smoothed_acc = pd.Series(val_acc).rolling(window=window, center=True).mean()
                epochs = range(1, len(smoothed_acc) + 1)
                color = self.colors.get(opt_name, 'black')
                ax.plot(epochs, smoothed_acc, label=opt_name, color=color, linewidth=2.5)
        
        ax.set_title('Validation Accuracy (Smoothed)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 수렴 속도 (정확도 증가율)
        ax = axes[0, 1]
        for opt_name, results in experiment_results.items():
            val_acc = results['training_history']['val_acc']
            if len(val_acc) > 1:
                # 에포크별 정확도 증가율
                improvement_rate = np.gradient(val_acc)
                epochs = range(1, len(improvement_rate) + 1)
                color = self.colors.get(opt_name, 'black')
                ax.plot(epochs, improvement_rate, label=opt_name, color=color, alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Learning Rate (Accuracy Gradient)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy Improvement per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 안정성 분석 (rolling std)
        ax = axes[1, 0]
        for opt_name, results in experiment_results.items():
            val_acc = results['training_history']['val_acc']
            if len(val_acc) > 5:
                # 5 에포크 rolling standard deviation
                rolling_std = pd.Series(val_acc).rolling(window=5).std()
                epochs = range(1, len(rolling_std) + 1)
                color = self.colors.get(opt_name, 'black')
                ax.plot(epochs, rolling_std, label=opt_name, color=color, alpha=0.8)
        
        ax.set_title('Training Stability (5-Epoch Rolling Std)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 최종 수렴 비교
        ax = axes[1, 1]
        
        # 수렴 지표들 계산
        opt_names = list(experiment_results.keys())
        convergence_metrics = []
        
        for opt_name in opt_names:
            results = experiment_results[opt_name]
            val_acc = results['training_history']['val_acc']
            
            # 수렴 지표들
            best_epoch = np.argmax(val_acc) + 1
            final_stability = np.std(val_acc[-min(10, len(val_acc)):])
            max_acc = max(val_acc)
            final_acc = val_acc[-1]
            
            convergence_metrics.append({
                'optimizer': opt_name,
                'best_epoch': best_epoch,
                'stability': final_stability,
                'max_accuracy': max_acc,
                'final_accuracy': final_acc
            })
        
        # 수렴 품질 점수 계산 (높을수록 좋음)
        quality_scores = []
        for metrics in convergence_metrics:
            # 점수 = 최대정확도 - 조기수렴패널티 - 불안정성패널티
            score = (metrics['max_accuracy'] - 
                    (metrics['best_epoch'] / 100) * 5 -  # 조기 수렴 보너스
                    metrics['stability'] * 10)  # 안정성 보너스
            quality_scores.append(score)
        
        colors_list = [self.colors.get(opt['optimizer'], 'gray') for opt in convergence_metrics]
        bars = ax.bar([opt['optimizer'] for opt in convergence_metrics], 
                     quality_scores, color=colors_list, alpha=0.8)
        
        ax.set_title('Convergence Quality Score\n(Higher = Better)', fontweight='bold')
        ax.set_ylabel('Quality Score')
        
        for bar, score, metrics in zip(bars, quality_scores, convergence_metrics):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{score:.1f}\n(E{metrics["best_epoch"]})', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, 
                               f'{dataset_name.lower()}_convergence_analysis_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 수렴성 분석 저장: {save_path}")
        return save_path
    
    def create_paper_figure(self, experiment_results: Dict[str, Any], 
                          dataset_name: str) -> str:
        """논문용 종합 그림 생성"""
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'AdamABS: A Square-Root-Free Adam Optimizer\nExperimental Results on {dataset_name}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 6개 서브플롯으로 구성
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        opt_names = list(experiment_results.keys())
        colors_list = [self.colors.get(name, 'gray') for name in opt_names]
        
        # 1. 정확도 비교 (좌상)
        ax1 = fig.add_subplot(gs[0, 0])
        val_accs = [experiment_results[name]['best_val_acc'] for name in opt_names]
        bars = ax1.bar(opt_names, val_accs, color=colors_list, alpha=0.8, edgecolor='black')
        ax1.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        
        for bar, acc in zip(bars, val_accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 훈련 시간 효율성 (중상)
        ax2 = fig.add_subplot(gs[0, 1])
        train_times = [experiment_results[name]['total_training_time'] for name in opt_names]
        bars = ax2.bar(opt_names, train_times, color=colors_list, alpha=0.8, edgecolor='black')
        ax2.set_title('Total Training Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        
        # Adam 대비 시간 절약 표시
        if 'Adam' in opt_names:
            adam_time = experiment_results['Adam']['total_training_time']
            for bar, time_val, opt_name in zip(bars, train_times, opt_names):
                improvement = (adam_time - time_val) / adam_time * 100
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.02,
                        f'{time_val:.0f}s\n({improvement:+.1f}%)', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 학습 곡선 (우상)
        ax3 = fig.add_subplot(gs[0, 2])
        for opt_name, results in experiment_results.items():
            val_acc = results['training_history']['val_acc']
            epochs = range(1, len(val_acc) + 1)
            color = self.colors.get(opt_name, 'black')
            ax3.plot(epochs, val_acc, label=opt_name, color=color, linewidth=2.5, alpha=0.9)
        
        ax3.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 수렴 속도 (좌하)
        ax4 = fig.add_subplot(gs[1, 0])
        convergence_epochs = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            best_epoch = np.argmax(val_acc_history) + 1
            convergence_epochs.append(best_epoch)
        
        bars = ax4.bar(opt_names, convergence_epochs, color=colors_list, alpha=0.8, edgecolor='black')
        ax4.set_title('Convergence Speed\n(Best Performance Epoch)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Epoch')
        
        for bar, epoch in zip(bars, convergence_epochs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{epoch}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.setp(ax4.get_xticklabels(), rotation=0, ha='center')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 안정성 (중하)
        ax5 = fig.add_subplot(gs[1, 1])
        stability_scores = []
        for name in opt_names:
            val_acc_history = experiment_results[name]['training_history']['val_acc']
            stability = np.std(val_acc_history[-min(10, len(val_acc_history)):])
            stability_scores.append(stability)
        
        bars = ax5.bar(opt_names, stability_scores, color=colors_list, alpha=0.8, edgecolor='black')
        ax5.set_title('Training Stability\n(Lower = More Stable)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Standard Deviation (%)')
        
        for bar, std in zip(bars, stability_scores):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_scores)*0.02,
                    f'{std:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.setp(ax5.get_xticklabels(), rotation=0, ha='center')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 종합 성능 점수 (우하)
        ax6 = fig.add_subplot(gs[1, 2])
        
        # 종합 점수 계산: 정확도 + 속도 + 안정성
        overall_scores = []
        for i, name in enumerate(opt_names):
            acc_score = val_accs[i] / max(val_accs) * 40  # 40점 만점
            speed_score = (max(train_times) - train_times[i]) / max(train_times) * 30  # 30점 만점
            stability_score = (max(stability_scores) - stability_scores[i]) / max(stability_scores) * 30  # 30점 만점
            
            total_score = acc_score + speed_score + stability_score
            overall_scores.append(total_score)
        
        bars = ax6.bar(opt_names, overall_scores, color=colors_list, alpha=0.8, edgecolor='black')
        ax6.set_title('Overall Performance Score\n(Accuracy + Speed + Stability)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score (0-100)')
        ax6.set_ylim(0, 100)
        
        for bar, score in zip(bars, overall_scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.setp(ax6.get_xticklabels(), rotation=0, ha='center')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 하단에 요약 텍스트 추가
        fig.text(0.5, 0.02, 
                f'Dataset: {dataset_name} | Optimizers Tested: {", ".join(opt_names)} | '
                f'Best Optimizer: {opt_names[np.argmax(overall_scores)]} (Score: {max(overall_scores):.1f})',
                ha='center', va='bottom', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, 
                               f'{dataset_name.lower()}_paper_figure_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📄 논문용 종합 그림 저장: {save_path}")
        return save_path


class BatchSizeVisualizer:
    """배치 사이즈 비교 실험 전용 시각화 클래스"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        self.colors = {
            'Adam': '#1f77b4',
            'AdamABS': '#2ca02c',
            'AdamW': '#ff7f0e'
        }
        self.batch_colors = {
            64: '#e74c3c',   # 빨간색
            128: '#f39c12',  # 주황색  
            256: '#8e44ad'   # 보라색
        }
        
        # 배치 사이즈 결과 디렉토리 생성
        self.batch_results_dir = os.path.join(results_dir, 'batch_comparison')
        os.makedirs(self.batch_results_dir, exist_ok=True)
        
        print(f"📊 BatchSizeVisualizer 초기화")
        print(f"   결과 저장 경로: {os.path.abspath(self.batch_results_dir)}")
    
    def plot_batch_size_comparison(self, all_results: Dict[str, Any], 
                                 save_name: Optional[str] = None) -> str:
        """
        배치 사이즈별 성능 비교 그래프
        
        Args:
            all_results: {dataset_name: {batch_64: {Adam: {...}, AdamABS: {...}}, ...}}
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        datasets = list(all_results.keys())
        batch_sizes = [64, 128, 256]
        optimizers = ['Adam', 'AdamABS']
        
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))
        if len(datasets) == 1:
            axes = [axes]
        
        fig.suptitle('Batch Size Comparison: Adam vs AdamABS', 
                    fontsize=16, fontweight='bold')
        
        for dataset_idx, (dataset_name, dataset_results) in enumerate(all_results.items()):
            ax = axes[dataset_idx]
            
            # 각 옵티마이저별로 배치 사이즈에 따른 성능 플롯
            for opt_idx, optimizer in enumerate(optimizers):
                accuracies = []
                available_batch_sizes = []
                
                for batch_size in batch_sizes:
                    batch_key = f"batch_{batch_size}"
                    if batch_key in dataset_results and optimizer in dataset_results[batch_key]:
                        acc = dataset_results[batch_key][optimizer].get('best_val_acc', 0)
                        accuracies.append(acc)
                        available_batch_sizes.append(batch_size)
                
                if accuracies:
                    color = self.colors.get(optimizer, 'gray')
                    ax.plot(available_batch_sizes, accuracies, 
                           marker='o', linewidth=2.5, markersize=8,
                           color=color, label=optimizer, alpha=0.8)
                    
                    # 각 점에 정확도 값 표시
                    for bs, acc in zip(available_batch_sizes, accuracies):
                        ax.annotate(f'{acc:.1f}%', (bs, acc), 
                                   xytext=(0, 10), textcoords='offset points',
                                   ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Best Validation Accuracy (%)')
            ax.set_title(f'{dataset_name}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(batch_sizes)
            ax.set_xscale('log', base=2)  # 로그 스케일로 배치 사이즈 표시
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'batch_size_comparison_{timestamp}.png'
        
        save_path = os.path.join(self.batch_results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 배치 사이즈 비교 그래프 저장: {save_path}")
        return save_path
    
    def plot_batch_size_heatmap(self, all_results: Dict[str, Any], 
                              optimizer: str = 'AdamABS',
                              save_name: Optional[str] = None) -> str:
        """
        데이터셋 × 배치사이즈 성능 히트맵
        
        Args:
            all_results: 모든 실험 결과
            optimizer: 분석할 옵티마이저
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        datasets = list(all_results.keys())
        batch_sizes = [64, 128, 256]
        
        # 데이터 매트릭스 생성
        data_matrix = []
        for dataset_name in datasets:
            dataset_results = all_results[dataset_name]
            row = []
            for batch_size in batch_sizes:
                batch_key = f"batch_{batch_size}"
                if (batch_key in dataset_results and 
                    optimizer in dataset_results[batch_key]):
                    acc = dataset_results[batch_key][optimizer].get('best_val_acc', 0)
                    row.append(acc)
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        
        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 축 라벨 설정
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([f'Batch {bs}' for bs in batch_sizes])
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        
        # 각 셀에 수치 표시
        for i in range(len(datasets)):
            for j in range(len(batch_sizes)):
                if not np.isnan(data_matrix[i][j]):
                    text = ax.text(j, i, f'{data_matrix[i][j]:.1f}%',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Validation Accuracy (%)', rotation=270, labelpad=20)
        
        ax.set_title(f'{optimizer} Performance Heatmap\n(Dataset × Batch Size)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{optimizer.lower()}_batch_heatmap_{timestamp}.png'
        
        save_path = os.path.join(self.batch_results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔥 {optimizer} 배치 사이즈 히트맵 저장: {save_path}")
        return save_path
    
    def plot_convergence_by_batch_size(self, all_results: Dict[str, Any],
                                     dataset_name: str,
                                     save_name: Optional[str] = None) -> str:
        """
        배치 사이즈별 수렴 속도 분석
        
        Args:
            all_results: 모든 실험 결과
            dataset_name: 분석할 데이터셋
            save_name: 저장할 파일명
        
        Returns:
            str: 저장된 파일 경로
        """
        if dataset_name not in all_results:
            raise ValueError(f"Dataset {dataset_name} not found in results")
        
        dataset_results = all_results[dataset_name]
        batch_sizes = [64, 128, 256]
        optimizers = ['Adam', 'AdamABS']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Convergence Analysis: {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        for opt_idx, optimizer in enumerate(optimizers):
            ax = axes[opt_idx]
            
            for batch_size in batch_sizes:
                batch_key = f"batch_{batch_size}"
                if (batch_key in dataset_results and 
                    optimizer in dataset_results[batch_key]):
                    
                    results = dataset_results[batch_key][optimizer]
                    if 'training_history' in results:
                        val_acc = results['training_history'].get('val_acc', [])
                        if val_acc:
                            epochs = range(1, len(val_acc) + 1)
                            color = self.batch_colors.get(batch_size, 'gray')
                            ax.plot(epochs, val_acc, 
                                   color=color, linewidth=2, alpha=0.8,
                                   label=f'Batch {batch_size}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation Accuracy (%)')
            ax.set_title(f'{optimizer}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{dataset_name.lower()}_convergence_analysis_{timestamp}.png'
        
        save_path = os.path.join(self.batch_results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 {dataset_name} 수렴 분석 저장: {save_path}")
        return save_path
    
    def create_batch_size_analysis(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """
        배치 사이즈 분석 종합 보고서 생성
        
        Args:
            all_results: 모든 실험 결과
        
        Returns:
            Dict[str, str]: 생성된 파일들의 경로
        """
        print("\n🚀 배치 사이즈 분석 종합 보고서 생성 중...")
        print("="*60)
        
        saved_files = {}
        
        # 1. 전체 배치 사이즈 비교
        try:
            saved_files['batch_comparison'] = self.plot_batch_size_comparison(all_results)
        except Exception as e:
            print(f"❌ 배치 사이즈 비교 그래프 생성 실패: {e}")
        
        # 2. AdamABS 히트맵
        try:
            saved_files['adamabs_heatmap'] = self.plot_batch_size_heatmap(all_results, 'AdamABS')
        except Exception as e:
            print(f"❌ AdamABS 히트맵 생성 실패: {e}")
        
        # 3. Adam 히트맵
        try:
            saved_files['adam_heatmap'] = self.plot_batch_size_heatmap(all_results, 'Adam')
        except Exception as e:
            print(f"❌ Adam 히트맵 생성 실패: {e}")
        
        # 4. 각 데이터셋별 수렴 분석
        for dataset_name in all_results.keys():
            try:
                key = f'{dataset_name.lower()}_convergence'
                saved_files[key] = self.plot_convergence_by_batch_size(all_results, dataset_name)
            except Exception as e:
                print(f"❌ {dataset_name} 수렴 분석 생성 실패: {e}")
        
        # 5. 종합 리포트 생성
        try:
            saved_files['comprehensive_report'] = self.create_comprehensive_report(all_results)
        except Exception as e:
            print(f"❌ 종합 리포트 생성 실패: {e}")
        
        print(f"\n✅ 배치 사이즈 분석 완료! {len(saved_files)}개 파일 생성")
        return saved_files
    
    def create_comprehensive_report(self, all_results: Dict[str, Any],
                                  save_name: Optional[str] = None) -> str:
        """종합 분석 리포트 생성"""
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Comprehensive Batch Size Analysis Report: Adam vs AdamABS', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 그리드 레이아웃 설정 (4x3)
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        datasets = list(all_results.keys())
        batch_sizes = [64, 128, 256]
        optimizers = ['Adam', 'AdamABS']
        
        # 1. 전체 성능 비교 (상단 2x3)
        for dataset_idx, dataset_name in enumerate(datasets):
            ax = fig.add_subplot(gs[0, dataset_idx])
            dataset_results = all_results[dataset_name]
            
            # 배치 사이즈별 성능 비교
            for opt_idx, optimizer in enumerate(optimizers):
                accuracies = []
                available_batches = []
                
                for batch_size in batch_sizes:
                    batch_key = f"batch_{batch_size}"
                    if batch_key in dataset_results and optimizer in dataset_results[batch_key]:
                        acc = dataset_results[batch_key][optimizer].get('best_val_acc', 0)
                        accuracies.append(acc)
                        available_batches.append(batch_size)
                
                if accuracies:
                    color = self.colors.get(optimizer, 'gray')
                    ax.plot(available_batches, accuracies, 
                           marker='o', linewidth=2, markersize=6,
                           color=color, label=optimizer, alpha=0.8)
            
            ax.set_title(f'{dataset_name}', fontweight='bold')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(batch_sizes)
        
        # 2. 효율성 분석 (중간 행)
        ax_efficiency = fig.add_subplot(gs[1, :])
        
        # 모든 실험의 정확도 vs 시간 산점도
        for dataset_name, dataset_results in all_results.items():
            for batch_key, batch_results in dataset_results.items():
                batch_size = int(batch_key.split('_')[1])
                for optimizer, results in batch_results.items():
                    if 'best_val_acc' in results and 'total_training_time' in results:
                        acc = results['best_val_acc']
                        time = results['total_training_time'] / 60  # 분 단위
                        
                        color = self.colors.get(optimizer, 'gray')
                        marker_size = 50 + (batch_size - 64) * 2  # 배치 사이즈에 따른 마커 크기
                        
                        ax_efficiency.scatter(time, acc, s=marker_size, c=color, alpha=0.7,
                                           label=f'{optimizer}' if dataset_name == datasets[0] and batch_key == 'batch_64' else "",
                                           edgecolors='black', linewidth=0.5)
        
        ax_efficiency.set_xlabel('Training Time (minutes)')
        ax_efficiency.set_ylabel('Best Validation Accuracy (%)')
        ax_efficiency.set_title('Efficiency Analysis: Accuracy vs Training Time\n(Larger markers = Larger batch size)', fontweight='bold')
        ax_efficiency.legend()
        ax_efficiency.grid(True, alpha=0.3)
        
        # 3. 통계 요약 테이블 (하단)
        ax_stats = fig.add_subplot(gs[2:, :])
        ax_stats.axis('off')
        
        # 통계 데이터 수집
        stats_data = []
        for dataset_name, dataset_results in all_results.items():
            for batch_key, batch_results in dataset_results.items():
                batch_size = batch_key.split('_')[1]
                for optimizer, results in batch_results.items():
                    if 'best_val_acc' in results:
                        stats_data.append([
                            dataset_name,
                            batch_size,
                            optimizer,
                            f"{results['best_val_acc']:.2f}%",
                            f"{results.get('total_training_time', 0)/60:.1f}min"
                        ])
        
        if stats_data:
            # 테이블 생성
            table = ax_stats.table(cellText=stats_data,
                                 colLabels=['Dataset', 'Batch Size', 'Optimizer', 'Best Accuracy', 'Training Time'],
                                 cellLoc='center',
                                 loc='center',
                                 bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # 헤더 스타일링
            for i in range(5):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # AdamABS 행 강조
            for i, row in enumerate(stats_data, 1):
                if row[2] == 'AdamABS':
                    for j in range(5):
                        table[(i, j)].set_facecolor('#E8F5E8')
        
        # 파일 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'batch_size_comprehensive_report_{timestamp}.png'
        
        save_path = os.path.join(self.batch_results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 종합 분석 리포트 저장: {save_path}")
        return save_path


if __name__ == "__main__":
    # 간단한 테스트
    print("ExperimentVisualizer 테스트")
    print("="*60)
    
    # 더미 실험 결과 생성
    dummy_results = {
        'Adam': {
            'best_val_acc': 95.2,
            'final_train_acc': 96.8,
            'final_val_acc': 95.0,
            'total_training_time': 120.5,
            'avg_epoch_time': 12.05,
            'total_epochs_trained': 10,
            'training_history': {
                'train_loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05],
                'train_acc': [85, 88, 90, 92, 94, 95, 96, 96.5, 96.8, 96.8],
                'val_loss': [0.6, 0.4, 0.25, 0.18, 0.15, 0.13, 0.12, 0.11, 0.11, 0.12],
                'val_acc': [82, 86, 89, 91, 93, 94, 95, 95.2, 95.0, 95.0],
                'lr_history': [0.001] * 10,
                'epoch_times': [12] * 10
            }
        },
        'AdamABS': {
            'best_val_acc': 95.8,
            'final_train_acc': 97.2,
            'final_val_acc': 95.5,
            'total_training_time': 115.2,
            'avg_epoch_time': 11.52,
            'total_epochs_trained': 10,
            'training_history': {
                'train_loss': [0.45, 0.28, 0.18, 0.14, 0.11, 0.09, 0.07, 0.06, 0.055, 0.05],
                'train_acc': [86, 89, 91, 93, 95, 96, 96.8, 97, 97.2, 97.2],
                'val_loss': [0.55, 0.38, 0.23, 0.17, 0.14, 0.12, 0.11, 0.105, 0.11, 0.115],
                'val_acc': [84, 87, 90, 92, 94, 95.2, 95.8, 95.6, 95.5, 95.5],
                'lr_history': [0.001] * 10,
                'epoch_times': [11.5] * 10
            }
        }
    }
    
    # Visualizer 테스트
    visualizer = ExperimentVisualizer('./test_results')
    
    print("\n더미 데이터로 시각화 테스트...")
    saved_files = visualizer.generate_all_visualizations(dummy_results, "Test_Dataset")
    
    print(f"\n✅ 테스트 완료!")
    print(f"생성된 파일들:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")