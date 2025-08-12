"""
논문용 고품질 시각화 모듈
Adam 논문 스타일의 전문적인 수렴 그래프 생성

Author: AI Research
Date: 2025
"""

import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
import os
from datetime import datetime

# 논문용 고품질 설정
plt.rcParams.update({
    'font.size': 12,
    'font.family': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.unicode_minus': False
})

class PaperQualityVisualizer:
    """논문용 고품질 시각화 클래스"""
    
    def __init__(self, results_dir: str = './paper_results'):
        """
        Args:
            results_dir: 결과 저장 디렉토리
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 논문용 전문적인 색상 팔레트 (색약자 친화적 + 흑백 인쇄 대응)
        self.colors = {
            'RMSProp': '#1f77b4',      # 깊은 파랑 (기본 알고리즘)
            'RMSPropABS': '#ff7f0e',   # 주황 (ABS 변형)
            'Adam': '#2ca02c',         # 초록 (대표 알고리즘)  
            'AdamW': '#d62728',        # 빨강 (가중치 감소 버전)
            'AdamABS': '#9467bd'       # 보라 (우리 제안 방법)
        }
        
        # 선 스타일 (흑백 인쇄 대응)
        self.line_styles = {
            'RMSProp': '-',           # 실선
            'RMSPropABS': '--',       # 대시
            'Adam': '-',              # 실선
            'AdamW': '-.',            # 대시-점
            'AdamABS': ':'            # 점선
        }
        
        # 마커 스타일
        self.markers = {
            'RMSProp': 'o',           # 원
            'RMSPropABS': 's',        # 사각형
            'Adam': '^',              # 삼각형
            'AdamW': 'D',             # 다이아몬드
            'AdamABS': 'v'            # 역삼각형
        }
    
    def create_convergence_plot(self, experiment_results: Dict[str, Any], 
                               dataset_name: str, 
                               plot_type: str = 'val_acc',
                               title_override: Optional[str] = None,
                               save_name: Optional[str] = None) -> str:
        """
        논문용 단일 수렴 곡선 플롯 생성
        
        Args:
            experiment_results: 실험 결과 딕셔너리
            dataset_name: 데이터셋 이름
            plot_type: 'val_acc', 'train_loss', 'val_loss', 'train_acc'
            title_override: 커스텀 제목
            save_name: 저장 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        # 고해상도 figure 설정
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        # 옵티마이저 순서 고정 (논문 일관성)
        optimizer_order = ['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS']
        available_optimizers = [opt for opt in optimizer_order if opt in experiment_results]
        
        # 플롯 타입별 설정
        plot_configs = {
            'val_acc': {
                'ylabel': 'Validation Accuracy (%)',
                'title': f'{dataset_name} - Validation Accuracy',
                'legend_loc': 'lower right',
                'scale': 'linear'
            },
            'train_loss': {
                'ylabel': 'Training Loss',
                'title': f'{dataset_name} - Training Loss',
                'legend_loc': 'upper right', 
                'scale': 'log'
            },
            'val_loss': {
                'ylabel': 'Validation Loss',
                'title': f'{dataset_name} - Validation Loss',
                'legend_loc': 'upper right',
                'scale': 'log'
            },
            'train_acc': {
                'ylabel': 'Training Accuracy (%)',
                'title': f'{dataset_name} - Training Accuracy',
                'legend_loc': 'lower right',
                'scale': 'linear'
            }
        }
        
        config = plot_configs[plot_type]
        
        # 데이터 플롯
        for opt_name in available_optimizers:
            if opt_name not in experiment_results:
                continue
                
            results = experiment_results[opt_name]
            history = results['training_history']
            
            # 데이터 추출
            if plot_type in history:
                y_data = history[plot_type]
            else:
                print(f"Warning: {plot_type} not found for {opt_name}")
                continue
            
            x_data = range(1, len(y_data) + 1)  # 1부터 시작하는 에포크
            
            # 플롯 그리기
            ax.plot(x_data, y_data,
                   color=self.colors.get(opt_name, '#808080'),
                   linestyle=self.line_styles.get(opt_name, '-'),
                   marker=self.markers.get(opt_name, 'o'),
                   markevery=max(1, len(y_data)//10),  # 마커 간격 조정
                   markersize=6,
                   linewidth=2.5,
                   alpha=0.9,
                   label=opt_name)
        
        # 축 및 제목 설정
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontweight='bold')
        
        if title_override:
            ax.set_title(title_override, fontweight='bold', pad=20)
        else:
            ax.set_title(config['title'], fontweight='bold', pad=20)
        
        # 축 스케일 설정
        if config['scale'] == 'log':
            ax.set_yscale('log')
        
        # 그리드 설정 (미묘하게)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 범례 설정 (논문 스타일)
        legend = ax.legend(loc=config['legend_loc'], 
                          frameon=True, 
                          fancybox=True, 
                          shadow=True,
                          ncol=1,
                          fontsize=11,
                          title='Optimizers',
                          title_fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 축 범위 최적화
        ax.margins(x=0.02, y=0.05)
        
        # 테두리 스타일
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # 레이아웃 최적화
        plt.tight_layout()
        
        # 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"{dataset_name.lower().replace(' ', '_')}_{plot_type}_{timestamp}"
        
        save_path = os.path.join(self.results_dir, f"{save_name}.png")
        plt.savefig(save_path, 
                   dpi=300,              # 고해상도
                   bbox_inches='tight',   # 여백 최소화
                   facecolor='white',     # 배경 흰색
                   edgecolor='none',      # 테두리 없음
                   format='png')
        
        # PDF 버전도 저장 (벡터 그래픽)
        pdf_path = os.path.join(self.results_dir, f"{save_name}.pdf")
        plt.savefig(pdf_path,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   format='pdf')
        
        plt.close()
        
        print(f"✅ 논문용 그래프 저장 완료:")
        print(f"   PNG: {save_path}")
        print(f"   PDF: {pdf_path}")
        
        return save_path
    
    def create_comparison_grid(self, experiment_results: Dict[str, Any],
                             dataset_name: str,
                             save_name: Optional[str] = None) -> str:
        """
        4개 메트릭 비교 그리드 (2x2)
        
        Args:
            experiment_results: 실험 결과
            dataset_name: 데이터셋 이름
            save_name: 저장 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{dataset_name} - Optimizer Comparison', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 플롯 설정
        plots = [
            ('train_loss', 'Training Loss', 'upper right', 'log'),
            ('val_loss', 'Validation Loss', 'upper right', 'log'),
            ('train_acc', 'Training Accuracy (%)', 'lower right', 'linear'),
            ('val_acc', 'Validation Accuracy (%)', 'lower right', 'linear')
        ]
        
        optimizer_order = ['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS']
        available_optimizers = [opt for opt in optimizer_order if opt in experiment_results]
        
        for idx, (metric, ylabel, legend_loc, scale) in enumerate(plots):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # 데이터 플롯
            for opt_name in available_optimizers:
                if opt_name not in experiment_results:
                    continue
                    
                results = experiment_results[opt_name]
                history = results['training_history']
                
                if metric in history:
                    y_data = history[metric]
                    x_data = range(1, len(y_data) + 1)
                    
                    ax.plot(x_data, y_data,
                           color=self.colors.get(opt_name, '#808080'),
                           linestyle=self.line_styles.get(opt_name, '-'),
                           marker=self.markers.get(opt_name, 'o'),
                           markevery=max(1, len(y_data)//8),
                           markersize=5,
                           linewidth=2.2,
                           alpha=0.9,
                           label=opt_name)
            
            # 축 설정
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(ylabel, fontweight='bold', pad=15)
            
            if scale == 'log':
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
            ax.set_axisbelow(True)
            
            # 범례 (첫 번째 플롯에만)
            if idx == 0:
                legend = ax.legend(loc=legend_loc, 
                                 frameon=True, 
                                 fancybox=True, 
                                 shadow=True,
                                 fontsize=10)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 저장
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"{dataset_name.lower().replace(' ', '_')}_comparison_grid_{timestamp}"
        
        save_path = os.path.join(self.results_dir, f"{save_name}.png")
        plt.savefig(save_path, 
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        pdf_path = os.path.join(self.results_dir, f"{save_name}.pdf")
        plt.savefig(pdf_path,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   format='pdf')
        
        plt.close()
        
        print(f"✅ 비교 그리드 저장 완료:")
        print(f"   PNG: {save_path}")
        print(f"   PDF: {pdf_path}")
        
        return save_path
    
    def create_single_metric_focus(self, experiment_results: Dict[str, Any],
                                  dataset_name: str,
                                  metric: str = 'val_acc',
                                  custom_title: Optional[str] = None) -> str:
        """
        논문 메인 그래프용 단일 메트릭 집중 플롯
        
        Args:
            experiment_results: 실험 결과
            dataset_name: 데이터셋 이름  
            metric: 'val_acc', 'train_loss' 등
            custom_title: 커스텀 제목
            
        Returns:
            str: 저장된 파일 경로
        """
        # 더 큰 사이즈로 메인 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        optimizer_order = ['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS']
        available_optimizers = [opt for opt in optimizer_order if opt in experiment_results]
        
        # 메트릭별 설정
        if metric == 'val_acc':
            ylabel = 'Validation Accuracy (%)'
            title = custom_title or f'{dataset_name} Validation Accuracy Convergence'
            legend_loc = 'lower right'
            scale = 'linear'
        elif metric == 'train_loss':
            ylabel = 'Training Loss'
            title = custom_title or f'{dataset_name} Training Loss Convergence'
            legend_loc = 'upper right'
            scale = 'log'
        elif metric == 'val_loss':
            ylabel = 'Validation Loss'
            title = custom_title or f'{dataset_name} Validation Loss Convergence'
            legend_loc = 'upper right'
            scale = 'log'
        else:
            ylabel = metric.replace('_', ' ').title()
            title = custom_title or f'{dataset_name} {ylabel} Convergence'
            legend_loc = 'best'
            scale = 'linear'
        
        # 데이터 플롯 (더 굵은 선)
        for opt_name in available_optimizers:
            if opt_name not in experiment_results:
                continue
                
            results = experiment_results[opt_name]
            history = results['training_history']
            
            if metric in history:
                y_data = history[metric]
                x_data = range(1, len(y_data) + 1)
                
                ax.plot(x_data, y_data,
                       color=self.colors.get(opt_name, '#808080'),
                       linestyle=self.line_styles.get(opt_name, '-'),
                       marker=self.markers.get(opt_name, 'o'),
                       markevery=max(1, len(y_data)//12),  # 마커 수 줄임
                       markersize=8,              # 큰 마커
                       linewidth=3.5,             # 두꺼운 선
                       alpha=0.9,
                       label=opt_name)
        
        # 축 및 제목 설정 (큰 폰트)
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
        
        if scale == 'log':
            ax.set_yscale('log')
        
        # 그리드
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.0)
        ax.set_axisbelow(True)
        
        # 범례 (큰 폰트)
        legend = ax.legend(loc=legend_loc,
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          fontsize=14,
                          title='Optimizers',
                          title_fontsize=15,
                          columnspacing=1.2,
                          handlelength=2.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        # 축 스타일
        ax.tick_params(axis='both', which='major', labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{dataset_name.lower().replace(' ', '_')}_{metric}_main_{timestamp}"
        
        save_path = os.path.join(self.results_dir, f"{save_name}.png")
        plt.savefig(save_path,
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        
        pdf_path = os.path.join(self.results_dir, f"{save_name}.pdf") 
        plt.savefig(pdf_path,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none',
                   format='pdf')
        
        plt.close()
        
        print(f"✅ 메인 그래프 저장 완료:")
        print(f"   PNG: {save_path}")
        print(f"   PDF: {pdf_path}")
        
        return save_path


def create_paper_plots(experiment_results: Dict[str, Any], dataset_name: str):
    """
    논문용 모든 플롯 생성 (간편 함수)
    
    Args:
        experiment_results: 실험 결과
        dataset_name: 데이터셋 이름
    """
    visualizer = PaperQualityVisualizer()
    
    print(f"\n📊 {dataset_name} 논문용 시각화 생성 중...")
    
    # 1. 메인 validation accuracy 그래프
    visualizer.create_single_metric_focus(
        experiment_results, 
        dataset_name, 
        'val_acc',
        f'{dataset_name} - Validation Accuracy Convergence'
    )
    
    # 2. 메인 training loss 그래프  
    visualizer.create_single_metric_focus(
        experiment_results,
        dataset_name,
        'train_loss', 
        f'{dataset_name} - Training Loss Convergence'
    )
    
    # 3. 4개 메트릭 비교 그리드
    visualizer.create_comparison_grid(experiment_results, dataset_name)
    
    print(f"✅ {dataset_name} 논문용 시각화 완료!")
    print(f"📁 저장 위치: {visualizer.results_dir}")


if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("📊 논문용 시각화 모듈 테스트")
    
    # 더미 실험 결과 생성
    epochs = 40
    dummy_results = {}
    
    optimizers = ['RMSProp', 'RMSPropABS', 'Adam', 'AdamW', 'AdamABS']
    
    for opt in optimizers:
        # 가상의 수렴 곡선 생성
        train_loss = [2.0 * np.exp(-0.1 * i) + 0.1 * np.random.random() for i in range(epochs)]
        val_loss = [2.2 * np.exp(-0.08 * i) + 0.15 * np.random.random() for i in range(epochs)]
        train_acc = [20 + 60 * (1 - np.exp(-0.12 * i)) + 2 * np.random.random() for i in range(epochs)]
        val_acc = [15 + 55 * (1 - np.exp(-0.1 * i)) + 3 * np.random.random() for i in range(epochs)]
        
        dummy_results[opt] = {
            'training_history': {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
        }
    
    # 테스트 시각화 생성
    create_paper_plots(dummy_results, "Test Dataset")
    print("✅ 테스트 완료!")