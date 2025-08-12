"""
빠른 논문용 그래프 생성 스크립트
실험 결과 JSON 파일을 읽어서 고품질 그래프 생성

사용법:
python quick_paper_plots.py results/tiny_imagenet_20250813_results.json
또는
python quick_paper_plots.py  # 최신 결과 파일 자동 탐색

Author: AI Research
Date: 2025
"""

import json
import os
import sys
import glob
from paper_visualizer import create_paper_plots

def find_latest_results_file(results_dir='./results'):
    """
    가장 최신 실험 결과 JSON 파일 찾기
    
    Args:
        results_dir: 결과 폴더 경로
        
    Returns:
        str: 최신 JSON 파일 경로
    """
    json_files = glob.glob(os.path.join(results_dir, '*_results.json'))
    if not json_files:
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {results_dir}")
    
    # 수정 시간 기준으로 최신 파일 선택
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def load_experiment_results(json_path):
    """
    JSON 파일에서 실험 결과 로드
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        tuple: (experiment_results, dataset_name)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터셋 이름 추출
    if 'dataset' in data:
        dataset_name = data['dataset']
    else:
        # 파일명에서 추출
        basename = os.path.basename(json_path)
        if 'mnist' in basename.lower():
            dataset_name = 'MNIST'
        elif 'cifar' in basename.lower():
            dataset_name = 'CIFAR-10'
        elif 'tiny' in basename.lower():
            dataset_name = 'Tiny ImageNet'
        else:
            dataset_name = 'Unknown Dataset'
    
    # 실험 결과 추출
    if 'results' in data:
        experiment_results = data['results']
    else:
        # JSON 전체가 결과인 경우
        experiment_results = data
    
    return experiment_results, dataset_name

def main():
    """메인 함수"""
    print("📊 논문용 고품질 그래프 생성기")
    print("=" * 50)
    
    # 입력 파일 결정
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if not os.path.exists(json_path):
            print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
            return
    else:
        try:
            json_path = find_latest_results_file()
            print(f"🔍 최신 결과 파일 자동 선택: {os.path.basename(json_path)}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    
    try:
        # 결과 로드
        print(f"📖 결과 파일 읽는 중: {os.path.basename(json_path)}")
        experiment_results, dataset_name = load_experiment_results(json_path)
        
        print(f"📊 데이터셋: {dataset_name}")
        print(f"🔧 옵티마이저: {list(experiment_results.keys())}")
        
        # 논문용 그래프 생성
        create_paper_plots(experiment_results, dataset_name)
        
        print("\n✅ 논문용 그래프 생성 완료!")
        print("📁 저장 위치: ./paper_results/")
        print("📄 PNG와 PDF 파일이 모두 생성되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()