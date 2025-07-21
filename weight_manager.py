"""
Weight 저장/로드 관리 모듈
훈련된 모델 가중치를 저장하고 다음 학습에서 재사용할 수 있는 기능 제공

Author: AI Research
Date: 2025
"""

import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
import shutil


@dataclass
class WeightCheckpoint:
    """Weight 체크포인트 메타데이터"""
    model_name: str
    dataset_name: str
    optimizer_name: str
    epoch: int
    best_val_acc: float
    training_time: float
    timestamp: str
    model_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    file_hash: str
    file_size: int


class WeightManager:
    """모델 가중치 저장/로드 관리 클래스"""
    
    def __init__(self, base_dir: str = "./weights"):
        """
        Args:
            base_dir: 가중치 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 저장 파일
        self.metadata_file = self.base_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()
        
        print(f"✅ WeightManager 초기화 완료")
        print(f"   저장 디렉토리: {self.base_dir.absolute()}")
        print(f"   기존 체크포인트: {len(self.metadata)}개")
    
    def _load_metadata(self) -> List[WeightCheckpoint]:
        """메타데이터 로드"""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoints = []
            for item in data:
                checkpoints.append(WeightCheckpoint(**item))
            
            return checkpoints
        except Exception as e:
            print(f"⚠️  메타데이터 로드 실패: {e}")
            return []
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            data = [asdict(checkpoint) for checkpoint in self.metadata]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 메타데이터 저장 실패: {e}")
    
    def _generate_checkpoint_name(self, model_name: str, dataset_name: str, 
                                 optimizer_name: str, epoch: int) -> str:
        """체크포인트 파일명 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{dataset_name}_{optimizer_name}_ep{epoch:03d}_{timestamp}.pth"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _find_best_checkpoint_for_combination(self, model_name: str, dataset_name: str, optimizer_name: str) -> Optional[WeightCheckpoint]:
        """특정 모델/데이터셋/옵티마이저 조합의 기존 최고 성능 체크포인트 찾기"""
        matching_checkpoints = [
            checkpoint for checkpoint in self.metadata
            if (checkpoint.model_name == model_name and 
                checkpoint.dataset_name == dataset_name and 
                checkpoint.optimizer_name == optimizer_name)
        ]
        
        if not matching_checkpoints:
            return None
        
        # 가장 높은 정확도의 체크포인트 반환
        return max(matching_checkpoints, key=lambda x: x.best_val_acc)
    
    def _delete_checkpoints_for_combination(self, model_name: str, dataset_name: str, optimizer_name: str):
        """특정 모델/데이터셋/옵티마이저 조합의 모든 체크포인트 삭제"""
        checkpoints_to_delete = []
        
        # 삭제할 체크포인트 찾기
        for i, checkpoint in enumerate(self.metadata):
            if (checkpoint.model_name == model_name and 
                checkpoint.dataset_name == dataset_name and 
                checkpoint.optimizer_name == optimizer_name):
                checkpoints_to_delete.append(i)
        
        # 파일 삭제
        deleted_files = 0
        for i in reversed(checkpoints_to_delete):  # 역순으로 삭제하여 인덱스 문제 방지
            checkpoint = self.metadata[i]
            
            # 파일 패턴으로 찾아서 삭제
            file_patterns = [
                f"{model_name}_{dataset_name}_{optimizer_name}_BEST.pth",
                f"{model_name}_{dataset_name}_{optimizer_name}_ep*_*.pth"
            ]
            
            for pattern in file_patterns:
                matching_files = list(self.base_dir.glob(pattern))
                for file_path in matching_files:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_files += 1
            
            # 메타데이터에서 제거
            self.metadata.pop(i)
        
        # 메타데이터 저장
        if deleted_files > 0:
            self._save_metadata()
            print(f"   🗑️  기존 파일 {deleted_files}개 삭제")
    
    def save_checkpoint(self, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       model_name: str,
                       dataset_name: str,
                       optimizer_name: str,
                       epoch: int,
                       best_val_acc: float,
                       training_time: float,
                       training_history: Dict[str, List],
                       additional_info: Optional[Dict[str, Any]] = None,
                       force_save: bool = False) -> Optional[str]:
        """
        체크포인트 저장 (더 좋은 성능일 때만)
        
        Args:
            model: 모델
            optimizer: 옵티마이저
            scheduler: 스케줄러 (선택)
            model_name: 모델 이름
            dataset_name: 데이터셋 이름
            optimizer_name: 옵티마이저 이름
            epoch: 현재 에포크
            best_val_acc: 최고 검증 정확도
            training_time: 훈련 시간
            training_history: 훈련 히스토리
            additional_info: 추가 정보
            force_save: 강제 저장 여부 (성능 비교 무시)
        
        Returns:
            Optional[str]: 저장된 파일 경로 (저장되지 않으면 None)
        """
        # 기존 체크포인트 찾기
        existing_checkpoint = self._find_best_checkpoint_for_combination(
            model_name, dataset_name, optimizer_name
        )
        
        # 성능 비교 (force_save가 False인 경우에만)
        if not force_save and existing_checkpoint is not None:
            if best_val_acc <= existing_checkpoint.best_val_acc:
                print(f"⚠️  현재 성능 ({best_val_acc:.2f}%) ≤ 기존 최고 성능 ({existing_checkpoint.best_val_acc:.2f}%)")
                print(f"   체크포인트 저장하지 않음")
                return None
            else:
                print(f"🚀 성능 향상 감지: {existing_checkpoint.best_val_acc:.2f}% → {best_val_acc:.2f}%")
                print(f"   기존 체크포인트를 새로운 최고 성능으로 대체합니다")
                # 기존 파일들 삭제
                self._delete_checkpoints_for_combination(model_name, dataset_name, optimizer_name)
        
        # 고정된 파일명 생성 (타임스탬프 없이)
        filename = f"{model_name}_{dataset_name}_{optimizer_name}_BEST.pth"
        file_path = self.base_dir / filename
        
        # 저장할 데이터 구성
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': model_name,
            'dataset_name': dataset_name,
            'optimizer_name': optimizer_name,
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'class_name': model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            },
            'optimizer_config': {
                'class_name': optimizer.__class__.__name__,
                'param_groups': optimizer.param_groups
            }
        }
        
        # 스케줄러 정보 추가
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['scheduler_config'] = {
                'class_name': scheduler.__class__.__name__,
                'last_epoch': scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else -1
            }
        
        # 추가 정보 병합
        if additional_info:
            checkpoint_data['additional_info'] = additional_info
        
        try:
            # 체크포인트 파일 저장
            torch.save(checkpoint_data, file_path)
            
            # 파일 정보 계산
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            
            # 메타데이터 생성
            checkpoint_meta = WeightCheckpoint(
                model_name=model_name,
                dataset_name=dataset_name,
                optimizer_name=optimizer_name,
                epoch=epoch,
                best_val_acc=best_val_acc,
                training_time=training_time,
                timestamp=checkpoint_data['timestamp'],
                model_config=checkpoint_data['model_config'],
                optimizer_config={
                    'class_name': checkpoint_data['optimizer_config']['class_name'],
                    'lr': optimizer.param_groups[0]['lr'],
                    'params': {k: v for k, v in optimizer.param_groups[0].items() 
                              if k not in ['params']}
                },
                file_hash=file_hash,
                file_size=file_size
            )
            
            # 메타데이터 추가
            self.metadata.append(checkpoint_meta)
            self._save_metadata()
            
            print(f"✅ 최고 성능 체크포인트 저장 완료")
            print(f"   파일: {filename}")
            print(f"   크기: {file_size / (1024*1024):.1f} MB")
            print(f"   최고 정확도: {best_val_acc:.2f}%")
            print(f"   에포크: {epoch}")
            
            return str(file_path)
            
        except Exception as e:
            print(f"❌ 체크포인트 저장 실패: {e}")
            # 실패한 파일 제거
            if file_path.exists():
                file_path.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
        
        Returns:
            Dict: 로드된 체크포인트 데이터
        """
        file_path = Path(checkpoint_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        
        try:
            checkpoint_data = torch.load(file_path, map_location='cpu')
            
            print(f"✅ 체크포인트 로드 완료")
            print(f"   파일: {file_path.name}")
            print(f"   모델: {checkpoint_data['model_name']}")
            print(f"   데이터셋: {checkpoint_data['dataset_name']}")
            print(f"   옵티마이저: {checkpoint_data['optimizer_name']}")
            print(f"   에포크: {checkpoint_data['epoch']}")
            print(f"   최고 정확도: {checkpoint_data['best_val_acc']:.2f}%")
            
            return checkpoint_data
            
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            raise
    
    def find_checkpoints(self, model_name: Optional[str] = None,
                        dataset_name: Optional[str] = None,
                        optimizer_name: Optional[str] = None,
                        min_accuracy: Optional[float] = None) -> List[WeightCheckpoint]:
        """
        조건에 맞는 체크포인트 검색
        
        Args:
            model_name: 모델 이름 필터
            dataset_name: 데이터셋 이름 필터
            optimizer_name: 옵티마이저 이름 필터
            min_accuracy: 최소 정확도 필터
        
        Returns:
            List: 조건에 맞는 체크포인트 리스트
        """
        results = []
        
        for checkpoint in self.metadata:
            # 필터 적용
            if model_name and checkpoint.model_name != model_name:
                continue
            if dataset_name and checkpoint.dataset_name != dataset_name:
                continue
            if optimizer_name and checkpoint.optimizer_name != optimizer_name:
                continue
            if min_accuracy and checkpoint.best_val_acc < min_accuracy:
                continue
            
            results.append(checkpoint)
        
        # 정확도 순으로 정렬
        results.sort(key=lambda x: x.best_val_acc, reverse=True)
        
        return results
    
    def get_best_checkpoint(self, model_name: str, dataset_name: str) -> Optional[WeightCheckpoint]:
        """특정 모델/데이터셋의 최고 성능 체크포인트 반환"""
        checkpoints = self.find_checkpoints(model_name=model_name, dataset_name=dataset_name)
        return checkpoints[0] if checkpoints else None
    
    def list_checkpoints(self, detailed: bool = False):
        """저장된 체크포인트 목록 출력"""
        if not self.metadata:
            print("저장된 체크포인트가 없습니다.")
            return
        
        print(f"\n📋 저장된 체크포인트 ({len(self.metadata)}개)")
        print("=" * 100)
        
        if detailed:
            for i, checkpoint in enumerate(self.metadata, 1):
                print(f"{i:2d}. {checkpoint.model_name} | {checkpoint.dataset_name} | {checkpoint.optimizer_name}")
                print(f"     에포크: {checkpoint.epoch} | 정확도: {checkpoint.best_val_acc:.2f}%")
                print(f"     훈련시간: {checkpoint.training_time:.1f}초 | 저장시간: {checkpoint.timestamp[:19]}")
                print(f"     파라미터: {checkpoint.model_config.get('num_parameters', 'N/A'):,}")
                print(f"     파일크기: {checkpoint.file_size / (1024*1024):.1f} MB")
                print("-" * 80)
        else:
            print(f"{'No':<3} {'Model':<15} {'Dataset':<12} {'Optimizer':<10} {'Epoch':<6} {'Acc':<8} {'Time'}")
            print("-" * 80)
            
            for i, checkpoint in enumerate(self.metadata, 1):
                print(f"{i:<3} {checkpoint.model_name:<15} {checkpoint.dataset_name:<12} "
                      f"{checkpoint.optimizer_name:<10} {checkpoint.epoch:<6} "
                      f"{checkpoint.best_val_acc:>6.2f}% {checkpoint.training_time:>6.1f}s")
    
    def delete_checkpoint(self, checkpoint_index: int) -> bool:
        """체크포인트 삭제"""
        if not 0 <= checkpoint_index < len(self.metadata):
            print(f"❌ 잘못된 인덱스: {checkpoint_index}")
            return False
        
        checkpoint = self.metadata[checkpoint_index]
        
        # 파일 경로 추정 (파일이 존재하는지 확인)
        possible_files = list(self.base_dir.glob(f"*{checkpoint.optimizer_name}_ep{checkpoint.epoch:03d}_*.pth"))
        
        # 파일 삭제
        deleted_files = 0
        for file_path in possible_files:
            if file_path.exists():
                file_path.unlink()
                deleted_files += 1
        
        # 메타데이터에서 제거
        self.metadata.pop(checkpoint_index)
        self._save_metadata()
        
        print(f"✅ 체크포인트 삭제 완료 (파일 {deleted_files}개 삭제)")
        return True
    
    def cleanup_old_checkpoints(self, keep_best_n: int = 5):
        """오래된 체크포인트 정리 (성능 기준으로 상위 N개만 유지)"""
        if len(self.metadata) <= keep_best_n:
            print("정리할 체크포인트가 없습니다.")
            return
        
        # 정확도 기준으로 정렬
        sorted_checkpoints = sorted(self.metadata, key=lambda x: x.best_val_acc, reverse=True)
        
        # 삭제할 체크포인트들
        to_delete = sorted_checkpoints[keep_best_n:]
        
        deleted_count = 0
        for checkpoint in to_delete:
            # 파일 삭제
            possible_files = list(self.base_dir.glob(f"*{checkpoint.optimizer_name}_ep{checkpoint.epoch:03d}_*.pth"))
            for file_path in possible_files:
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
            
            # 메타데이터에서 제거
            self.metadata.remove(checkpoint)
        
        self._save_metadata()
        
        print(f"✅ 정리 완료: {deleted_count}개 파일 삭제, {keep_best_n}개 체크포인트 유지")


class ContinuousTrainer:
    """연속 훈련을 위한 헬퍼 클래스"""
    
    def __init__(self, weight_manager: WeightManager):
        """
        Args:
            weight_manager: WeightManager 인스턴스
        """
        self.weight_manager = weight_manager
        self.loaded_checkpoint = None
    
    def resume_from_checkpoint(self, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                             checkpoint_path: str) -> Tuple[int, Dict[str, List]]:
        """
        체크포인트에서 훈련 재개
        
        Args:
            model: 모델 (가중치가 로드될 예정)
            optimizer: 옵티마이저 (상태가 로드될 예정)
            scheduler: 스케줄러 (상태가 로드될 예정)
            checkpoint_path: 체크포인트 파일 경로
        
        Returns:
            Tuple: (시작 에포크, 훈련 히스토리)
        """
        # 체크포인트 로드
        self.loaded_checkpoint = self.weight_manager.load_checkpoint(checkpoint_path)
        
        # 모델 가중치 로드
        model.load_state_dict(self.loaded_checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드
        optimizer.load_state_dict(self.loaded_checkpoint['optimizer_state_dict'])
        
        # 스케줄러 상태 로드 (있는 경우)
        if scheduler is not None and 'scheduler_state_dict' in self.loaded_checkpoint:
            scheduler.load_state_dict(self.loaded_checkpoint['scheduler_state_dict'])
        
        start_epoch = self.loaded_checkpoint['epoch'] + 1
        training_history = self.loaded_checkpoint.get('training_history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'epoch_times': [], 'lr_history': []
        })
        
        print(f"📥 훈련 재개 준비 완료")
        print(f"   시작 에포크: {start_epoch}")
        print(f"   이전 최고 정확도: {self.loaded_checkpoint['best_val_acc']:.2f}%")
        
        return start_epoch, training_history
    
    def find_and_resume_best(self, model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                           model_name: str,
                           dataset_name: str,
                           optimizer_name: str) -> Optional[Tuple[int, Dict[str, List]]]:
        """
        최고 성능 체크포인트에서 자동 재개
        
        Args:
            model, optimizer, scheduler: 훈련 객체들
            model_name: 모델 이름
            dataset_name: 데이터셋 이름
            optimizer_name: 옵티마이저 이름
        
        Returns:
            Optional[Tuple]: (시작 에포크, 훈련 히스토리) 또는 None
        """
        # 특정 조합의 최고 성능 체크포인트 찾기
        best_checkpoint = self.weight_manager._find_best_checkpoint_for_combination(
            model_name, dataset_name, optimizer_name
        )
        
        if best_checkpoint is None:
            print(f"📝 {model_name}/{dataset_name}/{optimizer_name}의 기존 체크포인트가 없습니다. 처음부터 훈련을 시작합니다.")
            return None
        
        # 새로운 파일명 패턴으로 체크포인트 파일 찾기
        checkpoint_file = self.weight_manager.base_dir / f"{model_name}_{dataset_name}_{optimizer_name}_BEST.pth"
        
        if not checkpoint_file.exists():
            print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_file}")
            return None
        
        checkpoint_path = str(checkpoint_file)
        return self.resume_from_checkpoint(model, optimizer, scheduler, checkpoint_path)


if __name__ == "__main__":
    # 테스트
    print("WeightManager 테스트")
    print("=" * 60)
    
    # WeightManager 생성
    wm = WeightManager("./test_weights")
    
    # 더미 모델과 옵티마이저로 테스트
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 더미 훈련 히스토리
    training_history = {
        'train_loss': [0.8, 0.6, 0.4],
        'train_acc': [60.0, 70.0, 80.0],
        'val_loss': [0.7, 0.5, 0.3],
        'val_acc': [65.0, 75.0, 85.0],
        'epoch_times': [10.0, 9.5, 9.0],
        'lr_history': [0.001, 0.001, 0.001]
    }
    
    print("\n1. 체크포인트 저장 테스트")
    saved_path = wm.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        model_name="TestModel",
        dataset_name="TestDataset",
        optimizer_name="Adam",
        epoch=3,
        best_val_acc=85.0,
        training_time=28.5,
        training_history=training_history
    )
    
    print(f"\n2. 체크포인트 목록")
    wm.list_checkpoints()
    
    print(f"\n3. 체크포인트 검색 테스트")
    found = wm.find_checkpoints(optimizer_name="Adam")
    print(f"Adam 옵티마이저 체크포인트: {len(found)}개")
    
    print(f"\n4. 최고 성능 체크포인트")
    best = wm.get_best_checkpoint("TestModel", "TestDataset")
    if best:
        print(f"최고 성능: {best.best_val_acc:.2f}%")
    
    print(f"\n5. 연속 훈련 테스트")
    ct = ContinuousTrainer(wm)
    
    # 새 모델과 옵티마이저 생성
    new_model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 5))
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    result = ct.find_and_resume_best(new_model, new_optimizer, None, "TestModel", "TestDataset", "Adam")
    if result:
        start_epoch, history = result
        print(f"재개 에포크: {start_epoch}")
    
    # 테스트 파일 정리
    shutil.rmtree("./test_weights", ignore_errors=True)
    print(f"\n✅ 테스트 완료! 테스트 파일 정리됨")