"""
訓練日誌記錄器（支援 TensorBoard / CSV / Console）

用於記錄訓練過程中的各種指標：
- Reward / Loss / Q-value
- Lagrangian 乘子與約束違規
- RUDDER 回填統計
- Episode 統計
"""

from __future__ import annotations
from typing import Dict, Optional
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import csv


class TrainingLogger:
    """
    訓練日誌記錄器
    
    Args:
        log_dir: 日誌保存目錄
        experiment_name: 實驗名稱
        enable_tensorboard: 是否啟用 TensorBoard
        enable_csv: 是否啟用 CSV 記錄
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        enable_tensorboard: bool = True,
        enable_csv: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 創建目錄
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.enable_tensorboard = enable_tensorboard
        if self.enable_tensorboard:
            tb_dir = self.log_dir / experiment_name / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
        else:
            self.tb_writer = None
        
        # CSV
        self.enable_csv = enable_csv
        if self.enable_csv:
            csv_dir = self.log_dir / experiment_name / 'csv'
            csv_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = csv_dir / 'training_log.csv'
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = None  # 延遲初始化（根據第一條記錄確定列名）
            self.csv_fieldnames = None
        else:
            self.csv_file = None
            self.csv_writer = None
        
        # 步數計數
        self.global_step = 0
    
    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = '',
    ) -> None:
        """
        記錄標量指標
        
        Args:
            metrics: 指標字典（key: 指標名稱, value: 標量值）
            step: 全局步數（若為 None 則使用內部計數器）
            prefix: 指標名稱前綴（例如 'train/', 'eval/'）
        """
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        # TensorBoard
        if self.enable_tensorboard and self.tb_writer is not None:
            for key, value in metrics.items():
                full_key = f"{prefix}{key}" if prefix else key
                self.tb_writer.add_scalar(full_key, value, step)
        
        # CSV
        if self.enable_csv and self.csv_file is not None:
            # 延遲初始化 CSV writer（根據第一條記錄確定列名）
            if self.csv_writer is None:
                self.csv_fieldnames = ['step'] + list(metrics.keys())
                self.csv_writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=self.csv_fieldnames
                )
                self.csv_writer.writeheader()
            
            # 寫入記錄
            row = {'step': step, **metrics}
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # 立即寫入磁碟
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """記錄文本信息（僅 TensorBoard）"""
        if self.enable_tensorboard and self.tb_writer is not None:
            if step is None:
                step = self.global_step
            self.tb_writer.add_text(tag, text, step)
    
    def log_histogram(
        self,
        tag: str,
        values,
        step: Optional[int] = None,
    ) -> None:
        """記錄直方圖（僅 TensorBoard）"""
        if self.enable_tensorboard and self.tb_writer is not None:
            if step is None:
                step = self.global_step
            self.tb_writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """關閉日誌記錄器"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
    
    def __del__(self) -> None:
        """析構函數（確保資源釋放）"""
        self.close()

