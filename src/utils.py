"""随机性控制、设备选择和文件系统操作等通用工具。"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """统一设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int = 0) -> None:
    """为完整训练流程启用尽可能确定性的计算设置。

    这些设置能够显著提高重复运行的一致性，但部分 CUDA 稀疏算子仍
    可能产生极小的浮点数波动，因此这里使用 ``warn_only=True``。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name: str = "auto") -> torch.device:
    """解析设备名称；``auto`` 会优先选择可用的 CUDA GPU。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    """创建目录及缺失的父目录，并返回对应的 ``Path`` 对象。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    """保存模型检查点，并在保存前自动创建父目录。"""
    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    torch.save(payload, checkpoint_path)
