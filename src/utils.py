"""通用工具:项目路径、随机种子、日志。"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

# 项目根目录(本文件位于 src/,parents[1] 即项目根)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 把 HuggingFace 缓存指向项目内 models/hf_cache,下载的预训练模型会落到
# models/(已在 .gitignore 中排除),避免污染用户 home 目录
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))


def set_seed(seed: int) -> None:
    """为 Python、NumPy、PyTorch(CPU + 所有 GPU) 统一设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """创建 logger,同时输出到控制台和可选的日志文件。重复调用幂等。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
