"""ChnSentiCorp 数据集加载与分词。"""
from __future__ import annotations

from datasets import Dataset, DatasetDict

from .utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "datasets" / "ChnSentiCorp" / "parquet"
SPLITS = ("train", "validation", "test")


def load_chnsenticorp() -> DatasetDict:
    """加载 train / validation / test 三个 parquet 划分。"""
    return DatasetDict({
        split: Dataset.from_parquet(str(DATA_DIR / f"{split}.parquet"))
        for split in SPLITS
    })


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """对 text 列做分词。这里不做 padding,留给 collator 动态 padding 以节省显存。"""
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="分词中",
    )
