"""数据集加载与分词:ChnSentiCorp (有标签下游任务) + online_shopping_10_cats (DAPT 无标签语料)。"""
from __future__ import annotations

from datasets import Dataset, DatasetDict

from .utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "datasets" / "ChnSentiCorp" / "parquet"
SPLITS = ("train", "validation", "test")

ONLINE_SHOPPING_CSV = (
    PROJECT_ROOT / "datasets" / "online_shopping_10_cats" / "online_shopping_10_cats.csv"
)


def load_chnsenticorp() -> DatasetDict:
    """加载 train / validation / test 三个 parquet 划分。"""
    return DatasetDict({
        split: Dataset.from_parquet(str(DATA_DIR / f"{split}.parquet"))
        for split in SPLITS
    })


def load_online_shopping() -> Dataset:
    """加载 online_shopping_10_cats 电商评论语料,只保留 review 列并改名为 text。

    DAPT 只需要无标签文本,所以丢掉 cat / label 两列,空评论(NaN)直接过滤掉。
    """
    # 用 datasets 自带的 csv loader,自动处理首行 BOM
    ds = Dataset.from_csv(str(ONLINE_SHOPPING_CSV))
    ds = ds.filter(lambda r: r["review"] is not None and len(r["review"].strip()) > 0)
    ds = ds.select_columns(["review"]).rename_column("review", "text")
    return ds


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
