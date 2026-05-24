"""模型工厂。"""
from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_classifier(model_name_or_path: str, num_labels: int = 2):
    """加载 tokenizer 和带序列分类头的模型。

    同时适配 `bert-base-chinese` 和备选的 `hfl/chinese-bert-wwm-ext`。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )
    return tokenizer, model
