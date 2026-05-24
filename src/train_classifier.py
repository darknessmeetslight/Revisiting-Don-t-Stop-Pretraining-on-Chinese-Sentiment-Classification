"""在 ChnSentiCorp 上微调 BERT 分类器。

用法(在项目根目录执行):
    uv run python -m src.train_classifier --config config/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .data import load_chnsenticorp, tokenize_dataset
from .model import build_classifier
from .utils import PROJECT_ROOT, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    return p.parse_args()


def compute_metrics(eval_pred):
    """评估时计算 accuracy / f1 / precision / recall。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="binary", pos_label=1),
        "recall": recall_score(labels, preds, average="binary", pos_label=1),
    }


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name: str = cfg["experiment_name"]
    seed: int = cfg["seed"]
    out_dir = PROJECT_ROOT / cfg["output"]["base_dir"] / exp_name
    log_dir = out_dir / "logs"
    tb_dir = out_dir / "tensorboard"
    for d in (log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # transformers 5.x 的 TensorBoardCallback 不读 TrainingArguments.logging_dir,
    # 只读这个环境变量。必须在 Trainer() 实例化前设置。
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    logger = setup_logger(exp_name, log_dir / "train.log")
    logger.info(f"实验名: {exp_name}, seed={seed}")
    logger.info(f"配置: {json.dumps(cfg, ensure_ascii=False)}")

    set_seed(seed)

    tokenizer, model = build_classifier(
        cfg["model"]["name_or_path"], cfg["model"]["num_labels"]
    )

    ds = load_chnsenticorp()
    ds = tokenize_dataset(ds, tokenizer, cfg["data"]["max_seq_length"])
    logger.info("分词后数据集大小: " + ", ".join(
        f"{k}={len(v)}" for k, v in ds.items()
    ))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["eval_batch_size"],
        learning_rate=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        num_train_epochs=cfg["train"]["num_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        logging_dir=str(tb_dir),
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=cfg["train"]["eval_strategy"],
        save_strategy=cfg["train"]["save_strategy"],
        save_total_limit=cfg["train"]["save_total_limit"],
        load_best_model_at_end=cfg["train"]["load_best_model_at_end"],
        metric_for_best_model=cfg["train"]["metric_for_best_model"],
        greater_is_better=cfg["train"]["greater_is_better"],
        fp16=cfg["train"]["fp16"],
        bf16=cfg["train"]["bf16"],
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info("开始训练...")
    trainer.train()

    logger.info("在测试集上评估...")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    logger.info(f"测试集指标: {test_metrics}")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"完成。输出目录: {out_dir}")


if __name__ == "__main__":
    main()
