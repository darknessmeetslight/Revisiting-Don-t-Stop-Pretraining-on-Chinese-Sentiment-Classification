"""在指定文本语料上继续做 MLM 预训练 (TAPT / DAPT 共用)。

通过 config 里的 data.source 字段切换不同语料:
  - chnsenticorp_train -> 取 ChnSentiCorp 训练集 text 列 (TAPT)
  - online_shopping_10_cats -> (DAPT,后续接入)

用法(项目根目录):
    uv run python -m src.continue_pretrain --config config/tapt.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .data import load_chnsenticorp
from .utils import PROJECT_ROOT, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    return p.parse_args()


def load_corpus(source: str) -> Dataset:
    """根据 source 标识加载继续预训练用的无标签文本,返回只含 text 列的 Dataset。"""
    if source == "chnsenticorp_train":
        # TAPT:直接用 ChnSentiCorp 训练集的文本(标签不参与 MLM)
        ds = load_chnsenticorp()
        return ds["train"].select_columns(["text"])
    # DAPT:online_shopping_10_cats 等领域语料后续在此追加分支
    raise ValueError(f"未知的数据源: {source}")


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name: str = cfg["experiment_name"]
    seed: int = cfg["seed"]

    # 输出布局:
    #   models/<exp_name>/   最终 MLM 模型,供下游 fine-tune 加载
    #   outputs/<exp_name>/  训练日志 + TB + Trainer 中间 checkpoint
    model_out_dir = PROJECT_ROOT / cfg["output"]["models_dir"] / exp_name
    run_dir = PROJECT_ROOT / "outputs" / exp_name
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"
    for d in (model_out_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # transformers 5.x 的 TensorBoardCallback 只读这个环境变量,
    # 不读 TrainingArguments.logging_dir,必须在 Trainer 实例化前设置。
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    logger = setup_logger(exp_name, log_dir / "train.log")
    logger.info(f"实验名: {exp_name}, seed={seed}")
    logger.info(f"配置: {json.dumps(cfg, ensure_ascii=False)}")

    set_seed(seed)

    base_name = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForMaskedLM.from_pretrained(base_name)

    corpus = load_corpus(cfg["data"]["source"])
    logger.info(f"原始语料样本数: {len(corpus)}")

    max_len = cfg["data"]["max_seq_length"]

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
        )

    tokenized = corpus.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="MLM 分词",
    )

    # 动态 MLM:每个 batch 现场随机 15% mask (BERT 原论文比例)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg["data"]["mlm_probability"],
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        seed=seed,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        learning_rate=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        num_train_epochs=cfg["train"]["num_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy=cfg["train"]["save_strategy"],
        save_total_limit=cfg["train"]["save_total_limit"],
        fp16=cfg["train"]["fp16"],
        bf16=cfg["train"]["bf16"],
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=collator,
    )

    logger.info("开始 MLM 继续预训练...")
    trainer.train()

    # 把最终模型保存到 models/<exp_name>/(供下游 fine-tune 加载)
    logger.info(f"保存模型到: {model_out_dir}")
    trainer.save_model(str(model_out_dir))
    tokenizer.save_pretrained(str(model_out_dir))

    logger.info(f"完成。MLM 模型: {model_out_dir}, 日志/TB: {run_dir}")


if __name__ == "__main__":
    main()
