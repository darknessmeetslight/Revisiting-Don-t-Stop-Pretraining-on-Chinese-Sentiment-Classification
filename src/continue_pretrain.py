"""在指定文本语料上继续做 MLM 预训练 (TAPT / DAPT 共用),支持多 seed 循环。

通过 config 里的 data.source 字段切换不同语料:
  - chnsenticorp_train      -> ChnSentiCorp 训练集 text 列 (TAPT)
  - online_shopping_10_cats -> 6 万条电商评论 text 列     (DAPT)

config 里 seeds 字段是 list,每个 seed 跑一次,产物落到 models/<exp>-s<seed>/。
model.name_or_path 可用 {seed} 占位符(DAPT+TAPT 阶段从 ./models/dapt-s{seed} 起步会用到)。
如果目标模型目录已存在则跳过该 seed,方便中断后继续跑。

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

from .data import load_chnsenticorp, load_online_shopping
from .utils import PROJECT_ROOT, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    return p.parse_args()


def load_corpus(source: str) -> Dataset:
    """根据 source 标识加载继续预训练用的无标签文本,返回只含 text 列的 Dataset。"""
    if source == "chnsenticorp_train":
        ds = load_chnsenticorp()
        return ds["train"].select_columns(["text"])
    if source == "online_shopping_10_cats":
        return load_online_shopping()
    raise ValueError(f"未知的数据源: {source}")


def run_one_seed(cfg: dict, seed: int, base_exp_name: str, corpus: Dataset) -> None:
    """跑单个 seed 的 MLM 继续预训练。"""
    exp_name = f"{base_exp_name}-s{seed}"
    model_out_dir = PROJECT_ROOT / cfg["output"]["models_dir"] / exp_name
    run_dir = PROJECT_ROOT / "outputs" / exp_name
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"

    # 已有最终模型(config.json 是 HF 模型保存的标志文件)则跳过
    if (model_out_dir / "config.json").exists():
        print(f"[skip] {exp_name}: 模型已存在 {model_out_dir}")
        return

    for d in (model_out_dir, log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # transformers 5.x 的 TensorBoardCallback 只读这个环境变量,必须在 Trainer() 前设置
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    # 每个 seed 用唯一 logger name,避免 setup_logger 复用同名 logger 时 handler 累积
    logger = setup_logger(exp_name, log_dir / "train.log")
    logger.info(f"实验名: {exp_name}, seed={seed}")
    logger.info(f"配置: {json.dumps(cfg, ensure_ascii=False)}")

    set_seed(seed)

    # 解析 model.name_or_path 的 {seed} 占位
    base_name = cfg["model"]["name_or_path"].format(seed=seed)
    logger.info(f"加载基础模型: {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForMaskedLM.from_pretrained(base_name)

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

    # 动态 MLM:每个 batch 现场随机 15% mask
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

    logger.info(f"保存模型到: {model_out_dir}")
    trainer.save_model(str(model_out_dir))
    tokenizer.save_pretrained(str(model_out_dir))
    logger.info(f"完成 seed={seed}。模型: {model_out_dir}")


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_exp_name = cfg["experiment_name"]
    # 兼容 seed: 42 单值 和 seeds: [42, 1337, 2026] 列表
    seeds = cfg.get("seeds")
    if seeds is None:
        seeds = [cfg["seed"]]

    # 语料只加载一次,所有 seed 共用(分词在循环内做,因为 tokenizer 可能随 base_model 变化)
    corpus = load_corpus(cfg["data"]["source"])
    print(f"原始语料样本数: {len(corpus)},将跑 {len(seeds)} 个 seed: {seeds}")

    for seed in seeds:
        run_one_seed(cfg, seed, base_exp_name, corpus)


if __name__ == "__main__":
    main()
