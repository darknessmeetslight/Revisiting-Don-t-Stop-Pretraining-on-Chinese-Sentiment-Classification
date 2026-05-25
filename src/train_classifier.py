"""在 ChnSentiCorp 上微调 BERT 分类器,支持多 seed 循环 + 自动聚合 metrics。

config 里 seeds 字段是 list,每个 seed 跑一次:
  - 训练日志/产物/metrics 各自落到 outputs/<exp>-s<seed>/
  - 跑完所有 seed 后,在 outputs/<exp>/aggregate.json 写入每个 seed 的指标 + 均值 + 标准差

model.name_or_path 里可用 {seed} 占位符(继续预训练后的 fine-tune 用得到)。
单个 seed 的 metrics.json 已存在则跳过,方便中断后继续跑。

用法(项目根目录):
    uv run python -m src.train_classifier --config config/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .data import load_chnsenticorp, tokenize_dataset
from .model import build_classifier
from .utils import PROJECT_ROOT, set_seed, setup_logger

# 聚合时关心的指标键(test_ 前缀,排除 runtime/samples_per_second/epoch 等)
METRIC_KEYS = (
    "test_loss",
    "test_accuracy",
    "test_f1",
    "test_f1_macro",
    "test_precision",
    "test_recall",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    return p.parse_args()


def compute_metrics(eval_pred):
    """评估时计算 accuracy / f1(binary, pos=1) / f1_macro / precision / recall。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="binary", pos_label=1),
        "recall": recall_score(labels, preds, average="binary", pos_label=1),
    }


def run_one_seed(cfg: dict, seed: int, base_exp_name: str) -> dict | None:
    """跑单个 seed 的分类 fine-tune。返回 test_metrics dict;已跑过则读 metrics.json 返回。"""
    exp_name = f"{base_exp_name}-s{seed}"
    out_dir = PROJECT_ROOT / cfg["output"]["base_dir"] / exp_name
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists():
        print(f"[skip] {exp_name}: metrics.json 已存在,直接读取")
        with metrics_path.open(encoding="utf-8") as f:
            return json.load(f)

    log_dir = out_dir / "logs"
    tb_dir = out_dir / "tensorboard"
    for d in (log_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

    # transformers 5.x 的 TensorBoardCallback 只读这个环境变量,必须在 Trainer() 前设置
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)

    # 每个 seed 唯一 logger name,避免重复 handler
    logger = setup_logger(exp_name, log_dir / "train.log")
    logger.info(f"实验名: {exp_name}, seed={seed}")
    logger.info(f"配置: {json.dumps(cfg, ensure_ascii=False)}")

    set_seed(seed)

    # 解析 model.name_or_path 的 {seed} 占位(baseline 等无占位的串 .format 后保持不变)
    base_name = cfg["model"]["name_or_path"].format(seed=seed)
    logger.info(f"加载基础模型: {base_name}")
    tokenizer, model = build_classifier(base_name, cfg["model"]["num_labels"])

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

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"完成 seed={seed}。输出: {out_dir}")
    return test_metrics


def aggregate(per_seed: dict[str, dict], base_exp_name: str) -> None:
    """汇总多 seed 指标的均值 + 样本标准差,写入 outputs/<exp>/aggregate.json。"""
    agg_path = PROJECT_ROOT / "outputs" / base_exp_name / "aggregate.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_seed.values() if key in m]
        if not vals:
            continue
        mean[key] = statistics.fmean(vals)
        # n=1 时样本标准差未定义,fallback 0.0,便于聚合 JSON 结构稳定
        std[key] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    agg = {
        "experiment": base_exp_name,
        "seeds": list(per_seed.keys()),
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
    }
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"\n聚合结果写入: {agg_path}")
    print("均值:", json.dumps(mean, ensure_ascii=False, indent=2))
    print("标准差:", json.dumps(std, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_exp_name = cfg["experiment_name"]
    # 兼容 seed: 42 单值 和 seeds: [42, 1337, 2026] 列表
    seeds = cfg.get("seeds")
    if seeds is None:
        seeds = [cfg["seed"]]

    print(f"实验 {base_exp_name}: 将跑 {len(seeds)} 个 seed: {seeds}")

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        m = run_one_seed(cfg, seed, base_exp_name)
        if m is not None:
            per_seed[str(seed)] = m

    if len(per_seed) > 0:
        aggregate(per_seed, base_exp_name)


if __name__ == "__main__":
    main()
