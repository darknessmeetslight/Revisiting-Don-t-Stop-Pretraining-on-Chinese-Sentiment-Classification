# 基于领域自适应微调的 BERT 中文情感分类

本项目研究**领域自适应继续预训练**对中文情感分类任务效果的影响。在公开数据集 ChnSentiCorp 上以 `bert-base-chinese` 作为 baseline,对比单独 TAPT、单独 DAPT、以及 DAPT+TAPT 叠加三种继续预训练策略相对于 baseline 的提升幅度,所有实验均使用多个随机种子重复以扣除 seed 噪声。

> TAPT (Task-Adaptive Pretraining) — 在下游任务自身的训练集文本上继续做 MLM。
> DAPT (Domain-Adaptive Pretraining) — 在同领域更大规模的无标签语料上继续做 MLM。

## 项目结构

```
NLP/
├── config/                     # 各实验的 YAML 配置(7 份)
├── datasets/
│   ├── ChnSentiCorp/parquet/   # 下游任务数据集(9600/1200/1200)
│   └── online_shopping_10_cats/ # DAPT 领域语料(~6.3 万条电商评论)
├── src/
│   ├── data.py                 # 数据集加载与分词
│   ├── model.py                # 分类器构建(AutoModelForSequenceClassification)
│   ├── train_classifier.py     # 分类 fine-tune + 多 seed 聚合
│   ├── continue_pretrain.py    # MLM 继续预训练(TAPT/DAPT 共用)
│   ├── utils.py                # 项目路径、随机种子、日志
│   └── scripts/                # 一次性脚本(数据集下载、校验)
├── models/                     # 继续预训练产物 + HF 缓存
├── outputs/                    # 训练日志、TensorBoard、metrics/aggregate
└── pyproject.toml              # 依赖与 CUDA 12.4 PyTorch 索引
```

## 环境配置

依赖管理使用 [uv](https://github.com/astral-sh/uv)。Python 版本固定为 3.11(见 `.python-version`),PyTorch 锁在 CUDA 12.4 wheel(见 `pyproject.toml` 中 `[tool.uv.sources]`),适配宿主机 NVIDIA 驱动支持的最高 CUDA 版本。

```bash
# 1. 安装 uv(若未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步依赖(读取 uv.lock 创建 .venv/)
uv sync

# 3. 验证 GPU 可用
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 数据集准备

**ChnSentiCorp**(下游任务):

```bash
uv run python -m src.scripts.download_chnsenticorp
```

脚本会从 Hugging Face Hub 下载三个 parquet 分片到 `datasets/ChnSentiCorp/parquet/`。

**online_shopping_10_cats**(DAPT 语料):

从 [SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip) 手动下载并解压,把 `online_shopping_10_cats.csv`(约 11 MB)放到 `datasets/online_shopping_10_cats/` 目录下即可。

## 实验设计

所有实验的 fine-tune 阶段共用相同超参(`lr=2e-5`, `batch=32`, `epochs=3`, `bf16`, `max_seq_length=128`),仅起点模型不同。继续预训练阶段使用 `lr=5e-5`、`mlm_probability=0.15`,TAPT 跑 3 epoch、DAPT 跑 1 epoch(因为 DAPT 语料约为 TAPT 的 6.5 倍,单 epoch 总步数已超过 TAPT 三 epoch)。

为扣除随机种子带来的噪声,每个实验都用 `seeds: [42, 1337, 2026]` 跑 3 次,脚本会自动聚合并产出 `outputs/<实验名>/aggregate.json`(包含 `per_seed`、`mean`、`std`)。如需修改种子数量,只需编辑对应 YAML 的 `seeds` 字段。

四组对比实验:

| 实验 | 是否继续预训练 | 起点模型 | 配置文件 |
| --- | --- | --- | --- |
| baseline   | 否               | `bert-base-chinese`       | `config/baseline.yaml` |
| TAPT       | 任务文本 MLM     | `bert-base-chinese`       | `config/tapt.yaml` + `config/tapt_finetune.yaml` |
| DAPT       | 领域语料 MLM     | `bert-base-chinese`       | `config/dapt.yaml` + `config/dapt_finetune.yaml` |
| DAPT+TAPT  | 先 DAPT 再 TAPT  | DAPT 产出的同 seed 模型   | `config/dapt_tapt.yaml` + `config/dapt_tapt_finetune.yaml` |

## 实验复现

在项目根目录依次执行(每条命令内部都会跑完 3 个 seed):

```bash
# baseline
uv run python -m src.train_classifier --config config/baseline.yaml

# TAPT
uv run python -m src.continue_pretrain --config config/tapt.yaml
uv run python -m src.train_classifier --config config/tapt_finetune.yaml

# DAPT
uv run python -m src.continue_pretrain --config config/dapt.yaml
uv run python -m src.train_classifier --config config/dapt_finetune.yaml

# DAPT + TAPT(依赖上一步 DAPT 产物 ./models/dapt-s{seed})
uv run python -m src.continue_pretrain --config config/dapt_tapt.yaml
uv run python -m src.train_classifier --config config/dapt_tapt_finetune.yaml
```

跑完后产物落点:

- 单 seed 测试指标:`outputs/<实验名>-s<seed>/metrics.json`
- 聚合指标(mean ± std):`outputs/<实验名>/aggregate.json`
- TensorBoard 日志:`outputs/<实验名>-s<seed>/tensorboard/`
- 训练日志:`outputs/<实验名>-s<seed>/logs/train.log`

如需查看训练曲线:

```bash
uv run tensorboard --logdir outputs
```

## 实验结果

下游任务指标基于 ChnSentiCorp 测试集(1200 条),报告 3 个 seed(42 / 1337 / 2026)的均值 ± 样本标准差,百分制保留两位小数。

| 实验        | Accuracy        | F1 (binary)     | F1 (macro)      | Precision       | Recall          |
| ----------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| baseline    | 94.00 ± 0.00    | 94.06 ± 0.03    | 94.00 ± 0.00    | 94.33 ± 0.47    | 93.80 ± 0.53    |
| TAPT        | 94.03 ± 0.32    | 94.07 ± 0.28    | 94.03 ± 0.32    | 94.63 ± 0.79    | 93.53 ± 0.25    |
| DAPT        | 94.36 ± 0.27    | 94.36 ± 0.27    | 94.36 ± 0.27    | 95.66 ± 0.49    | 93.09 ± 0.44    |
| DAPT+TAPT   | **94.69 ± 0.29**| **94.71 ± 0.27**| **94.69 ± 0.29**| **95.75 ± 0.72**| 93.70 ± 0.19    |

完整 per-seed 数据见 `outputs/<实验名>/aggregate.json`。
