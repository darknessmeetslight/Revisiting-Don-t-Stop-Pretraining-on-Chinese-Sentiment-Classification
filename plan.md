# 项目计划：基于领域自适应微调的 BERT 中文情感分类改进研究

> 课程：NLP（Level 0 课程作业，占总分 80%）
> 截止时间：2026-06-26（Week 16 Friday 18:00），今日 2026-05-24，剩约 5 周
> 小组：3 人
> 硬件：本地 4 × RTX 4080 SUPER
> 主参考论文：BERT (Devlin et al., NAACL 2019)
> 次参考论文：Don't Stop Pretraining (Gururangan et al., ACL 2020)

---

## 0. 关于"使用论文代码"的说明

我们 **不跑** [bert/](bert/) 里 Google 原版 TF 1.x 代码（环境难配、年代久远、对评分无加分），只在需要时**阅读**以下文件理解模型结构：

- [bert/modeling.py](bert/modeling.py)：理解 BERT 内部 encoder
- [bert/run_classifier.py](bert/run_classifier.py)：理解 fine-tune 输入构造
- [bert/create_pretraining_data.py](bert/create_pretraining_data.py)：理解 MLM mask 策略

我们用 **HuggingFace Transformers（PyTorch）**——这是 BERT 论文代码的现代等价实现，加上论文产出的预训练权重 `bert-base-chinese`，等价于在使用 BERT 论文的成果。

---

## 1. 作业是做什么的

来自 [docx/Ch1_Introduction.pptx](docx/Ch1_Introduction.pptx)：

- **任务等级**：Level 0（最高分档）—— "An improvement based on current frameworks"。
- **必含内容**：主框架核心思想 + 改进点 + 相关工作 + 理论 + framework 描述 + 源码 + **baseline + ablation study** + 结论。
- **论文结构**：Title / Abstract (~250 词) / Keywords (3–5) / Introduction（背景、缺点、本文如何解决、主要贡献）/ Related Works / Framework（含图）/ Experiments（设置、数据、结果、表与图）/ Conclusion & Discussion。
- **评分原则**："don't worry about whether the idea could improve the evaluation metrics. The key is a suitable idea." —— 改进即使没拉高指标，**只要 idea 合适、实验扎实、写作清晰**就拿高分。
- **学术诚信**：可用 ChatGPT 翻译，idea 与实验必须自做；有 AI 检测。
- **提交**：附 **成员贡献声明**。

---

## 2. 研究方向与实验思路

### 2.1 一句话概括

> 在中文情感分类任务上复现 BERT baseline，并通过"在情感评论领域/任务文本上继续预训练 BERT"（DAPT + TAPT）来提升模型对中文评论文本的适应能力，通过对比与消融实验验证改进的有效性。

### 2.2 方法对比

| 方法 | 含义 | 备注 |
|---|---|---|
| **BERT Baseline** | 直接用 `bert-base-chinese` 在 ChnSentiCorp 上 fine-tune | 复现起点 |
| **+ TAPT** | 先用 ChnSentiCorp 训练集文本（去标签）做 MLM 继续预训练，再 fine-tune | 任务自适应，成本低 |
| **+ DAPT** | 先用更大的中文电商评论无标签语料（online_shopping_10_cats）做 MLM，再 fine-tune | 领域自适应 |
| **+ DAPT + TAPT**（主推改进） | DAPT 之后再 TAPT，最后 fine-tune | 论文里效果通常最好 |

### 2.3 框架流程图（文字版，后续在论文里画正式图）

```text
[大规模通用中文语料]  -> bert-base-chinese (已开源,直接下载)
                              |
                              v
[online_shopping_10_cats] -> 继续 MLM 预训练 (DAPT)   <- 改进点①
                              |
                              v
[ChnSentiCorp 训练集文本]  -> 继续 MLM 预训练 (TAPT)   <- 改进点②
                              |
                              v
[ChnSentiCorp 带标签]     -> 监督 fine-tune (CLS + Linear)
                              |
                              v
                         情感预测 (正面/负面)
```

### 2.4 消融实验设计

| 实验 | 目的 |
|---|---|
| Baseline vs Baseline + TAPT | 验证 TAPT 单独是否有效 |
| Baseline vs Baseline + DAPT | 验证 DAPT 单独是否有效 |
| TAPT vs DAPT vs DAPT+TAPT | 验证两者叠加是否更优 |
| DAPT + 不同语料规模（如 10k / 30k / 60k） | 验证领域数据量影响 |
| TAPT epoch 数（1 / 3 / 10） | 验证继续预训练步数影响 |

### 2.5 评价指标

- **Accuracy**
- **F1-score**（binary on positive class，必要时也报 macro F1）
- 每个配置至少跑 **3 个随机种子**，取均值 ± 标准差。

---

## 3. 已确认的关键决策

| 项目 | 决策 |
|---|---|
| 代码框架 | HuggingFace Transformers (PyTorch) |
| 改进方法 | TAPT + DAPT 都做（含 DAPT+TAPT 组合） |
| 预训练模型 | **优先 `bert-base-chinese`**；若获取困难，回退 `hfl/chinese-bert-wwm-ext` |
| 任务数据集 | ChnSentiCorp（已下载） |
| DAPT 领域语料 | `online_shopping_10_cats`（约 6 万电商评论） |
| Google bert/ 原代码 | 不跑，仅作参考 |
| 论文语言 | 先写中文版，后续视情况转英文（流畅英文可加 3–4 分） |
| 参考文献 | 直接复用 BERT 论文 + Don't Stop Pretraining 论文的参考文献 |
| 错误分析 | 暂不做 |
| 硬件 | 本地 4 × RTX 4080 SUPER |
| 环境管理 | uv + Python 3.11 + `.venv` |
| 日志 | Python logging（文件 + 控制台）+ TensorBoard |

---

## 4. 项目文件结构

```
NLP/                                # 项目根目录(后续推 GitHub 前会改名)
├── plan.md                         # 项目规划(本文件)
├── pyproject.toml                  # uv 项目配置(待生成)
├── uv.lock                         # uv 依赖锁定(待生成)
├── .gitignore                      # 排除 bert/, datasets/, models/, outputs/, .venv/ 等
├── .venv/                          # uv 创建的虚拟环境(不上传 GitHub)
│
├── bert/                           # Google TF 1.x 原版代码,仅作参考阅读,不上传 GitHub
│
├── datasets/                       # 数据集(不上传 GitHub,提供下载脚本)
│   ├── ChnSentiCorp/               # 已就绪
│   │   ├── parquet/{train,validation,test}.parquet
│   │   └── dataset_info.json
│   └── online_shopping_10_cats/    # DAPT 语料,待下载
│
├── models/                         # 预训练 / 继续预训练模型权重(跨实验共享)
│   ├── bert-base-chinese/          # HF 基础权重(也可让 HF 默认缓存到 ~/.cache)
│   ├── tapt/                       # TAPT 继续预训练 checkpoint
│   ├── dapt/                       # DAPT 继续预训练 checkpoint
│   └── dapt_tapt/                  # DAPT+TAPT 叠加 checkpoint
│
├── config/                         # 实验配置(yaml),换 config 即换实验
│   ├── baseline.yaml
│   ├── tapt.yaml
│   ├── dapt.yaml
│   ├── dapt_tapt.yaml
│   └── ablation_*.yaml             # 数据规模/epoch 等消融
│
├── src/                            # 我们自己实现的代码
│   ├── __init__.py
│   ├── data.py                     # 数据集加载与预处理
│   ├── model.py                    # 模型封装(SeqCls / MaskedLM)
│   ├── train_classifier.py         # 分类 fine-tune (baseline + 继续预训练后微调)
│   ├── continue_pretrain.py        # MLM 继续预训练 (TAPT/DAPT 共用,差别在数据)
│   ├── evaluate.py                 # 评估(Accuracy / F1 / 混淆矩阵)
│   ├── utils.py                    # 随机种子、日志初始化等
│   └── scripts/                    # 一次性脚本
│       ├── download_online_shopping.py
│       └── verify_dataset.py
│
├── outputs/                        # 单次实验输出(按实验名隔离)
│   └── {experiment_name}/
│       ├── checkpoint-*/           # 分类模型 checkpoint
│       ├── logs/train.log          # 文本日志
│       ├── tensorboard/            # TB 事件文件
│       └── metrics.json            # 最终指标 (Acc/F1, 含多种子均值方差)
│
├── paper/                          # 论文撰写(先中文,后续可能转英文)
│   ├── main.md                     # 论文正文
│   ├── figures/                    # 图表 PDF/PNG
│   └── tables/                     # 结果表 CSV
│
└── docx/                           # 说明文档(不上传 GitHub 或上传部分)
    ├── bert_paper_summary.md
    ├── dont_stop_pretraining_paper_summary.md
    ├── level0_coursework_template.md
    ├── BERT...pdf
    ├── Don't Stop Pretraining...pdf
    └── Ch1_Introduction.pptx
```

**`models/` vs `outputs/` 的分工**
- `models/`：**跨实验共享的可复用权重**（基础模型、继续预训练产物）。
- `outputs/`：**单次实验的产物**（fine-tune 后的分类模型、训练日志、TB、指标），按 `experiment_name` 隔离。

**日志与监控**
- 训练曲线：HuggingFace `Trainer` 的 `report_to=["tensorboard"]`，事件写到 `outputs/{exp}/tensorboard/`。
- 文本日志：Python `logging` + `FileHandler` → `outputs/{exp}/logs/train.log`，同时打印到控制台。
- 启动 TensorBoard：`tensorboard --logdir outputs --port 6006`。

**`.gitignore` 要排除**
```
.venv/
datasets/
models/
outputs/
bert/
docx/*.pdf
docx/*.pptx
__pycache__/
*.pyc
.DS_Store
```

---

## 5. 环境配置（uv + Python 3.11）

### 5.1 Python 版本推荐：**3.11**

理由：
- transformers 4.40+ / PyTorch 2.4+ / datasets 完整支持 3.11，无兼容问题。
- 3.11 比 3.10 在解释器层面快 10–25%，对大量数据预处理有可感收益。
- 3.12 也可以，但部分老牌 NLP 工具（如旧版 jieba）在 3.12 偶有问题；3.11 更稳。
- 不选 3.9：HuggingFace 生态正逐步放弃 3.9 支持。

### 5.2 安装命令（在 `/home/chifujin/project/NLP/` 下执行）

```bash
# 1. 如未装 uv,先安装(已装跳过)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 安装后可能需要 source 一下 shell rc 或重开终端

# 2. 进入项目目录,创建 Python 3.11 虚拟环境
#    uv 会自动下载缺失的 Python 版本
cd /home/chifujin/project/NLP
uv venv .venv --python 3.11

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装 PyTorch (CUDA 12.1, 适配 RTX 40 系)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. 安装其他依赖
uv pip install \
    transformers \
    datasets \
    accelerate \
    scikit-learn \
    pandas \
    pyarrow \
    tensorboard \
    pyyaml \
    tqdm

# 6. 验证 GPU 可见(应输出 True 和 4)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU 数:', torch.cuda.device_count())"
```

### 5.3 多卡说明

- 前期开发：单卡跑（`CUDA_VISIBLE_DEVICES=0 python ...`），BERT-base + ChnSentiCorp 几分钟一轮。
- 大规模 DAPT（online_shopping_10_cats 6 万条）：用 `accelerate launch --multi_gpu` 或 `torchrun --nproc_per_node=4`，加速近 4 倍。
- 消融批量跑：4 卡可以 4 个实验并行（每卡跑一个种子），节省墙钟时间。

