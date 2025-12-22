# LightningGrep ⚡

> **首个开源的并行检索小模型**
>
> The First Open-Source Parallel Retrieval Model for Small LLMs

## 🎯 项目简介

LightningGrep 是一个开源的**并行检索工具模型**，作为大模型的专精检索助手：

- **并行搜索**：一次发出多个独立查询，减少交互轮数
- **智能判断**：识别何时可以并行，何时必须串行
- **精准定位**：返回相关文档 + 行号，不是直接回答问题
- **轻量高效**：1.7B 小模型，可本地部署

### 定位

```
大模型（规划推理）──调用──→ LightningGrep（专精检索）──返回──→ 相关文档位置
                              ↑
                         我们做的是这个
```

### 核心能力

```
查询：找到两个杂志的创办时间信息

模型输出：
<think>两个杂志独立，可以并行查询</think>
<search>A杂志创办时间 ## B杂志创办时间</search>  ← 并行搜索

[环境返回搜索结果]

<think>找到相关文档</think>
<result>
  - Arthur's Magazine: lines [0]
  - First for Women: lines [1]
</result>
```

**注意**：我们返回的是文档位置，不是答案。大模型拿到位置后自己推理。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python scripts/download_data.py
```

### 3. 合成训练数据

```bash
# 设置 SiliconFlow API Key（或 OpenAI）
export SILICONFLOW_API_KEY="your-api-key"

# 测试 Prompt
python src/data_synthesis/synthesize.py --dry-run

# 合成 5000 条数据（支持断点续传）
python src/data_synthesis/synthesize.py --limit 5000 --output raw_5k.json

# 如果中断，用 --resume 继续
python src/data_synthesis/synthesize.py --limit 5000 --output raw_5k.json --resume
```

### 4. 转换为 SFT 格式

```bash
# 转换并划分 90% 训练 / 10% 验证
python src/data_synthesis/convert_to_sft.py data/synthetic/raw_5k.json --split 0.9
```

### 5. 训练模型（QLoRA）

```bash
python src/training/sft_qlora.py \
    --train_data data/synthetic/raw_5k_train.json \
    --val_data data/synthetic/raw_5k_val.json \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --mask_info \
    --output_dir outputs/sft_v1
```

### 6. 评测

```bash
# TODO: 评测脚本
python src/evaluation/eval.py --model outputs/sft_v1 --dataset hotpotqa
```

## 📁 项目结构

```
LightningGrep/
├── data/
│   ├── raw/                 # HotpotQA 原始数据
│   └── synthetic/           # 合成数据
├── src/
│   ├── data_synthesis/      # 数据合成
│   │   ├── synthesize.py    # 合成脚本
│   │   ├── convert_to_sft.py # 格式转换
│   │   └── synthesis_prompt.py # Prompt 定义
│   ├── training/            # 训练代码
│   │   └── sft_qlora.py     # QLoRA SFT 训练
│   └── evaluation/          # 评测代码（TODO）
├── scripts/
│   └── download_data.py     # 数据下载
├── research-plan.md         # 研究计划
└── requirements.txt
```

## � 研究计划

### 目标

受 [SWE-grep](https://www.cognition.ai/blog/swe-grep) 启发，用 **1.7B 小模型** 实现并行检索能力。

> ⚠️ 注意：这不是 SWE-grep 的复现（他们未开源），而是受其启发的独立实现。

### 阶段规划

| 阶段 | 内容 | 状态 |
|------|------|------|
| **V1** | 基础并行检索 | 🔄 进行中 |
| **V2** | 行号级别召回 | ⏳ 待开始 |
| **V3** | 置信度动态终止 | ⏳ 待开始 |

### V1 详细计划

- [x] 研究计划制定
- [x] 数据合成 Prompt 设计
- [x] 数据合成脚本开发
- [x] SFT 训练脚本开发（QLoRA）
- [ ] 合成 5000 条 SFT 数据（⏳ 进行中...）
- [ ] 转换为 SFT 格式（填充真实内容）
- [ ] QLoRA SFT 训练
- [ ] 评测脚本开发
- [ ] HotpotQA 评测
- [ ] RL 训练脚本开发
- [ ] RL 训练 + 评测

## 评测指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Recall** | 召回率（返回的位置覆盖 GT）| > 80% |
| **Precision** | 精确率（返回的位置相关性）| > 70% |
| **Avg Rounds** | 平均搜索轮数 | < 2.5 |
| **Parallel Rate** | 并行搜索比例 | > 50% |

### 对比基线

| 模型 | 参数量 | HotpotQA EM | 开源 |
|------|--------|-------------|------|
| GAP | 3B | 42.5% | ❌ |
| Search-R1 | 7B | 37.6% | ✅ |
| ParallelSearch | 7B | +2.9% | ❌ |
| **LightningGrep (目标)** | **1.7B** | **>35%** | **✅** |

## 🔬 方法论

### 训练流程

```
SFT（格式 + 基础能力）
  │
  │  使用合成数据，教模型：
  │  - 输出格式（<think>, <search>, <result>）
  │  - 并行/串行判断
  │  - 基本搜索策略
  │
  ▼
RL（策略优化）
  │
  │  与真实搜索环境交互，优化：
  │  - 搜索精准度
  │  - 结果筛选能力
  │  - 效率（减少轮数）
  │
  ▼
最终模型
```

### 核心技术

受 [SWE-grep](https://www.cognition.ai/blog/swe-grep) 博客启发：

- **Policy Gradient** + Per-Sequence Importance Sampling
- **Leave-One-Out Baseline**
- **Weighted F1 奖励** (β=0.5)
- **Mask 环境 Token**：训练时不学习 `<information>` 的生成

详见 [research-plan.md](research-plan.md)

## 📚 参考

- [SWE-grep Blog](https://www.cognition.ai/blog/swe-grep) - Windsurf 的并行检索方法（未开源）
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - 开源 RL 检索模型
- [ParallelSearch](https://arxiv.org/abs/2508.09303) - 并行查询分解（代码未公开）
- [HotpotQA](https://hotpotqa.github.io/) - 多跳问答数据集

## 📝 License

MIT

## 🤝 贡献

欢迎 Issue 和 PR！
