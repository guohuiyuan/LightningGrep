#!/bin/bash
# ============================================
# LightningGrep SFT 训练脚本
# A100 单卡一键启动
# ============================================

set -e  # 出错即退出

echo "============================================"
echo "LightningGrep SFT Training"
echo "============================================"

# 检查 GPU
echo "[1/4] 检查 GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未检测到 nvidia-smi，请确认 GPU 环境"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 安装依赖
echo "[2/4] 安装依赖..."
pip install -q torch transformers peft accelerate datasets bitsandbytes tqdm
echo "✓ 依赖安装完成"
echo ""

# 检查数据
echo "[3/4] 检查数据..."
TRAIN_DATA="data/code_search/sft_all_train.json"
VAL_DATA="data/code_search/sft_all_val.json"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据不存在: $TRAIN_DATA"
    exit 1
fi
if [ ! -f "$VAL_DATA" ]; then
    echo "❌ 验证数据不存在: $VAL_DATA"
    exit 1
fi

TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
VAL_COUNT=$(python -c "import json; print(len(json.load(open('$VAL_DATA'))))")
echo "✓ 训练集: $TRAIN_COUNT 条"
echo "✓ 验证集: $VAL_COUNT 条"
echo ""

# 开始训练
echo "[4/4] 开始训练..."
echo "============================================"
echo ""

# 模型会自动从 HuggingFace 下载 (~3.5GB)
python src/training/sft_qlora.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir outputs/sft_v2 \
    --model_name Qwen/Qwen3-1.7B \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --max_length 4096 \
    --lora_r 64 \
    --lora_alpha 128 \
    --learning_rate 2e-4 \
    --early_stopping_patience 3

echo ""
echo "============================================"
echo "✓ 训练完成！模型保存在: outputs/sft_v2"
echo "============================================"
