"""
QLoRA SFT 训练脚本
用于训练并行检索模型

使用方法:
    python src/training/sft_qlora.py \
        --train_data data/code_search/sft_all_train.json \
        --val_data data/code_search/sft_all_val.json \
        --output_dir outputs/sft_v2
"""
import os
import sys
import json
import torch
import argparse
from typing import Dict, List
from dataclasses import dataclass, field


def check_environment():
    """检查训练环境"""
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    # Python 版本
    print(f"Python: {sys.version.split()[0]}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA 不可用！请检查 GPU 环境")
        sys.exit(1)
    
    # 检查依赖
    try:
        import transformers
        import peft
        import bitsandbytes
        import datasets
        print(f"Transformers: {transformers.__version__}")
        print(f"PEFT: {peft.__version__}")
        print("✓ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install transformers peft accelerate datasets bitsandbytes")
        sys.exit(1)
    
    print("=" * 50)
    print()


# 环境检查
check_environment()

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_sft_data(data_path: str) -> List[Dict]:
    """加载 SFT 数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_fc_messages(messages: List[Dict], tools: List[Dict] = None) -> str:
    """
    将 Function Calling 格式的 messages 转换为 Qwen chat 格式
    """
    parts = []
    
    # 添加 tools 定义（如果有）
    if tools:
        tools_str = json.dumps(tools, ensure_ascii=False)
        parts.append(f"<|im_start|>system\nYou are a code search agent. Available tools:\n{tools_str}<|im_end|>")
    
    for msg in messages:
        role = msg["role"]
        
        if role == "user":
            parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
        
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if tool_calls:
                # 有 tool calls 的 assistant 消息
                tc_str = json.dumps(tool_calls, ensure_ascii=False)
                if content:
                    parts.append(f"<|im_start|>assistant\n{content}\n<tool_calls>\n{tc_str}\n</tool_calls><|im_end|>")
                else:
                    parts.append(f"<|im_start|>assistant\n<tool_calls>\n{tc_str}\n</tool_calls><|im_end|>")
            else:
                # 普通 assistant 消息（最终结果）
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        elif role == "tool":
            # Tool 结果
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>tool\n<tool_call_id>{tool_call_id}</tool_call_id>\n{content}<|im_end|>")
    
    return "\n".join(parts)


def preprocess_function(
    examples: Dict,
    tokenizer,
    max_length: int = 4096,
) -> Dict:
    """预处理 Function Calling 格式的 SFT 数据"""
    
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for i in range(len(examples["messages"])):
        messages = examples["messages"][i]
        tools_list = examples.get("tools", [])
        tools = tools_list[i] if i < len(tools_list) else None
        
        # 转换为 chat 格式
        full_text = format_fc_messages(messages, tools)
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 创建 labels：只在 assistant 消息上计算 loss
        # 高效方法：在文本层面找到 assistant 区域，再转换为 token 索引
        labels = [-100] * len(input_ids)  # 默认全部 mask
        
        # 找到所有 assistant 区域的字符位置
        assistant_start_tag = "<|im_start|>assistant\n"
        assistant_end_tag = "<|im_end|>"
        
        search_pos = 0
        while True:
            start_pos = full_text.find(assistant_start_tag, search_pos)
            if start_pos == -1:
                break
            
            # assistant 内容从 tag 之后开始
            content_start = start_pos + len(assistant_start_tag)
            end_pos = full_text.find(assistant_end_tag, content_start)
            if end_pos == -1:
                end_pos = len(full_text)
            
            # 包含 end tag
            content_end = end_pos + len(assistant_end_tag)
            
            # 使用 offset_mapping 精确计算 token 位置
            # 先获取完整文本的 token 对应的字符范围
            encoded_with_offsets = tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            offsets = encoded_with_offsets.get("offset_mapping", [])
            
            # 找到 content_start 和 content_end 对应的 token 索引
            prefix_tokens = 0
            suffix_tokens = 0
            for idx, (start, end) in enumerate(offsets):
                if start < content_start:
                    prefix_tokens = idx + 1
                if start < content_end:
                    suffix_tokens = idx + 1
            
            # 这些位置的 token 需要计算 loss
            for j in range(prefix_tokens, min(suffix_tokens, len(input_ids))):
                labels[j] = input_ids[j]
            
            search_pos = content_end
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT 训练")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径（强烈建议提供）")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B",
                        help="基座模型")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft", help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 其他
    parser.add_argument("--dry_run", action="store_true", help="只打印配置，不训练")
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 断点续训")
    parser.add_argument("--low_memory", action="store_true", help="低显存模式 (8GB 显卡)")
    
    args = parser.parse_args()
    
    # 低显存模式覆盖参数 (6GB 显卡如 3060 Laptop)
    if args.low_memory:
        args.batch_size = 1
        args.gradient_accumulation = 16
        args.max_length = 256  # 6GB 显存必须短
        args.lora_r = 4        # 最小 rank
        args.lora_alpha = 8
        print("⚠️ 低显存模式已启用 (6GB)！")
        print("  - max_length=256, lora_r=4")
        print("  - 如果还是 OOM，试试关闭其他程序")
    
    print("=" * 50)
    print("QLoRA SFT 训练")
    print("=" * 50)
    print(f"模型: {args.model_name}")
    print(f"训练数据: {args.train_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Max length: {args.max_length}")
    print("=" * 50)
    
    if args.dry_run:
        print("[Dry run] 配置检查完成，退出")
        return
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    train_data = load_sft_data(args.train_data)
    print(f"  训练样本: {len(train_data)}")
    
    if args.val_data:
        val_data = load_sft_data(args.val_data)
        print(f"  验证样本: {len(val_data)}")
    else:
        val_data = None
    
    # 转换为 Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    
    # 加载 tokenizer
    print("\n[2/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 预处理数据
    print("\n[3/5] 预处理数据...")
    
    def preprocess_fn(examples):
        return preprocess_function(
            examples, 
            tokenizer, 
            max_length=args.max_length,
        )
    
    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练集",
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="处理验证集",
        )
    
    # 加载模型（4-bit 量化）
    print("\n[4/5] 加载模型...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 低显存模式：启用梯度检查点
    if args.low_memory:
        model.gradient_checkpointing_enable()
        print("  [低显存] 梯度检查点已启用")
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练配置
    print("\n[5/5] 开始训练...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100,
        save_total_limit=2,  # 保留 best + last
        load_best_model_at_end=True if val_dataset else False,  # 加载最好的模型
        metric_for_best_model="eval_loss",  # 用 eval_loss 判断
        greater_is_better=False,  # loss 越小越好
        bf16=not args.low_memory,  # 低显存模式用 fp16
        fp16=args.low_memory,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs",
        remove_unused_columns=False,
        gradient_checkpointing=args.low_memory,  # 低显存模式启用
        dataloader_pin_memory=not args.low_memory,  # 低显存模式禁用
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Callbacks
    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"  [Early Stopping] patience={args.early_stopping_patience}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 训练
    if args.resume:
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")] if os.path.exists(args.output_dir) else []
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_path = os.path.join(args.output_dir, latest)
            print(f"[续训] 从 {resume_path} 恢复")
            trainer.train(resume_from_checkpoint=resume_path)
        else:
            print("[续训] 未找到 checkpoint，从头开始")
            trainer.train()
    else:
        trainer.train()
    
    # 保存
    print("\n[保存模型...]")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n[完成] 模型保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
