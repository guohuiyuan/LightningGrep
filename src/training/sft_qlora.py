"""
QLoRA SFT 训练脚本
用于训练并行检索模型
"""
import os
import json
import torch
import argparse
from typing import Dict, List
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# 特殊标签，需要 mask 的部分
MASK_TAGS = ["information"]


def load_sft_data(data_path: str) -> List[Dict]:
    """加载 SFT 数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_mask_for_tags(
    input_ids: List[int], 
    tokenizer, 
    full_text: str,
    mask_tags: List[str]
) -> List[int]:
    """
    创建 label mask，对于 mask_tags 内的内容不计算 loss
    返回 labels，其中被 mask 的位置为 -100
    """
    labels = input_ids.copy()
    
    for tag in mask_tags:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        # 找到所有 tag 的位置
        text = full_text
        search_start = 0
        
        while True:
            start_pos = text.find(start_tag, search_start)
            if start_pos == -1:
                break
            
            end_pos = text.find(end_tag, start_pos)
            if end_pos == -1:
                break
            
            # 计算这段文本对应的 token 位置
            # 简化处理：找到 tag 内容的 token 范围
            prefix = text[:start_pos + len(start_tag)]
            content = text[start_pos + len(start_tag):end_pos]
            
            # 编码前缀和内容，计算 token 位置
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            
            # Mask 这些位置（包括 tag 本身）
            tag_start_tokens = tokenizer.encode(start_tag, add_special_tokens=False)
            tag_end_tokens = tokenizer.encode(end_tag, add_special_tokens=False)
            
            # 找到在 input_ids 中的位置并 mask
            # 简化：mask start_tag 到 end_tag 的所有 token
            start_idx = len(tokenizer.encode(text[:start_pos], add_special_tokens=False))
            end_idx = len(tokenizer.encode(text[:end_pos + len(end_tag)], add_special_tokens=False))
            
            for i in range(start_idx, min(end_idx, len(labels))):
                labels[i] = -100
            
            search_start = end_pos + len(end_tag)
    
    return labels


def preprocess_function(
    examples: Dict,
    tokenizer,
    max_length: int = 2048,
    mask_info: bool = True,
) -> Dict:
    """预处理函数：将 input/output 转换为模型输入格式"""
    
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        # 构建完整的对话格式
        # Qwen3 格式
        full_text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
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
        
        # 创建 labels
        # 1. 首先，input 部分不计算 loss
        user_part = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        user_tokens = tokenizer.encode(user_part, add_special_tokens=False)
        
        labels = [-100] * len(user_tokens) + input_ids[len(user_tokens):]
        
        # 2. 如果需要 mask <information>，处理 output 部分
        if mask_info and "mask_tags" in examples:
            mask_tags = examples["mask_tags"][i] if examples["mask_tags"][i] else []
            if mask_tags:
                # 简化处理：找到 <information> 并 mask
                assistant_part = output_text
                for tag in mask_tags:
                    start_tag = f"<{tag}>"
                    end_tag = f"</{tag}>"
                    
                    search_start = 0
                    while True:
                        start_pos = assistant_part.find(start_tag, search_start)
                        if start_pos == -1:
                            break
                        end_pos = assistant_part.find(end_tag, start_pos)
                        if end_pos == -1:
                            break
                        
                        # 计算在 full_text 中的位置
                        full_start = len(user_part) + start_pos
                        full_end = len(user_part) + end_pos + len(end_tag)
                        
                        # 转换为 token 索引
                        token_start = len(tokenizer.encode(full_text[:full_start], add_special_tokens=False))
                        token_end = len(tokenizer.encode(full_text[:full_end], add_special_tokens=False))
                        
                        # Mask
                        for j in range(token_start, min(token_end, len(labels))):
                            if j < len(labels):
                                labels[j] = -100
                        
                        search_start = end_pos + len(end_tag)
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT 训练")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-Instruct",
                        help="基座模型")
    parser.add_argument("--output_dir", type=str, default="./outputs/sft", help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 其他
    parser.add_argument("--mask_info", action="store_true", help="是否 mask <information> 标签")
    parser.add_argument("--dry_run", action="store_true", help="只打印配置，不训练")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("QLoRA SFT 训练")
    print("=" * 50)
    print(f"模型: {args.model_name}")
    print(f"训练数据: {args.train_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Mask <information>: {args.mask_info}")
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
            mask_info=args.mask_info,
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
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 训练
    trainer.train()
    
    # 保存
    print("\n[保存模型...]")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n[完成] 模型保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
