"""
REINFORCE RL 训练脚本
基于 SWE-grep 的方法：Per-sequence Importance Sampling + Leave-one-out Baseline

使用方法:
    python src/training/rl_reinforce.py \
        --sft_model outputs/sft_v2 \
        --train_data data/swebench/rl_train.json \
        --val_data data/swebench/rl_val.json \
        --repo_dir data/swebench/repos \
        --output_dir outputs/rl_v1
"""
import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 可选: wandb 日志
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.environment.code_search_env import CodeSearchEnv, TOOLS_DEFINITION


@dataclass
class Trajectory:
    """一条完整的搜索轨迹"""
    instance_id: str
    query: str
    messages: List[Dict]
    tool_calls_count: int
    turns: int
    files_found: List[str]
    ground_truth_files: List[str]
    reward: float
    log_probs: List[torch.Tensor]  # 每个 token 的 log prob
    format_error: bool = False  # 是否有格式错误
    kl_divergence: float = 0.0  # KL 散度 (相对于 reference policy)
    

def compute_log_probs(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """计算生成 token 的 log probabilities"""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算 log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 取出对应 label 的 log prob
    token_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask padding
    mask = (shift_labels != -100).float()
    token_log_probs = token_log_probs * mask
    
    return token_log_probs, mask


def generate_trajectory(
    model,
    tokenizer,
    env: CodeSearchEnv,
    query: str,
    ground_truth_files: List[str],
    max_turns: int = 4,
    temperature: float = 0.7,
    device: str = "cuda",
) -> Trajectory:
    """生成一条搜索轨迹"""
    env.reset()
    
    messages = [
        {"role": "system", "content": f"You are a code search agent. Use the tools to find relevant code. Available tools: {json.dumps(TOOLS_DEFINITION)}"},
        {"role": "user", "content": query}
    ]
    
    all_log_probs = []
    
    for turn in range(max_turns):
        # 构建输入
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # 解码生成的文本
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 计算 log probs
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab]
            log_probs = F.log_softmax(scores, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=generated_ids.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).squeeze(0)
            all_log_probs.append(token_log_probs)
        
        # 解析 tool calls
        tool_calls, format_error = parse_tool_calls(generated_text)
        
        if format_error:
            # 格式错误，给 0 reward 并结束 (SWE-grep 的做法)
            messages.append({"role": "assistant", "content": generated_text})
            return Trajectory(
                instance_id="",
                query=query,
                messages=messages,
                tool_calls_count=env.total_tool_calls,
                turns=env.current_turn,
                files_found=list(env.files_found),
                ground_truth_files=ground_truth_files,
                reward=0.0,  # 格式错误 = 0 reward
                log_probs=all_log_probs,
                format_error=True,
            )
        
        if not tool_calls:
            # 没有 tool calls，认为搜索结束
            messages.append({"role": "assistant", "content": generated_text})
            break
        
        # 执行 tool calls
        results, done = env.step(tool_calls)
        
        # 添加到 messages
        messages.append({
            "role": "assistant",
            "content": generated_text,
            "tool_calls": tool_calls
        })
        
        for tc, result in zip(tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result["content"]
            })
        
        if done:
            break
    
    # 计算 reward
    reward = env.compute_reward(ground_truth_files, beta=0.5)
    search_result = env.get_result()
    
    return Trajectory(
        instance_id="",
        query=query,
        messages=messages,
        tool_calls_count=search_result.tool_calls,
        turns=search_result.turns,
        files_found=search_result.files_found,
        ground_truth_files=ground_truth_files,
        reward=reward,
        log_probs=all_log_probs,
        format_error=False,
    )


def parse_tool_calls(text: str) -> Tuple[List[Dict], bool]:
    """
    从生成文本中解析 tool calls
    Returns: (tool_calls, has_format_error)
    """
    tool_calls = []
    has_format_error = False
    
    # 尝试解析 JSON 格式的 tool calls
    # 格式: <tool_calls>[{...}]</tool_calls>
    import re
    pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                calls = json.loads(match)
                if isinstance(calls, list):
                    tool_calls.extend(calls)
                elif isinstance(calls, dict):
                    tool_calls.append(calls)
            except json.JSONDecodeError:
                has_format_error = True  # JSON 解析失败
    
    # 也尝试解析函数调用格式
    # 格式: {"name": "grep", "arguments": {...}}
    if not tool_calls and not has_format_error:
        try:
            # 尝试直接解析整个文本为 JSON
            data = json.loads(text)
            if isinstance(data, list):
                tool_calls = data
            elif isinstance(data, dict) and "name" in data:
                tool_calls = [data]
        except:
            pass
    
    # 验证 tool calls 格式
    valid_calls = []
    for call in tool_calls:
        if isinstance(call, dict) and "name" in call:
            valid_calls.append(call)
        else:
            has_format_error = True
    
    return valid_calls, has_format_error


def compute_advantages(rewards: List[float]) -> List[float]:
    """计算 Leave-one-out baseline advantages"""
    n = len(rewards)
    if n <= 1:
        return rewards
    
    advantages = []
    for i in range(n):
        # Leave-one-out baseline: 其他样本的平均 reward
        other_rewards = [r for j, r in enumerate(rewards) if j != i]
        baseline = sum(other_rewards) / len(other_rewards)
        advantages.append(rewards[i] - baseline)
    
    return advantages


def reinforce_loss(
    trajectories: List[Trajectory],
    model,
    tokenizer,
    device: str = "cuda",
    scale_by_tool_calls: bool = True,
) -> torch.Tensor:
    """
    计算 REINFORCE loss
    L = -1/G * sum_j [ IS_ratio_j * A_j * sum_t log π(a_t|s_t) ]
    """
    if not trajectories:
        return torch.tensor(0.0, device=device)
    
    # 计算 advantages (leave-one-out baseline)
    rewards = [t.reward for t in trajectories]
    advantages = compute_advantages(rewards)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for traj, advantage in zip(trajectories, advantages):
        if not traj.log_probs:
            continue
        
        # 合并所有轮次的 log probs
        all_log_probs = torch.cat(traj.log_probs)
        trajectory_log_prob = all_log_probs.sum()
        
        # 按 tool calls 数量缩放 (SWE-grep 的技巧)
        if scale_by_tool_calls and traj.tool_calls_count > 0:
            avg_calls_per_turn = traj.tool_calls_count / max(traj.turns, 1)
            scale = 1.0 / max(avg_calls_per_turn, 1.0)
        else:
            scale = 1.0
        
        # REINFORCE: -advantage * log_prob
        loss = -advantage * scale * trajectory_log_prob
        total_loss = total_loss + loss
    
    return total_loss / len(trajectories)


class RLTrainer:
    """REINFORCE 训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_data: List[Dict],
        val_data: List[Dict],
        repo_dir: str,
        output_dir: str,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        num_rollouts: int = 4,  # 每个样本采样的轨迹数
        max_turns: int = 4,
        temperature: float = 0.7,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.repo_dir = Path(repo_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.max_turns = max_turns
        self.temperature = temperature
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.step = 0
        self.best_val_reward = 0.0
        self.use_wandb = False
        self.reward_history = []  # 用于 reward 归一化
        self.debug_trajectories = []  # 保存最近的轨迹用于调试
    
    def init_wandb(self, project: str = "lightninggrep-rl", run_name: str = None):
        """初始化 wandb 日志"""
        if HAS_WANDB:
            wandb.init(
                project=project,
                name=run_name or f"rl-{self.step}",
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "num_rollouts": self.num_rollouts,
                    "max_turns": self.max_turns,
                    "temperature": self.temperature,
                }
            )
            self.use_wandb = True
            print("✓ WandB 日志已启用")
        else:
            print("⚠️ wandb 未安装，跳过日志")
    
    def normalize_rewards(self, rewards: List[float]) -> List[float]:
        """归一化 rewards (running mean/std)"""
        self.reward_history.extend(rewards)
        # 保留最近 1000 个 rewards
        self.reward_history = self.reward_history[-1000:]
        
        if len(self.reward_history) < 10:
            return rewards
        
        mean = sum(self.reward_history) / len(self.reward_history)
        std = (sum((r - mean) ** 2 for r in self.reward_history) / len(self.reward_history)) ** 0.5
        
        if std < 1e-8:
            return rewards
        
        return [(r - mean) / (std + 1e-8) for r in rewards]
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """单步训练"""
        self.model.train()
        
        all_trajectories = []
        batch_rewards = []
        batch_tool_calls = []
        format_errors = 0
        
        for item in batch:
            repo_path = self.repo_dir / item["repo"].replace("/", "_")
            
            if not repo_path.exists():
                continue
            
            env = CodeSearchEnv(
                repo_path=str(repo_path),
                max_turns=self.max_turns,
            )
            
            # 采样多条轨迹
            for _ in range(self.num_rollouts):
                traj = generate_trajectory(
                    self.model,
                    self.tokenizer,
                    env,
                    item["query"],
                    item["ground_truth_files"],
                    max_turns=self.max_turns,
                    temperature=self.temperature,
                    device=self.device,
                )
                traj.instance_id = item["instance_id"]
                all_trajectories.append(traj)
                batch_rewards.append(traj.reward)
                batch_tool_calls.append(traj.tool_calls_count)
                if traj.format_error:
                    format_errors += 1
        
        if not all_trajectories:
            return {"loss": 0.0, "reward": 0.0, "tool_calls": 0.0, "format_error_rate": 0.0}
        
        # 保存用于调试
        self.debug_trajectories = all_trajectories[-10:]
        
        # 计算 loss
        loss = reinforce_loss(
            all_trajectories,
            self.model,
            self.tokenizer,
            self.device,
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        metrics = {
            "loss": loss.item(),
            "reward": sum(batch_rewards) / len(batch_rewards),
            "reward_max": max(batch_rewards),
            "reward_min": min(batch_rewards),
            "tool_calls": sum(batch_tool_calls) / len(batch_tool_calls),
            "format_error_rate": format_errors / len(all_trajectories),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
        
        # WandB 日志
        if self.use_wandb:
            wandb.log(metrics, step=self.step)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, data: List[Dict], max_samples: int = 50) -> Dict:
        """评估"""
        self.model.eval()
        
        rewards = []
        tool_calls = []
        
        samples = random.sample(data, min(len(data), max_samples))
        
        for item in tqdm(samples, desc="Evaluating"):
            repo_path = self.repo_dir / item["repo"].replace("/", "_")
            
            if not repo_path.exists():
                continue
            
            env = CodeSearchEnv(
                repo_path=str(repo_path),
                max_turns=self.max_turns,
            )
            
            traj = generate_trajectory(
                self.model,
                self.tokenizer,
                env,
                item["query"],
                item["ground_truth_files"],
                max_turns=self.max_turns,
                temperature=0.1,  # 评估时用低温度
                device=self.device,
            )
            
            rewards.append(traj.reward)
            tool_calls.append(traj.tool_calls_count)
        
        return {
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
        }
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """保存检查点"""
        save_path = self.output_dir / name
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存训练状态
        state = {
            "step": self.step,
            "best_val_reward": self.best_val_reward,
        }
        with open(save_path / "trainer_state.json", "w") as f:
            json.dump(state, f)
        
        print(f"✓ 保存检查点: {save_path}")
    
    def train(self, num_steps: int = 1000, eval_every: int = 100, save_every: int = 200):
        """训练"""
        print("=" * 50)
        print("开始 RL 训练")
        print("=" * 50)
        print(f"训练数据: {len(self.train_data)} 条")
        print(f"验证数据: {len(self.val_data)} 条")
        print(f"Batch size: {self.batch_size}")
        print(f"Rollouts per sample: {self.num_rollouts}")
        print(f"Max turns: {self.max_turns}")
        print("=" * 50)
        
        pbar = tqdm(total=num_steps, desc="Training")
        
        while self.step < num_steps:
            # 随机采样 batch
            batch = random.sample(self.train_data, min(len(self.train_data), self.batch_size))
            
            # 训练
            metrics = self.train_step(batch)
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "reward": f"{metrics['reward']:.4f}",
                "calls": f"{metrics['tool_calls']:.1f}",
            })
            pbar.update(1)
            
            # 评估
            if self.step % eval_every == 0:
                val_metrics = self.evaluate(self.val_data)
                print(f"\n[Step {self.step}] Val reward: {val_metrics['reward']:.4f}, Tool calls: {val_metrics['tool_calls']:.1f}")
                
                if val_metrics['reward'] > self.best_val_reward:
                    self.best_val_reward = val_metrics['reward']
                    self.save_checkpoint("best")
            
            # 保存
            if self.step % save_every == 0:
                self.save_checkpoint(f"checkpoint-{self.step}")
        
        pbar.close()
        self.save_checkpoint("final")
        print(f"\n✓ 训练完成! Best val reward: {self.best_val_reward:.4f}")


def main():
    parser = argparse.ArgumentParser(description="REINFORCE RL 训练")
    
    # 模型参数
    parser.add_argument("--sft_model", type=str, required=True, help="SFT 模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="基座模型")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据")
    parser.add_argument("--val_data", type=str, required=True, help="验证数据")
    parser.add_argument("--repo_dir", type=str, required=True, help="仓库目录")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/rl", help="输出目录")
    parser.add_argument("--num_steps", type=int, default=1000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_rollouts", type=int, default=4, help="每样本采样轨迹数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--max_turns", type=int, default=4, help="最大轮数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    
    # 调试参数
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 日志")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印轨迹")
    parser.add_argument("--save_trajectories", type=str, default=None, help="保存轨迹到文件")
    
    args = parser.parse_args()
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ 未检测到 GPU，训练会很慢！")
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    with open(args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    print(f"  训练: {len(train_data)} 条")
    print(f"  验证: {len(val_data)} 条")
    
    # 加载模型
    print("\n[2/3] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    
    # 4-bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载 SFT LoRA
    model = PeftModel.from_pretrained(base_model, args.sft_model)
    model = model.merge_and_unload()  # 合并 LoRA
    
    # 创建新的 LoRA 用于 RL
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练
    print("\n[3/3] 开始训练...")
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        repo_dir=args.repo_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_rollouts=args.num_rollouts,
        max_turns=args.max_turns,
        temperature=args.temperature,
        device=device,
    )
    
    # 启用 wandb
    if args.wandb:
        trainer.init_wandb()
    
    # 调试模式
    if args.debug:
        trainer.debug_mode = True
        print("⚠️ 调试模式已启用，将打印轨迹详情")
    
    trainer.train(
        num_steps=args.num_steps,
        eval_every=100,
        save_every=200,
    )
    
    # 保存轨迹
    if args.save_trajectories and trainer.debug_trajectories:
        from src.training.rl_debug import save_trajectories
        save_trajectories(trainer.debug_trajectories, args.save_trajectories)


if __name__ == "__main__":
    main()
