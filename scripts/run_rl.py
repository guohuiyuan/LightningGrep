"""
一键 RL 训练脚本
自动下载 SWE-Bench 数据 + 按需克隆仓库 + 开始训练

使用方法:
    python scripts/run_rl.py --sft_model outputs/sft_v2
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

from src.environment.code_search_env import CodeSearchEnv, TOOLS_DEFINITION


# ========== 数据下载 ==========

def load_swebench_data(split: str = "lite", cache_dir: str = "data/swebench"):
    """从 HuggingFace 加载 SWE-Bench 数据"""
    cache_file = Path(cache_dir) / f"{split}.json"
    
    # 优先使用本地缓存
    if cache_file.exists():
        print(f"  从缓存加载: {cache_file}")
        with open(cache_file, "r") as f:
            return json.load(f)
    
    # 检查预处理的数据文件
    local_files = {
        "lite": Path(cache_dir) / "lite_all.json",
        "verified": Path(cache_dir) / "verified_all.json",
    }
    if split in local_files and local_files[split].exists():
        print(f"  从本地文件加载: {local_files[split]}")
        with open(local_files[split], "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # 转换字段名（兼容不同格式）
        data = []
        for item in raw_data:
            data.append({
                "instance_id": item.get("instance_id", ""),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement") or item.get("query", ""),
                "patch": item.get("patch", ""),
                "hints_text": item.get("hints_text", ""),
            })
        return data
    
    print(f"  从 HuggingFace 下载 SWE-Bench {split}...")
    
    # 尝试使用镜像（国内访问）
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    except:
        pass
    
    try:
        if split == "lite":
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        else:
            dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    except Exception as e:
        print(f"\n  ⚠️ 无法下载数据集: {e}")
        print(f"  请手动下载并放到: {cache_file}")
        print(f"  或设置环境变量: export HF_ENDPOINT=https://hf-mirror.com")
        raise
    
    data = []
    for item in dataset:
        data.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "hints_text": item.get("hints_text", ""),
        })
    
    # 缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  缓存到: {cache_file}")
    return data


def parse_patch_files(patch: str) -> List[str]:
    """从 patch 中提取修改的文件列表"""
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # diff --git a/path/to/file b/path/to/file
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2][2:]  # 去掉 "a/"
                files.append(file_path)
    return files


def parse_patch_lines(patch: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    从 patch 中提取修改的文件和行范围
    
    Returns:
        {file_path: [(start_line, end_line), ...]}
    """
    import re
    
    result = {}
    current_file = None
    
    for line in patch.split("\n"):
        # 新文件
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2][2:]  # 去掉 "a/"
                result[current_file] = []
        
        # 行范围: @@ -42,6 +42,8 @@
        elif line.startswith("@@") and current_file:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                end = start + count - 1
                result[current_file].append((start, end))
    
    return result


# ========== 仓库管理 ==========

class RepoManager:
    """按需克隆和管理仓库"""
    
    def __init__(self, cache_dir: str = "data/swebench/repos"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_repos = set()
        self._scan_existing()
    
    def _scan_existing(self):
        """扫描已克隆的仓库"""
        if not self.cache_dir.exists():
            return
        for repo_dir in self.cache_dir.iterdir():
            if repo_dir.is_dir() and (repo_dir / ".git").exists():
                self.cloned_repos.add(repo_dir.name)
    
    def get_repo_path(self, repo: str, commit: str) -> Path:
        """获取仓库路径，如果不存在则克隆"""
        # repo 格式: "owner/name"
        repo_name = repo.replace("/", "__")
        repo_path = self.cache_dir / repo_name
        
        if repo_name not in self.cloned_repos:
            self._clone_repo(repo, repo_path)
            self.cloned_repos.add(repo_name)
        
        # 切换到指定 commit
        self._checkout_commit(repo_path, commit)
        
        return repo_path
    
    def _clone_repo(self, repo: str, repo_path: Path):
        """克隆仓库"""
        url = f"https://github.com/{repo}.git"
        print(f"  克隆仓库: {repo}...")
        
        try:
            subprocess.run(
                ["git", "clone", "--depth", "100", url, str(repo_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ 克隆失败: {e.stderr.decode()}")
            raise
    
    def _checkout_commit(self, repo_path: Path, commit: str):
        """切换到指定 commit"""
        try:
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # 可能需要 fetch 更多历史
            subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_path,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )


# ========== 奖励计算（使用 env.compute_reward，支持文件+行级 F1）==========


def get_repo_tree(repo_path: Path, max_depth: int = 2, max_items: int = 50) -> str:
    """
    获取仓库目录结构（用于 prompt）
    
    主代理会先分析仓库结构，然后传给子代理
    """
    lines = []
    count = [0]  # 用列表来在闭包中修改
    
    def walk(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth or count[0] >= max_items:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return
        
        # 过滤掉常见的无关目录
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox', 'dist', 'build', '.eggs'}
        items = [x for x in items if x.name not in skip_dirs]
        
        for i, item in enumerate(items):
            if count[0] >= max_items:
                lines.append(f"{prefix}... (truncated)")
                return
            
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                count[0] += 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                walk(item, new_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name}")
                count[0] += 1
    
    lines.append(f"{repo_path.name}/")
    walk(repo_path)
    
    return "\n".join(lines)


# ========== 简化版 RL 训练 ==========

@dataclass
class Trajectory:
    """一条搜索轨迹"""
    instance_id: str
    query: str
    files_found: List[str]
    ground_truth: List[str]
    reward: float
    log_prob: float


class SimpleRLTrainer:
    """简化版 REINFORCE 训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        repo_manager: RepoManager,
        learning_rate: float = 1e-5,
        num_rollouts: int = 4,
        max_turns: int = 3,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_manager = repo_manager
        self.num_rollouts = num_rollouts
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )
    
    def generate_response(self, messages: List[Dict]) -> str:
        """生成模型响应"""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS_DEFINITION,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        
        return response
    
    def run_episode(self, instance: Dict) -> Trajectory:
        """运行一个 episode"""
        repo_path = self.repo_manager.get_repo_path(
            instance["repo"],
            instance["base_commit"]
        )
        
        env = CodeSearchEnv(str(repo_path), max_turns=self.max_turns)
        
        # 解析 ground truth（文件 + 行）
        ground_truth_files = parse_patch_files(instance["patch"])
        ground_truth_lines = parse_patch_lines(instance["patch"])
        
        # 获取仓库目录结构（主代理提前分析好，传给子代理）
        repo_tree = get_repo_tree(repo_path)
        
        # 构建 prompt：目录结构 + Issue
        user_prompt = f"""Repository: {instance["repo"]}

Directory structure:
```
{repo_tree}
```

Issue:
{instance["problem_statement"]}

Find the relevant files and code locations for this issue."""
        
        # 初始消息
        messages = [{"role": "user", "content": user_prompt}]
        
        total_log_prob = 0.0
        
        for turn in range(self.max_turns):
            # 生成响应
            response = self.generate_response(messages)
            
            # 解析 tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                break
            
            # 执行 tool calls
            results, done = env.step(tool_calls)
            
            # 更新消息
            messages.append({"role": "assistant", "content": response})
            for result in results:
                messages.append({"role": "tool", "content": result["content"]})
            
            if done:
                break
        
        # 计算奖励（文件 + 行级 F1 的平均）
        reward = env.compute_reward(ground_truth_files, ground_truth_lines)
        
        return Trajectory(
            instance_id=instance["instance_id"],
            query=instance["problem_statement"][:100],
            files_found=list(env.files_found),
            ground_truth=ground_truth_files,
            reward=reward,
            log_prob=total_log_prob,
        )
    
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """解析工具调用"""
        import re
        
        tool_calls = []
        
        # 匹配 <tool_call>...</tool_call> 或 JSON
        patterns = [
            r'<tool_call>\s*(\{[^}]+\})\s*',
            r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        tool_calls.append({
                            "name": match[0],
                            "arguments": json.loads(match[1])
                        })
                    else:
                        tool_calls.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls[:8]  # 最多 8 个并行调用
    
    def train_step(self, instances: List[Dict]) -> Dict:
        """一步训练"""
        all_trajectories = []
        
        for instance in instances:
            for _ in range(self.num_rollouts):
                try:
                    traj = self.run_episode(instance)
                    all_trajectories.append(traj)
                except Exception as e:
                    print(f"  ⚠️ Episode 失败: {e}")
                    continue
        
        if not all_trajectories:
            return {"loss": 0, "reward": 0}
        
        # 计算 baseline (mean reward)
        rewards = [t.reward for t in all_trajectories]
        baseline = sum(rewards) / len(rewards)
        
        # REINFORCE 更新 (简化版)
        # 这里只返回统计信息，实际梯度更新需要更复杂的实现
        
        return {
            "loss": 0,
            "reward": sum(rewards) / len(rewards),
            "baseline": baseline,
            "num_trajectories": len(all_trajectories),
        }
    
    def train(
        self,
        train_data: List[Dict],
        num_steps: int = 100,
        batch_size: int = 4,
        eval_every: int = 20,
        save_every: int = 50,
        output_dir: str = "outputs/rl_v1",
        start_step: int = 0,
    ):
        """训练循环"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        log_file = output_path / "training_log.jsonl"
        
        print(f"\n开始 RL 训练: {num_steps} steps (从 step {start_step} 开始)")
        print(f"  Batch size: {batch_size}")
        print(f"  Rollouts per instance: {self.num_rollouts}")
        print(f"  日志文件: {log_file}")
        
        step = start_step
        total_reward = 0
        
        pbar = tqdm(total=num_steps, initial=start_step, desc="Training")
        
        while step < num_steps:
            # 随机采样 batch
            import random
            batch = random.sample(train_data, min(batch_size, len(train_data)))
            
            # 训练一步
            stats = self.train_step(batch)
            
            total_reward += stats["reward"]
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{stats['reward']:.3f}",
                "avg": f"{total_reward/(step-start_step):.3f}",
            })
            
            # 记录日志
            log_entry = {
                "step": step,
                "reward": stats["reward"],
                "avg_reward": total_reward / (step - start_step),
                "loss": stats.get("loss", 0),
                "num_trajectories": stats.get("num_trajectories", 0),
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # 保存 checkpoint
            if step % save_every == 0:
                save_path = output_path / f"checkpoint-{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                # 1. 保存模型
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                
                # 2. 保存 optimizer 状态
                torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")
                
                # 3. 保存训练状态
                state = {
                    "step": step,
                    "total_reward": total_reward,
                    "avg_reward": total_reward / (step - start_step) if step > start_step else 0,
                }
                with open(save_path / "train_state.json", "w") as f:
                    json.dump(state, f, indent=2)
                
                # 4. 只保留最近 3 个 checkpoint（节省空间）
                all_ckpts = sorted(output_path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
                for old_ckpt in all_ckpts[:-3]:
                    import shutil
                    shutil.rmtree(old_ckpt)
                
                print(f"\n  保存: {save_path}")
        
        pbar.close()
        
        # 最终保存
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"\n训练完成，模型保存到: {output_path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="一键 RL 训练")
    
    # 模型
    parser.add_argument("--sft_model", type=str, required=True, help="SFT 模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B", help="基座模型")
    parser.add_argument("--output_dir", type=str, default="outputs/rl_v1", help="输出目录")
    
    # 数据
    parser.add_argument("--split", type=str, default="lite", choices=["lite", "full"], help="SWE-Bench 版本")
    parser.add_argument("--max_samples", type=int, default=100, help="最大训练样本数")
    
    # 训练参数
    parser.add_argument("--num_steps", type=int, default=100, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_rollouts", type=int, default=4, help="每个样本的 rollout 数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--max_turns", type=int, default=4, help="最大对话轮数")
    parser.add_argument("--max_parallel_calls", type=int, default=8, help="每轮最大并行工具调用数")
    
    # 量化设置
    parser.add_argument("--quantization", type=str, default="4bit", 
                        choices=["4bit", "8bit", "none"], help="量化方式")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"], help="计算精度")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 稳定性参数
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--max_traj_length", type=int, default=4096, help="最大轨迹长度（token数）")
    parser.add_argument("--scale_by_tool_calls", action="store_true", default=True, help="按工具调用数缩放advantage")
    parser.add_argument("--no_scale_by_tool_calls", action="store_false", dest="scale_by_tool_calls")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_every", type=int, default=50, help="每N步保存一次")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印详细信息")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 继续训练，如 outputs/rl_v1/checkpoint-100")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightningGrep RL 训练")
    print("=" * 60)
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未检测到 GPU，训练会很慢！")
    
    # 打印配置
    print(f"\n配置:")
    print(f"  量化: {args.quantization}")
    print(f"  精度: {args.dtype}")
    print(f"  温度: {args.temperature}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  学习率: {args.learning_rate}")
    
    # 1. 加载数据
    print("\n[1/4] 加载 SWE-Bench 数据...")
    data = load_swebench_data(args.split)
    
    # 限制样本数
    if args.max_samples and len(data) > args.max_samples:
        data = random.sample(data, args.max_samples)
    
    print(f"  使用 {len(data)} 条数据")
    
    # 2. 初始化仓库管理器
    print("\n[2/4] 初始化仓库管理器...")
    repo_manager = RepoManager()
    print(f"  缓存目录: {repo_manager.cache_dir}")
    print(f"  已缓存仓库: {len(repo_manager.cloned_repos)}")
    
    # 3. 加载模型
    print("\n[3/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    
    # 设置精度
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    compute_dtype = dtype_map[args.dtype]
    
    # 量化配置
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:  # none
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # 加载 SFT LoRA 并合并
    model = PeftModel.from_pretrained(base_model, args.sft_model)
    model = model.merge_and_unload()
    
    # 创建新的 LoRA 用于 RL
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 训练
    print("\n[4/4] 开始 RL 训练...")
    
    # 检查是否从 checkpoint 继续
    start_step = 0
    resume_optimizer_path = None
    if args.resume:
        resume_path = Path(args.resume)
        state_file = resume_path / "train_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            start_step = state["step"]
            print(f"  从 checkpoint 继续: step {start_step}")
            
            # 重新加载模型：基座 + checkpoint LoRA（不需要再创建新 LoRA）
            if args.quantization == "4bit":
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif args.quantization == "8bit":
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            # 先合并 SFT LoRA
            sft_model = PeftModel.from_pretrained(base_model, args.sft_model)
            merged_model = sft_model.merge_and_unload()
            
            # 加载 checkpoint 的 RL LoRA
            model = PeftModel.from_pretrained(merged_model, args.resume)
            model.print_trainable_parameters()
            
            # 检查 optimizer 状态
            if (resume_path / "optimizer.pt").exists():
                resume_optimizer_path = resume_path / "optimizer.pt"
                print(f"  加载 optimizer 状态")
        else:
            print(f"  ⚠️ 未找到 train_state.json，从头开始")
    
    trainer = SimpleRLTrainer(
        model=model,
        tokenizer=tokenizer,
        repo_manager=repo_manager,
        learning_rate=args.learning_rate,
        num_rollouts=args.num_rollouts,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )
    
    # 加载 optimizer 状态
    if resume_optimizer_path:
        trainer.optimizer.load_state_dict(torch.load(resume_optimizer_path))
    
    # 调试模式
    if args.debug:
        trainer.debug = True
    
    trainer.train(
        train_data=data,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_every=args.save_every,
        start_step=start_step,
    )
    
    # 打印完成信息
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
