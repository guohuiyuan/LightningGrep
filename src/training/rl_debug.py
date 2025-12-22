"""
RL 训练调试工具
用于分析轨迹、奖励分布、梯度等
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import statistics


@dataclass
class TrainingStats:
    """训练统计"""
    step: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    tool_calls_mean: float
    format_error_rate: float
    trajectories_count: int
    positive_reward_rate: float  # reward > 0 的比例


def compute_stats(trajectories: List, step: int = 0) -> TrainingStats:
    """计算轨迹统计"""
    if not trajectories:
        return TrainingStats(step, 0, 0, 0, 0, 0, 0, 0, 0)
    
    rewards = [t.reward for t in trajectories]
    tool_calls = [t.tool_calls_count for t in trajectories]
    format_errors = [1 if t.format_error else 0 for t in trajectories]
    
    return TrainingStats(
        step=step,
        reward_mean=statistics.mean(rewards),
        reward_std=statistics.stdev(rewards) if len(rewards) > 1 else 0,
        reward_min=min(rewards),
        reward_max=max(rewards),
        tool_calls_mean=statistics.mean(tool_calls),
        format_error_rate=statistics.mean(format_errors),
        trajectories_count=len(trajectories),
        positive_reward_rate=sum(1 for r in rewards if r > 0) / len(rewards),
    )


def print_trajectory(traj, max_content_len: int = 200):
    """打印单条轨迹用于调试"""
    print("=" * 60)
    print(f"Query: {traj.query[:100]}...")
    print(f"Reward: {traj.reward:.4f}")
    print(f"Format Error: {traj.format_error}")
    print(f"Tool Calls: {traj.tool_calls_count}")
    print(f"Turns: {traj.turns}")
    print(f"Ground Truth: {traj.ground_truth_files}")
    print(f"Found: {traj.files_found}")
    print("-" * 40)
    
    for i, msg in enumerate(traj.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")[:max_content_len]
        
        if role == "assistant" and "tool_calls" in msg:
            tc = msg["tool_calls"]
            print(f"[{i}] ASSISTANT: {len(tc)} tool calls")
            for call in tc[:3]:  # 最多显示 3 个
                name = call.get("name", "?")
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = args[:50]
                print(f"      - {name}({args})")
        elif role == "tool":
            content = content[:100] + "..." if len(content) > 100 else content
            print(f"[{i}] TOOL: {content}")
        else:
            print(f"[{i}] {role.upper()}: {content}")
    
    print("=" * 60)


def analyze_rewards(trajectories: List) -> Dict:
    """分析奖励分布"""
    rewards = [t.reward for t in trajectories]
    
    if not rewards:
        return {}
    
    # 按区间统计
    bins = {
        "0.0": 0,
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    
    for r in rewards:
        if r == 0:
            bins["0.0"] += 1
        elif r < 0.2:
            bins["0.0-0.2"] += 1
        elif r < 0.4:
            bins["0.2-0.4"] += 1
        elif r < 0.6:
            bins["0.4-0.6"] += 1
        elif r < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    
    return {
        "distribution": bins,
        "mean": statistics.mean(rewards),
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0,
        "median": statistics.median(rewards),
    }


def save_trajectories(trajectories: List, path: str):
    """保存轨迹用于分析"""
    data = []
    for t in trajectories:
        data.append({
            "instance_id": t.instance_id,
            "query": t.query[:500],
            "reward": t.reward,
            "format_error": t.format_error,
            "tool_calls_count": t.tool_calls_count,
            "turns": t.turns,
            "ground_truth_files": t.ground_truth_files,
            "files_found": t.files_found,
            "messages": t.messages,
        })
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存 {len(data)} 条轨迹到: {path}")


def check_gradient_health(model) -> Dict:
    """检查梯度健康状态"""
    stats = {
        "total_params": 0,
        "trainable_params": 0,
        "grad_norm": 0.0,
        "zero_grad_params": 0,
        "nan_grad_params": 0,
        "inf_grad_params": 0,
    }
    
    total_norm = 0.0
    for name, param in model.named_parameters():
        stats["total_params"] += 1
        
        if param.requires_grad:
            stats["trainable_params"] += 1
            
            if param.grad is not None:
                grad = param.grad.data
                
                if torch.isnan(grad).any():
                    stats["nan_grad_params"] += 1
                elif torch.isinf(grad).any():
                    stats["inf_grad_params"] += 1
                elif (grad == 0).all():
                    stats["zero_grad_params"] += 1
                
                total_norm += grad.norm(2).item() ** 2
    
    stats["grad_norm"] = total_norm ** 0.5
    
    return stats


# 需要 torch
try:
    import torch
except ImportError:
    pass
