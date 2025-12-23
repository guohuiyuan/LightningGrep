"""
Code Search 环境
用于 RL 训练的交互式代码搜索环境
"""
import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果"""
    files_found: List[str]
    lines_found: Dict[str, List[Tuple[int, int]]]  # file -> [(start, end), ...]
    tool_calls: int
    turns: int


class CodeSearchEnv:
    """代码搜索环境"""
    
    def __init__(
        self,
        repo_path: str,
        max_turns: int = 4,
        max_parallel_calls: int = 8,
        max_grep_results: int = 50,
        max_read_lines: int = 100,
    ):
        self.repo_path = Path(repo_path)
        self.max_turns = max_turns
        self.max_parallel_calls = max_parallel_calls
        self.max_grep_results = max_grep_results
        self.max_read_lines = max_read_lines
        
        self.current_turn = 0
        self.total_tool_calls = 0
        self.files_found = set()
        self.lines_found = {}
        self.history = []
    
    def reset(self):
        """重置环境"""
        self.current_turn = 0
        self.total_tool_calls = 0
        self.files_found = set()
        self.lines_found = {}
        self.history = []
    
    def step(self, tool_calls: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        执行一轮 tool calls
        
        Args:
            tool_calls: [{"name": "grep", "arguments": {...}}, ...]
        
        Returns:
            results: 每个 tool call 的结果
            done: 是否结束
        """
        if self.current_turn >= self.max_turns:
            return [], True
        
        # 限制并行数
        tool_calls = tool_calls[:self.max_parallel_calls]
        
        results = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            
            result = self._execute_tool(name, args)
            results.append({
                "tool_call_id": call.get("id", ""),
                "content": result
            })
            self.total_tool_calls += 1
        
        self.current_turn += 1
        self.history.append({"tool_calls": tool_calls, "results": results})
        
        done = self.current_turn >= self.max_turns
        return results, done
    
    def _execute_tool(self, name: str, args: Dict) -> str:
        """执行单个工具"""
        try:
            if name == "grep":
                return self._grep(args.get("query", ""), args.get("path", "."))
            elif name == "read":
                return self._read(
                    args.get("file", ""),
                    args.get("start", 1),
                    args.get("end", 50)
                )
            elif name == "glob":
                return self._glob(args.get("pattern", "*"))
            elif name == "find":
                return self._find(args.get("name", ""))
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _grep(self, query: str, path: str = ".") -> str:
        """搜索文本"""
        if not query:
            return "Error: empty query"
        
        search_path = self.repo_path / path
        if not search_path.exists():
            search_path = self.repo_path
        
        try:
            # 使用 grep 或 ripgrep
            cmd = ["rg", "-n", "-l", "--max-count", str(self.max_grep_results), query, str(search_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.strip().split("\n")[:self.max_grep_results]
                # 转换为相对路径
                rel_files = []
                for f in files:
                    try:
                        rel = Path(f).relative_to(self.repo_path)
                        rel_files.append(str(rel))
                        self.files_found.add(str(rel))
                    except:
                        rel_files.append(f)
                return "\n".join(rel_files)
            else:
                return "No matches found"
        except subprocess.TimeoutExpired:
            return "Search timeout"
        except FileNotFoundError:
            # fallback to grep
            try:
                cmd = ["grep", "-r", "-n", "-l", query, str(search_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.stdout:
                    return result.stdout.strip()
                return "No matches found"
            except:
                return "grep not available"
    
    def _read(self, file: str, start: int = 1, end: int = 50) -> str:
        """读取文件"""
        file_path = self.repo_path / file
        
        if not file_path.exists():
            return f"File not found: {file}"
        
        if not file_path.is_file():
            return f"Not a file: {file}"
        
        # 限制行数
        end = min(end, start + self.max_read_lines)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            start = max(1, start)
            end = min(end, len(lines))
            
            content = []
            for i in range(start - 1, end):
                content.append(f"{i + 1}: {lines[i].rstrip()}")
            
            # 记录找到的文件和行范围
            self.files_found.add(file)
            if file not in self.lines_found:
                self.lines_found[file] = []
            self.lines_found[file].append((start, end))
            
            return "\n".join(content)
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _glob(self, pattern: str) -> str:
        """列出匹配文件"""
        try:
            matches = list(self.repo_path.glob(pattern))[:self.max_grep_results]
            if not matches:
                return "No matches found"
            
            result = []
            for m in matches:
                try:
                    rel = m.relative_to(self.repo_path)
                    result.append(str(rel))
                except:
                    result.append(str(m))
            
            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"
    
    def _find(self, name: str) -> str:
        """按名称查找文件"""
        try:
            matches = []
            for p in self.repo_path.rglob(f"*{name}*"):
                if p.is_file():
                    try:
                        rel = p.relative_to(self.repo_path)
                        matches.append(str(rel))
                    except:
                        pass
                if len(matches) >= self.max_grep_results:
                    break
            
            if not matches:
                return "No matches found"
            return "\n".join(matches)
        except Exception as e:
            return f"Error: {e}"
    
    def get_result(self) -> SearchResult:
        """获取搜索结果"""
        return SearchResult(
            files_found=list(self.files_found),
            lines_found=self.lines_found,
            tool_calls=self.total_tool_calls,
            turns=self.current_turn
        )
    
    def compute_reward(
        self,
        ground_truth_files: List[str],
        ground_truth_lines: Dict[str, List[Tuple[int, int]]] = None,
        beta: float = 0.5
    ) -> float:
        """
        计算 reward (Weighted F1, β=0.5 偏向 Precision)
        
        博客设计: reward = (file_f1 + line_f1) / 2
        
        Args:
            ground_truth_files: 真实需要找到的文件
            ground_truth_lines: 真实需要找到的行范围 {file: [(start, end), ...]}
            beta: F-beta 的 beta 值，<1 偏向 precision，>1 偏向 recall
        
        Returns:
            reward: F-beta score
        """
        if not ground_truth_files:
            return 0.0
        
        # 1. 文件级 F1
        file_f1 = self._compute_file_f1(ground_truth_files, beta)
        
        # 2. 行级 F1（如果有 ground truth）
        if ground_truth_lines:
            line_f1 = self._compute_line_f1(ground_truth_lines, beta)
            # 博客: 文件和行的平均
            return (file_f1 + line_f1) / 2
        
        return file_f1
    
    def _compute_file_f1(self, ground_truth_files: List[str], beta: float) -> float:
        """计算文件级 F-beta"""
        gt_set = set(ground_truth_files)
        found_set = self.files_found
        
        if not found_set:
            return 0.0
        
        correct = len(gt_set & found_set)
        
        precision = correct / len(found_set) if found_set else 0
        recall = correct / len(gt_set) if gt_set else 0
        
        if precision + recall == 0:
            return 0.0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    def _compute_line_f1(
        self, 
        ground_truth_lines: Dict[str, List[Tuple[int, int]]], 
        beta: float
    ) -> float:
        """
        计算行级 F-beta
        
        使用行覆盖率：预测的行范围与 ground truth 行范围的重叠
        """
        # 将行范围转换为行集合
        gt_lines = set()
        for file, ranges in ground_truth_lines.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    gt_lines.add((file, line))
        
        found_lines = set()
        for file, ranges in self.lines_found.items():
            for start, end in ranges:
                for line in range(start, end + 1):
                    found_lines.add((file, line))
        
        if not found_lines or not gt_lines:
            return 0.0
        
        correct = len(gt_lines & found_lines)
        
        precision = correct / len(found_lines) if found_lines else 0
        recall = correct / len(gt_lines) if gt_lines else 0
        
        if precision + recall == 0:
            return 0.0
        
        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


# 工具定义（供模型使用）
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for text patterns in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "File path"},
                    "start": {"type": "integer", "description": "Start line", "default": 1},
                    "end": {"type": "integer", "description": "End line", "default": 50}
                },
                "required": ["file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find files by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "File name to search"}
                },
                "required": ["name"]
            }
        }
    }
]
