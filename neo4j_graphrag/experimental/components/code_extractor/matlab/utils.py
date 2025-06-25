# Copyright (c) "Neo4j"
# Neo4j Sweden AB [https://neo4j.com]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

logger = logging.getLogger(__name__)

# Common MATLAB keyword set for filtering uninteresting identifiers
_MATLAB_KEYWORDS = {
    'if','for','while','function','end','else','elseif','return','break','continue',
    'switch','case','otherwise','try','catch','classdef','properties','methods',
    'true','false','pi','inf','nan','eps','realmax','realmin','i','j'
}

# Cache for builtin function names
_BUILTIN_FUNCS: Optional[Set[str]] = None

def get_matlab_builtin_functions() -> Set[str]:
    """Return a set of MATLAB builtin function names (lower-case)."""
    global _BUILTIN_FUNCS
    if _BUILTIN_FUNCS is None:
        builtin_path = Path(__file__).with_name('matlab_builtin_functions.json')
        try:
            with builtin_path.open('r', encoding='utf-8') as f:
                _BUILTIN_FUNCS = {name.lower() for name in json.load(f)}
        except Exception as e:
            logger.warning("Failed to load builtin function list: %s", e)
            _BUILTIN_FUNCS = set()
    return _BUILTIN_FUNCS


def _remove_strings(line: str) -> str:
    """Remove single- or double-quoted string literals from a line, preserving length."""
    result = []
    in_single = False
    in_double = False
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
            result.append(' ')
        elif ch == '"' and not in_single:
            in_double = not in_double
            result.append(' ')
        elif in_single or in_double:
            result.append(' ')  # mask content
        else:
            result.append(ch)
    return ''.join(result)



def ensure_neo4j_compatible(value: Any) -> Any:
    """确保值兼容Neo4j"""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return [ensure_neo4j_compatible(item) for item in value]
    elif isinstance(value, dict):
        return {k: ensure_neo4j_compatible(v) for k, v in value.items()}
    else:
        return str(value)


def get_code_snippet(node: Dict[str, Any], max_length: int = None) -> str:
    """获取代码片段"""
    code = node.get('properties', {}).get('code_snippet', '')
    if max_length and len(code) > max_length:
        return code[:max_length] + "..."
    return code


def extract_variables_from_code(code: str, min_length: int = 2) -> List[str]:
    """Extract meaningful variable names from MATLAB code string.

    Rules:
    1. Skip comment-only lines (starting with '%').
    2. Remove quoted strings before regex scans.
    3. Ignore MATLAB keywords and identifiers shorter than *min_length*.
    """
    variables: Set[str] = set()
    # Assignment beginning of line
    assign_regex = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=')
    # Generic identifier pattern
    ident_regex = re.compile(r'\b([a-zA-Z_]\w*)\b')

    for raw_line in code.split('\n'):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('%'):
            continue

        clean_line = _remove_strings(stripped)

        # Direct assignment at start of line
        m = assign_regex.match(clean_line)
        if m:
            name = m.group(1)
            if len(name) >= min_length and name not in _MATLAB_KEYWORDS:
                variables.add(name)
            # still scan rest for other identifiers after '=' (e.g., a=b):
            remainder = clean_line[m.end():]
            clean_line = remainder  # fall through to generic scan

        for match in ident_regex.finditer(clean_line):
            name = match.group(1)
            if len(name) >= min_length and name not in _MATLAB_KEYWORDS:
                variables.add(name)

    return sorted(variables)


def extract_function_calls_from_code(code: str) -> List[str]:
    """Extract function calls, ignoring comments, strings, and MATLAB keywords."""
    result: Set[str] = set()
    call_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')

    for raw_line in code.split('\n'):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        clean_line = _remove_strings(stripped)
        for match in call_regex.finditer(clean_line):
            name = match.group(1)
            if name not in _MATLAB_KEYWORDS:
                result.add(name)

    return sorted(result)


def extract_script_calls_from_code(code: str) -> List[str]:
    """Extract script invocations (direct or via run()), skipping comments/strings."""
    result: Set[str] = set()
    direct_regex = re.compile(r'^\s*([a-zA-Z_]\w*)\s*;')
    run_regex = re.compile(r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")

    for raw_line in code.split('\n'):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        clean_line = _remove_strings(stripped)

        m = direct_regex.match(clean_line)
        if m:
            name = m.group(1)
            if name not in _MATLAB_KEYWORDS:
                result.add(name.replace('.m', ''))
            continue

        for m in run_regex.finditer(clean_line):
            name = m.group(1)
            if name:
                result.add(name.replace('.m', ''))

    return sorted(result)


def get_line_number(text: str, offset: int = 0) -> int:
    """根据文本位置计算行号"""
    return text[:offset].count('\n') + 1


def clean_matlab_code(code: str) -> str:
    """清理MATLAB代码，移除注释和空行"""
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 移除行内注释
        comment_pos = line.find('%')
        if comment_pos != -1:
            line = line[:comment_pos]
        
        # 移除首尾空白
        line = line.strip()
        
        # 保留非空行
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def is_matlab_file(file_path: str) -> bool:
    """检查是否为MATLAB文件"""
    return file_path.lower().endswith('.m')


def extract_file_name_from_path(file_path: str) -> str:
    """从文件路径中提取文件名（不含扩展名）"""
    import os
    return os.path.splitext(os.path.basename(file_path))[0]


def create_node_id(node_type: str, name: str, file_path: str) -> str:
    """创建节点ID"""
    file_name = extract_file_name_from_path(file_path)
    return f"{node_type}_{name}_{file_name}.m"


def validate_matlab_syntax(code: str) -> bool:
    """简单的MATLAB语法验证"""
    # 检查基本的大括号平衡
    brace_count = 0
    for char in code:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count < 0:
                return False
    
    return brace_count == 0


def extract_function_signature(code: str) -> Optional[Dict[str, Any]]:
    """提取函数签名"""
    function_pattern = r'^\s*function\s+(?:\[([^\]]+)\]\s*=\s*)?(\w+)\s*\(([^)]*)\)'
    match = re.match(function_pattern, code.strip(), re.MULTILINE)
    
    if match:
        output_vars = match.group(1)
        func_name = match.group(2)
        input_params = match.group(3)
        
        return {
            'name': func_name,
            'input_parameters': [p.strip() for p in input_params.split(',') if p.strip()],
            'output_variables': [v.strip() for v in output_vars.split(',')] if output_vars else []
        }
    
    return None


def count_code_lines(code: str) -> int:
    """计算代码行数（排除注释和空行）"""
    lines = code.split('\n')
    code_lines = 0
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('%'):
            code_lines += 1
    
    return code_lines


def extract_comments_from_code(code: str) -> List[str]:
    """从代码中提取注释"""
    comments = []
    lines = code.split('\n')
    
    for line in lines:
        comment_pos = line.find('%')
        if comment_pos != -1:
            comment = line[comment_pos + 1:].strip()
            if comment:
                comments.append(comment)
    
    return comments