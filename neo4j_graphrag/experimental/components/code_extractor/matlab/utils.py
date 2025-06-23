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
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


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


def get_code_snippet(node: Dict[str, Any], max_length: int = 500) -> str:
    """获取代码片段"""
    code = node.get('properties', {}).get('code_snippet', '')
    if len(code) > max_length:
        return code[:max_length] + "..."
    return code


def extract_variables_from_code(code: str) -> List[str]:
    """从代码中提取变量名"""
    variables = set()
    
    # 变量定义模式
    var_def_patterns = [
        r'^\s*([a-zA-Z_]\w*)\s*=\s*',  # 简单赋值
        r'^\s*([a-zA-Z_]\w*)\s*=\s*\[',  # 数组赋值
        r'^\s*([a-zA-Z_]\w*)\s*=\s*{',   # 元胞数组赋值
        r'^\s*([a-zA-Z_]\w*)\s*=\s*struct\(',  # 结构体赋值
    ]
    
    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        for pattern in var_def_patterns:
            match = re.match(pattern, line)
            if match:
                variables.add(match.group(1))
                break
    
    return list(variables)


def extract_function_calls_from_code(code: str) -> List[str]:
    """从代码中提取函数调用"""
    function_calls = set()
    
    # 函数调用模式
    func_call_pattern = r'\b([a-zA-Z_]\w*)\s*\('
    
    for match in re.finditer(func_call_pattern, code):
        func_name = match.group(1)
        
        # 跳过MATLAB关键字
        if func_name in ['if', 'for', 'while', 'function', 'end', 'else', 'elseif', 'return', 'break', 'continue']:
            continue
        
        function_calls.add(func_name)
    
    return list(function_calls)


def extract_script_calls_from_code(code: str) -> List[str]:
    """从代码中提取脚本调用"""
    script_calls = set()
    
    # 脚本调用模式
    script_call_patterns = [
        r'^\s*([a-zA-Z_]\w*)\s*;',  # 直接脚本调用
        r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",  # run()函数调用
    ]
    
    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        
        for pattern in script_call_patterns:
            match = re.match(pattern, line)
            if match:
                script_calls.add(match.group(1))
                break
    
    return list(script_calls)


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