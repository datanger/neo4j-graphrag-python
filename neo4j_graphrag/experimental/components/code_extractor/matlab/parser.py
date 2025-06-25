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
from .utils import _MATLAB_KEYWORDS, _remove_strings
import re
from .utils import get_matlab_builtin_functions
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MatlabCodeParser:
    """MATLAB 代码解析器，负责解析脚本、函数和类定义

    新增功能：
        1. 解析 `classdef` 块并生成 Class 节点。
        2. 解析 `properties` 块并生成 Variable 节点，并通过 HAS_PROPERTY 连接到 Class。
        3. 解析 `methods` 块并生成 Function 节点，并通过 HAS_METHOD 连接到 Class。
        4. 如有继承，生成 INHERITS_FROM 关系。
        5. 在函数解析阶段跳过 classdef 块中的方法，避免重复解析。
    """

    _BUILTIN_FUNCS = get_matlab_builtin_functions()
    """MATLAB代码解析器，负责解析脚本和函数定义"""

    def __init__(self):
        self.current_file_path = ""
        self.current_content = ""
        self.line_offset = 0
        # 存储每个 classdef 块在文件中的 (start, end) 位置范围，
        # 用于在全局函数解析阶段跳过类方法的解析，避免重复。
        self._class_ranges: List[Tuple[int, int]] = []

    def parse_matlab_code(self, file_path: str, content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析MATLAB代码，返回节点和边的列表"""
        self.current_file_path = file_path
        self.current_content = content
        self.line_offset = 0

        nodes = []
        edges = []

        # 解析类（需先解析，便于记录范围）
        class_nodes, class_edges = self._parse_classes()
        nodes.extend(class_nodes)
        edges.extend(class_edges)

        # 解析函数
        function_nodes, function_edges = self._parse_functions()
        nodes.extend(function_nodes)
        edges.extend(function_edges)

        # 解析脚本
        script_nodes, script_edges = self._parse_scripts()
        nodes.extend(script_nodes)
        edges.extend(script_edges)

        return nodes, edges

    def _parse_classes(self) -> Tuple[List[Dict], List[Dict]]:
        """解析 classdef 定义，包括属性和方法"""
        nodes: list[Dict] = []
        edges: list[Dict] = []

        class_pattern = re.compile(r"^\s*classdef\s+(?P<name>\w+)(?:\s*<\s*(?P<super>\w+))?", re.MULTILINE)

        for class_match in class_pattern.finditer(self.current_content):
            class_name = class_match.group("name")
            super_class = class_match.group("super")
            class_start = class_match.start()
            class_start_line = self._get_line_number(class_start)

            # 找到 classdef 对应的 end —— 简单地计数 end 出现，直到与 classdef 平衡
            lines_after = self.current_content[class_start:].split("\n")
            depth = 0
            end_pos = None
            for idx, line in enumerate(lines_after):
                if re.match(r"^\s*classdef\b", line):
                    depth += 1
                if re.match(r"^\s*end\b", line):
                    depth -= 1
                    if depth == 0:
                        # 当前位置是匹配的 end
                        end_pos = class_start + len("\n".join(lines_after[: idx + 1]))
                        break
            if end_pos is None:
                end_pos = len(self.current_content)
            class_end_line = self._get_line_number(end_pos)

            # 记录 class 范围用于后续跳过方法解析
            self._class_ranges.append((class_start, end_pos))

            class_id = f"class_{class_name}_{self.current_file_path}"
            class_node = {
                "id": class_id,
                "labels": ["Class"],
                "properties": {
                    "name": class_name,
                    "file_path": self.current_file_path,
                    "line_range": f"{class_start_line}-{class_end_line}",
                    "description": "",  # TODO: 可解析前置注释作为描述
                    "superclass": super_class or "",
                    "scope_id": Path(self.current_file_path).stem,
                    "scope_type": "script",
                },
            }
            nodes.append(class_node)

            # 继承关系
            if super_class:
                super_id = f"class_{super_class}_{self.current_file_path}"
                edges.append(
                    {
                        "source": class_id,
                        "target": super_id,
                        "type": "INHERITS_FROM",
                        "properties": {},
                    }
                )

            # 获取 class 块内容
            class_block = self.current_content[class_start:end_pos]

            # 解析 properties
            prop_block_pattern = re.compile(
                r"^\s*properties[^\n]*\n(?P<body>.*?)^\s*end\b",
                re.MULTILINE | re.DOTALL,
            )
            for prop_match in prop_block_pattern.finditer(class_block):
                body = prop_match.group("body")
                for line in body.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    # 属性名可能带默认值，如 `prop = 1`
                    prop_name_match = re.match(r"(\w+)", line)
                    if not prop_name_match:
                        continue
                    prop_name = prop_name_match.group(1)
                    var_id = f"var_{prop_name}_class_{class_name}"
                    var_node = {
                        "id": var_id,
                        "labels": ["Variable"],
                        "properties": {
                            "name": prop_name,
                            "file_path": self.current_file_path,
                            "scope_id": class_name,
                            "scope_type": "class",
                            "line_range": "",  # 可选解析
                        },
                    }
                    nodes.append(var_node)
                    edges.append(
                        {
                            "source": class_id,
                            "target": var_id,
                            "type": "HAS_PROPERTY",
                            "properties": {},
                        }
                    )

            # 解析 methods
            methods_block_pattern = re.compile(
                r"^\s*methods[^\n]*\n(?P<body>.*?)^\s*end\b",
                re.MULTILINE | re.DOTALL,
            )
            for meth_match in methods_block_pattern.finditer(class_block):
                body = meth_match.group("body")
                # 在 body 中寻找函数定义
                func_pattern = re.compile(
                    r"^\s*function\s+(?:\[([^\]]+)\]\s*=\s*)?(\w+)\s*\(([^)]*)\)",
                    re.MULTILINE,
                )
                for fmatch in func_pattern.finditer(body):
                    output_vars = fmatch.group(1)
                    func_name = fmatch.group(2)
                    input_params = fmatch.group(3)
                    # 绝对位置信息需要偏移: fmatch.start() relative to body start
                    abs_start = class_start + meth_match.start("body") + fmatch.start()
                    abs_end = abs_start + len(self._get_function_code_snippet(fmatch))
                    func_id = f"function_{func_name}_class_{class_name}_{self.current_file_path}"
                    func_node = {
                        "id": func_id,
                        "labels": ["Function"],
                        "properties": {
                            "name": func_name,
                            "file_path": self.current_file_path,
                            "line_range": f"{self._get_line_number(abs_start)}-{self._get_line_number(abs_end)}",
                            "description": "",
                            "parameters": input_params,
                            "returns": output_vars or "",
                            "scope_id": class_name,
                            "scope_type": "class",
                        },
                    }
                    nodes.append(func_node)
                    edges.append(
                        {
                            "source": class_id,
                            "target": func_id,
                            "type": "HAS_METHOD",
                            "properties": {},
                        }
                    )
                    # 提取方法体中的变量
                    method_code = self._get_function_code_snippet(fmatch)
                    var_nodes, var_edges = self.extract_variables_from_code(
                        method_code,
                        func_id,
                        self.current_file_path,
                        self._get_line_number(abs_start),
                    )
                    nodes.extend(var_nodes)
                    edges.extend(var_edges)

        return nodes, edges

    # ------------------------------------------------------------------
    # 以下为现有函数解析逻辑，但需跳过 classdef 范围中的函数（即方法）
    def _parse_functions(self) -> Tuple[List[Dict], List[Dict]]:
        """解析函数定义"""
        nodes = []
        edges = []

        # 匹配函数定义
        function_pattern = r'^\s*function\s+(?:\[([^\]]+)\]\s*=\s*)?(\w+)\s*\(([^)]*)\)'

        for match in re.finditer(function_pattern, self.current_content, re.MULTILINE):
            # 跳过 classdef 范围内的方法，避免重复解析
            if any(start <= match.start() < end for start, end in self._class_ranges):
                continue
            
            output_vars = match.group(1)
            input_params = match.group(3)

            # Handle MATLAB packages (directories starting with '+')
            package_parts = [part[1:] for part in Path(self.current_file_path).parts if part.startswith('+')]
            if package_parts:
                package_prefix = '.'.join(package_parts)
                original_func_name = match.group(2)
                func_name = f"{package_prefix}.{original_func_name}"
            else:
                func_name = match.group(2)

            # 获取函数代码片段
            func_code = self._get_function_code_snippet(match)

            # 创建函数节点
            func_node = {
                'id': f"function_{func_name}_{self.current_file_path}",
                'labels': ['Function'],
                'properties': {
                    'name': func_name,
                    'file_path': self.current_file_path,
                    'start_line': self._get_line_number(match.start()),
                    'end_line': self._get_line_number(match.start() + len(func_code)),
                    'input_parameters': [p.strip() for p in input_params.split(',') if p.strip()],
                    'output_variables': [v.strip() for v in output_vars.split(',')] if output_vars else [],
                    'code_snippet': func_code,
                    'scope_id': Path(self.current_file_path).stem,  # 脚本名称作为作用域ID
                    'scope_type': 'script'  # 函数的作用域是脚本
                }
            }
            nodes.append(func_node)

            # 处理输入参数
            for param in func_node['properties']['input_parameters']:
                param_node = {
                    'id': f"var_{param}_{func_name}_{self.current_file_path}",
                    'labels': ['Variable'],
                    'properties': {
                        'name': param,
                        'type': 'parameter',
                        'function_name': func_name,
                        'file_path': self.current_file_path,
                        'scope_id': func_name,
                        'scope_type': 'function',
                        'scope_binding': f"function:{func_name}"  # 明确的作用域绑定标识
                    }
                }
                nodes.append(param_node)

                # 创建DEFINES关系
                defines_edge = {
                    'source': func_node['id'],
                    'target': param_node['id'],
                    'type': 'DEFINES',
                    'properties': {
                        'definition_type': 'parameter',
                        'line_number': self._get_line_number(match.start())
                    }
                }
                edges.append(defines_edge)

            # 提取函数体中的变量
            func_body_nodes, func_body_edges = self.extract_variables_from_code(
                func_code,
                func_node['id'],
                self.current_file_path,
                self._get_line_number(match.start())
            )
            nodes.extend(func_body_nodes)
            edges.extend(func_body_edges)

        # 处理可能遗漏的函数定义（如 'function result = func_name(input)' 格式）
        alt_function_pattern = r'^\s*function\s+[^=]*=\s*(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(alt_function_pattern, self.current_content, re.MULTILINE):
            input_params = match.group(2)

            # Handle MATLAB packages (directories starting with '+')
            package_parts = [part[1:] for part in Path(self.current_file_path).parts if part.startswith('+')]
            if package_parts:
                package_prefix = '.'.join(package_parts)
                original_func_name = match.group(1)
                func_name = f"{package_prefix}.{original_func_name}"
            else:
                func_name = match.group(1)

            # 检查是否已经处理过这个函数
            existing_func_id = f"function_{func_name}_{self.current_file_path}"
            if any(node.get('id') == existing_func_id for node in nodes):
                continue

            # 获取函数代码片段
            func_code = self._get_function_code_snippet(match)

            # 创建函数节点
            func_node = {
                'id': existing_func_id,
                'labels': ['Function'],
                'properties': {
                    'name': func_name,
                    'file_path': self.current_file_path,
                    'start_line': self._get_line_number(match.start()),
                    'end_line': self._get_line_number(match.start() + len(func_code)),
                    'input_parameters': [p.strip() for p in input_params.split(',') if p.strip()],
                    'output_variables': [],
                    'code_snippet': func_code,
                    'scope_id': Path(self.current_file_path).stem,  # 脚本名称作为作用域ID
                    'scope_type': 'script'  # 函数的作用域是脚本
                }
            }
            nodes.append(func_node)

            # 处理输入参数
            for param in func_node['properties']['input_parameters']:
                param_node = {
                    'id': f"var_{param}_{func_name}_{self.current_file_path}",
                    'labels': ['Variable'],
                    'properties': {
                        'name': param,
                        'type': 'parameter',
                        'function_name': func_name,
                        'file_path': self.current_file_path,
                        'scope_id': func_name,
                        'scope_type': 'function',
                        'scope_binding': f"function:{func_name}"  # 明确的作用域绑定标识
                    }
                }
                nodes.append(param_node)

                # 创建DEFINES关系
                defines_edge = {
                    'source': func_node['id'],
                    'target': param_node['id'],
                    'type': 'DEFINES',
                    'properties': {
                        'definition_type': 'parameter',
                        'line_number': self._get_line_number(match.start())
                    }
                }
                edges.append(defines_edge)

            # 提取函数体中的变量
            func_body_nodes, func_body_edges = self.extract_variables_from_code(
                func_code,
                func_node['id'],
                self.current_file_path,
                self._get_line_number(match.start())
            )
            nodes.extend(func_body_nodes)
            edges.extend(func_body_edges)

        return nodes, edges

    def _parse_scripts(self) -> Tuple[List[Dict], List[Dict]]:
        """解析脚本定义"""
        nodes = []
        edges = []

        # 检查是否为脚本文件（包含函数定义）
        function_pattern = r'^\s*function\s+'
        functions_in_script = list(re.finditer(function_pattern, self.current_content, re.MULTILINE))

        print(f"DEBUG: Found {len(functions_in_script)} functions in script: {self.current_file_path}")

        if functions_in_script:
            # 这是一个包含函数的脚本文件
            script_name = Path(self.current_file_path).stem  # 获取不带扩展名的文件名
            script_id = f"script_{script_name}_{self.current_file_path}"

            script_node = {
                'id': script_id,
                'labels': ['Script'],
                'properties': {
                    'name': script_name,
                    'file_path': self.current_file_path,
                    'start_line': 1,
                    'end_line': len(self.current_content.split('\n')),
                    'code_snippet': self.current_content,
                    'scope_id': script_name,  # 脚本的作用域是自身
                    'scope_type': 'script'  # 脚本的作用域类型
                }
            }
            nodes.append(script_node)

            # 为脚本中定义的每个函数创建DEFINES关系
            for func_match in functions_in_script:
                # 获取完整的函数定义行
                lines = self.current_content.split('\n')
                line_number = self._get_line_number(func_match.start())
                if line_number <= len(lines):
                    func_line = lines[line_number - 1].strip()
                    func_name_match = re.search(r'function\s+(?:\[([^\]]+)\]\s*=\s*)?(\w+)\s*\(', func_line)
                    if func_name_match:
                        func_name = func_name_match.group(2)
                    else:
                        # Try alternative pattern for 'function result = func_name(input)' format
                        alt_match = re.search(r'function\s+[^=]*=\s*(\w+)\s*\(', func_line)
                        if alt_match:
                            func_name = alt_match.group(1)
                        else:
                            print(f"DEBUG: Could not extract function name from line: {func_line}")
                            continue
                    func_node_id = f"function_{func_name}_{self.current_file_path}"

                    print(f"DEBUG: Creating Script -[DEFINES]-> Function: {script_id} -> {func_node_id}")

                    # 创建脚本定义函数的关系
                    defines_edge = {
                        'source': script_id,
                        'target': func_node_id,
                        'type': 'DEFINES',
                        'properties': {
                            'definition_type': 'function_in_script',
                            'line_number': line_number
                        }
                    }
                    edges.append(defines_edge)
                else:
                    print(f"DEBUG: Line number {line_number} out of range")

            # 提取脚本中的变量（不包括函数体内的变量）
            script_code = self._extract_script_code_only()
            var_nodes, var_edges = self.extract_variables_from_code(
                script_code,
                script_id,
                self.current_file_path
            )
            nodes.extend(var_nodes)
            edges.extend(var_edges)
        else:
            # 这是一个纯脚本文件
            script_name = Path(self.current_file_path).stem  # 获取不带扩展名的文件名
            script_id = f"script_{script_name}_{self.current_file_path}"

            script_node = {
                'id': script_id,
                'labels': ['Script'],
                'properties': {
                    'name': script_name,
                    'file_path': self.current_file_path,
                    'start_line': 1,
                    'end_line': len(self.current_content.split('\n')),
                    'code_snippet': self.current_content,
                    'scope_id': script_name,  # 脚本的作用域是自身
                    'scope_type': 'script'  # 脚本的作用域类型
                }
            }
            nodes.append(script_node)

            # 提取脚本中的变量
            var_nodes, var_edges = self.extract_variables_from_code(
                self.current_content,
                script_id,
                self.current_file_path
            )
            nodes.extend(var_nodes)
            edges.extend(var_edges)

        return nodes, edges

    def _extract_script_code_only(self) -> str:
        """提取脚本代码，排除函数定义部分"""
        lines = self.current_content.split('\n')
        script_lines = []
        in_function = False
        function_depth = 0

        for line in lines:
            # 检查是否进入函数
            if re.match(r'^\s*function\s+', line):
                in_function = True
                function_depth = 1
                continue

            # 检查函数结束
            if in_function:
                # 计算大括号深度
                function_depth += line.count('{') - line.count('}')
                if re.match(r'^\s*end\s*$', line.strip()) and function_depth <= 0:
                    in_function = False
                    function_depth = 0
                    continue
                continue

            # 只有在函数外的代码才添加到脚本代码中
            if not in_function:
                script_lines.append(line)

        return '\n'.join(script_lines)

    def _get_function_code_snippet(self, func_match) -> str:
        """获取函数的完整代码片段"""
        start_pos = func_match.start()

        # 查找函数的结束位置
        lines = self.current_content[start_pos:].split('\n')
        brace_count = 0
        end_pos = start_pos

        for i, line in enumerate(lines):
            # 计算大括号
            brace_count += line.count('{') - line.count('}')

            # 如果遇到end关键字且大括号平衡，则找到函数结束
            if (re.match(r'^\s*end\s*$', line.strip()) and brace_count <= 0):
                end_pos = start_pos + len('\n'.join(lines[:i+1]))
                break

        if end_pos == start_pos:
            # 如果没有找到明确的结束，使用整个文件
            end_pos = len(self.current_content)

        return self.current_content[start_pos:end_pos]

    def _get_line_number(self, text_pos: int) -> int:
        """根据文本位置计算行号"""
        return self.current_content[:text_pos].count('\n') + 1

    def extract_variables_from_code(self, code: str, parent_id: str, file_path: str, line_offset: int = 0) -> Tuple[List[Dict], List[Dict]]:
        """从代码中提取变量定义、使用、赋值和修改关系"""
        nodes = []
        edges = []

        # 确定作用域类型
        scope_type = 'function' if parent_id.startswith('function_') else 'script'

        # 变量定义模式
        var_def_patterns = [
            r'^\s*([a-zA-Z_]\w*)\s*=\s*',  # 简单赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*\[',  # 数组赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*{',   # 元胞数组赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*struct\(',  # 结构体赋值
        ]

        # 函数调用模式
        func_call_pattern = r'\b([a-zA-Z_]\w*)\s*\('

        # 脚本调用模式
        script_call_patterns = [
            r'^\s*run\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',  # run('script.m')
            r'^\s*([a-zA-Z_]\w*)\s*;',  # script_name;
        ]

        # 变量使用模式（不包括定义）
        var_use_pattern = r'\b([a-zA-Z_]\w*)\b'

        # 预处理：去除整行注释，保留长度一致
        code_lines = []
        for ln in code.split('\n'):
            if ln.lstrip().startswith('%'):
                code_lines.append('')  # keep line count
            else:
                # mask strings before further regex to avoid "a='foo'" counting 'foo'
                code_lines.append(_remove_strings(ln))
        code_no_comments = '\n'.join(code_lines)

        # 收集所有变量定义
        defined_vars = set()
        created_var_nodes = {}  # 用于跟踪已创建的变量节点

        for pattern in var_def_patterns:
            for match in re.finditer(pattern, code_no_comments, re.MULTILINE):
                var_name = match.group(1)
                defined_vars.add(var_name)

                # 创建变量节点，确保作用域绑定
                var_node_id = f"var_{var_name}_{scope_type}_{parent_id}"

                # 检查是否已创建过该变量节点
                if var_node_id not in created_var_nodes:
                    var_node = {
                        'id': var_node_id,
                        'labels': ['Variable'],
                        'properties': {
                            'name': var_name,
                            'type': 'local',
                            'file_path': file_path,
                            'scope_id': parent_id,
                            'scope_type': scope_type,
                            'definition_line': self._get_line_number(match.start()) + line_offset,
                            'scope_binding': f"{scope_type}:{parent_id}"  # 明确的作用域绑定标识
                        }
                    }
                    nodes.append(var_node)
                    created_var_nodes[var_node_id] = var_node

                # 创建DEFINES关系 - 确保每个变量都与其作用域绑定
                defines_edge = {
                    'source': parent_id,
                    'target': var_node_id,
                    'type': 'DEFINES',
                    'properties': {
                        'definition_type': 'assignment',
                        'line_number': self._get_line_number(match.start()) + line_offset
                    }
                }
                edges.append(defines_edge)

        # 收集所有函数调用名称用于后续过滤变量
        function_call_names = {m.group(1) for m in re.finditer(func_call_pattern, code_no_comments, re.MULTILINE)}

        # 处理变量使用 - 只处理同作用域内的变量使用
        used_vars = set()
        for match in re.finditer(var_use_pattern, code_no_comments, re.MULTILINE):
            var_name = match.group(1)
            # 跳过函数调用名、已定义变量、MATLAB关键字、数字等
            if (var_name in function_call_names or
                var_name in defined_vars or
                var_name in used_vars or
                var_name in _MATLAB_KEYWORDS or
                var_name.isdigit() or
                var_name.startswith('_') or
                len(var_name) <= 1):
                continue
            used_vars.add(var_name)

            # 检查变量是否在当前作用域中已定义
            var_node_id = f"var_{var_name}_{scope_type}_{parent_id}"
            var_node_exists = var_node_id in created_var_nodes

            # 只处理在当前作用域中已定义的变量
            # 跨作用域的变量使用将在后处理器中处理
            if var_name in defined_vars:
                # 创建变量节点（如果不存在）
                if not var_node_exists:
                    var_node = {
                        'id': var_node_id,
                        'labels': ['Variable'],
                        'properties': {
                            'name': var_name,
                            'type': 'local',
                            'file_path': file_path,
                            'scope_id': parent_id,
                            'scope_type': scope_type,
                            'usage_line': self._get_line_number(match.start()) + line_offset,
                            'scope_binding': f"{scope_type}:{parent_id}"  # 明确的作用域绑定标识
                        }
                    }
                    nodes.append(var_node)
                    created_var_nodes[var_node_id] = var_node

                # 只创建同作用域内的USES关系
                uses_edge = {
                    'source': parent_id,
                    'target': var_node_id,
                    'type': 'USES',
                    'properties': {
                        'usage_type': 'local_variable',
                        'line_number': self._get_line_number(match.start()) + line_offset
                    }
                }
                edges.append(uses_edge)
            else:
                # 对于未在当前作用域定义的变量，只创建变量节点，不创建USES关系
                # 这些跨作用域的USES关系将在后处理器中处理
                if not var_node_exists:
                    var_node = {
                        'id': var_node_id,
                        'labels': ['Variable'],
                        'properties': {
                            'name': var_name,
                            'type': 'external',
                            'file_path': file_path,
                            'scope_id': parent_id,
                            'scope_type': scope_type,
                            'usage_line': self._get_line_number(match.start()) + line_offset,
                            'scope_binding': f"{scope_type}:{parent_id}"  # 明确的作用域绑定标识
                        }
                    }
                    nodes.append(var_node)
                    created_var_nodes[var_node_id] = var_node

        # 处理函数调用 - 只创建CALLS关系，不创建USES关系
        for match in re.finditer(func_call_pattern, code_no_comments, re.MULTILINE):
            func_name = match.group(1)
            # 跳过已定义的变量和MATLAB内置函数
            if (func_name not in defined_vars and
                 func_name.lower() not in self._BUILTIN_FUNCS and
                 func_name not in used_vars and
                not func_name.isdigit()):

                # 跳过自调用（函数体内调用与自身同名的函数 -> 递归或误判）
                current_func_name = parent_id.split('_')[1] if scope_type == 'function' else None
                if scope_type == 'function' and func_name == current_func_name:
                    continue

                # 创建CALLS关系（目标节点会在后处理中处理）
                calls_edge = {
                    'source': parent_id,
                    'target': f"function_{func_name}_{file_path}",
                    'type': 'CALLS',
                    'properties': {
                        'call_type': 'function_call',
                        'function_name': func_name,
                        'line_number': self._get_line_number(match.start()) + line_offset
                    }
                }
                # CALLS edge generation deferred to post-processing
                pass

        # 处理脚本调用 - 只创建CALLS关系，不创建USES关系
        for pattern in script_call_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                script_name = match.group(1)
                if script_name and not script_name.isdigit():
                    # 移除.m扩展名以获取脚本名称
                    script_name_clean = script_name.replace('.m', '')
                    # 创建CALLS关系（目标节点会在后处理中处理）
                    calls_edge = {
                        'source': parent_id,
                        'target': f"script_{script_name_clean}_placeholder",  # 占位符，后处理器会解析
                        'type': 'CALLS',
                        'properties': {
                            'call_type': 'script_call',
                            'script_name': script_name,
                            'line_number': self._get_line_number(match.start()) + line_offset
                        }
                    }
                    # CALLS edge generation deferred to post-processing
                pass

        # 处理变量赋值关系 - 只处理同作用域内的赋值
        assignment_pattern = r'([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)'
        for match in re.finditer(assignment_pattern, code, re.MULTILINE):
            target_var = match.group(1)
            source_var = match.group(2)

            # 检查源变量和目标变量是否都在当前作用域中定义或使用
            target_var_id = f"var_{target_var}_{scope_type}_{parent_id}"
            source_var_id = f"var_{source_var}_{scope_type}_{parent_id}"

            # 确保两个变量都在当前作用域中
            target_in_scope = target_var_id in created_var_nodes
            source_in_scope = source_var_id in created_var_nodes

            # 只创建同作用域内的ASSIGNED_TO关系
            # 跨作用域的ASSIGNED_TO关系将在后处理器中处理
            if target_in_scope and source_in_scope:
                assigned_edge = {
                    'source': source_var_id,
                    'target': target_var_id,
                    'type': 'ASSIGNED_TO',
                    'properties': {
                        'assignment_type': 'local_assignment',
                        'line_number': self._get_line_number(match.start()) + line_offset
                    }
                }
                edges.append(assigned_edge)

        # 处理变量修改关系（通过函数调用）
        modify_pattern = r'\[([^\]]+)\]\s*=\s*([a-zA-Z_]\w*)\s*\('
        for match in re.finditer(modify_pattern, code, re.MULTILINE):
            modified_vars = [v.strip() for v in match.group(1).split(',')]
            func_name = match.group(2)

            for var_name in modified_vars:
                if var_name and not var_name.isdigit():
                    # 创建MODIFIES关系
                    modifies_edge = {
                        'source': parent_id,
                        'target': f"var_{var_name}_{scope_type}_{parent_id}",
                        'type': 'MODIFIES',
                        'properties': {
                            'modification_type': 'function_output',
                            'function_name': func_name,
                            'line_number': self._get_line_number(match.start()) + line_offset
                        }
                    }
                    edges.append(modifies_edge)

        return nodes, edges
