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
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class MatlabCodeParser:
    """MATLAB代码解析器，负责解析脚本和函数定义"""

    def __init__(self):
        self.current_file_path = ""
        self.current_content = ""
        self.line_offset = 0

    def parse_matlab_code(self, file_path: str, content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析MATLAB代码，返回节点和边的列表"""
        self.current_file_path = file_path
        self.current_content = content
        self.line_offset = 0
        
        nodes = []
        edges = []
        
        # 解析函数
        function_nodes, function_edges = self._parse_functions()
        nodes.extend(function_nodes)
        edges.extend(function_edges)
        
        # 解析脚本
        script_nodes, script_edges = self._parse_scripts()
        nodes.extend(script_nodes)
        edges.extend(script_edges)
        
        return nodes, edges

    def _parse_functions(self) -> Tuple[List[Dict], List[Dict]]:
        """解析函数定义"""
        nodes = []
        edges = []
        
        # 匹配函数定义
        function_pattern = r'^\s*function\s+(?:\[([^\]]+)\]\s*=\s*)?(\w+)\s*\(([^)]*)\)'
        
        for match in re.finditer(function_pattern, self.current_content, re.MULTILINE):
            output_vars = match.group(1)
            func_name = match.group(2)
            input_params = match.group(3)
            
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
                    'code_snippet': func_code[:500] + "..." if len(func_code) > 500 else func_code
                }
            }
            nodes.append(func_node)
            
            # 处理输入参数
            for param in func_node['properties']['input_parameters']:
                param_node = {
                    'id': f"var_{param}_{func_name}",
                    'labels': ['Variable'],
                    'properties': {
                        'name': param,
                        'type': 'parameter',
                        'function_name': func_name,
                        'file_path': self.current_file_path,
                        'scope_id': func_name,
                        'scope_type': 'function'
                    }
                }
                nodes.append(param_node)
                
                # 添加参数关系
                param_edge = {
                    'source': func_node['id'],
                    'target': param_node['id'],
                    'type': 'DEFINES',
                    'properties': {'parameter_type': 'input'}
                }
                edges.append(param_edge)
            
            # 处理输出变量
            for output_var in func_node['properties']['output_variables']:
                output_node = {
                    'id': f"var_{output_var}_{func_name}",
                    'labels': ['Variable'],
                    'properties': {
                        'name': output_var,
                        'type': 'output',
                        'function_name': func_name,
                        'file_path': self.current_file_path,
                        'scope_id': func_name,
                        'scope_type': 'function'
                    }
                }
                nodes.append(output_node)
                
                # 添加输出变量关系
                output_edge = {
                    'source': func_node['id'],
                    'target': output_node['id'],
                    'type': 'DEFINES',
                    'properties': {'parameter_type': 'output'}
                }
                edges.append(output_edge)
        
        return nodes, edges

    def _parse_scripts(self) -> Tuple[List[Dict], List[Dict]]:
        """解析脚本定义"""
        nodes = []
        edges = []
        
        # 检查是否为脚本文件（没有函数定义）
        function_pattern = r'^\s*function\s+'
        if not re.search(function_pattern, self.current_content, re.MULTILINE):
            # 这是一个脚本文件
            script_name = self.current_file_path.split('/')[-1].replace('.m', '')
            script_id = f"script_{script_name}_{self.current_file_path}"
            
            script_node = {
                'id': script_id,
                'labels': ['Script'],
                'properties': {
                    'name': script_name,
                    'file_path': self.current_file_path,
                    'start_line': 1,
                    'end_line': len(self.current_content.split('\n')),
                    'code_snippet': self.current_content[:500] + "..." if len(self.current_content) > 500 else self.current_content
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
        """从代码中提取变量定义和使用"""
        nodes = []
        edges = []
        
        # 变量定义模式
        var_def_patterns = [
            r'^\s*([a-zA-Z_]\w*)\s*=\s*',  # 简单赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*\[',  # 数组赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*{',   # 元胞数组赋值
            r'^\s*([a-zA-Z_]\w*)\s*=\s*struct\(',  # 结构体赋值
        ]
        
        # 变量使用模式
        var_use_pattern = r'\b([a-zA-Z_]\w*)\b'
        
        lines = code.split('\n')
        defined_vars = set()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            # 检查变量定义
            for pattern in var_def_patterns:
                match = re.match(pattern, line)
                if match:
                    var_name = match.group(1)
                    if var_name not in defined_vars:
                        defined_vars.add(var_name)
                        
                        var_node = {
                            'id': f"var_{var_name}_{parent_id}",
                            'labels': ['Variable'],
                            'properties': {
                                'name': var_name,
                                'type': 'local',
                                'defined_at': line_num + line_offset,
                                'parent_id': parent_id,
                                'file_path': file_path,
                                'scope_id': parent_id,
                                'scope_type': 'script' if 'script_' in parent_id else 'function'
                            }
                        }
                        nodes.append(var_node)
                        
                        # 添加定义关系
                        def_edge = {
                            'source': parent_id,
                            'target': var_node['id'],
                            'type': 'DEFINES',
                            'properties': {'line_number': line_num + line_offset}
                        }
                        edges.append(def_edge)
                    break
            
            # 检查变量使用
            for match in re.finditer(var_use_pattern, line):
                var_name = match.group(1)
                
                # 跳过MATLAB关键字和已定义的变量
                if (var_name in ['if', 'for', 'while', 'function', 'end', 'else', 'elseif', 'return', 'break', 'continue'] or
                    var_name in defined_vars):
                    continue
                
                # 检查是否为函数调用
                if '(' in line and var_name in line[:line.find('(')]:
                    continue
                
                # 创建变量使用节点
                use_node = {
                    'id': f"variable_use_{var_name}_{parent_id}_{line_num}",
                    'labels': ['VariableUse'],
                    'properties': {
                        'name': var_name,
                        'used_at': line_num + line_offset,
                        'parent_id': parent_id,
                        'file_path': file_path
                    }
                }
                nodes.append(use_node)
                
                # 添加使用关系
                use_edge = {
                    'source': parent_id,
                    'target': use_node['id'],
                    'type': 'USES',
                    'properties': {'line_number': line_num + line_offset}
                }
                edges.append(use_edge)
        
        return nodes, edges