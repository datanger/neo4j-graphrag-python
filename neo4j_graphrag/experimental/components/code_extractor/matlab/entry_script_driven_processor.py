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

"""
启动脚本驱动的MATLAB跨作用域关系处理器

这个模块专门处理基于启动脚本执行顺序的跨作用域变量调用关系。
充分利用MATLAB脚本的特殊性，考虑变量空间和执行顺序。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from pathlib import Path

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from .registry import get_global_registry

logger = logging.getLogger(__name__)


class EntryScriptDrivenProcessor:
    """启动脚本驱动的跨作用域关系处理器"""

    def __init__(self):
        self.registry = get_global_registry()
        self.entry_script_path = None
        self.entry_script_content = None
        self.execution_flow = []
        self.variable_scope_timeline = {}  # 变量作用域时间线
        self.script_execution_order = []  # 脚本执行顺序
        self.variable_definitions = defaultdict(dict)  # script_name -> {var_name: line_info}
        self.variable_usage = defaultdict(dict)  # script_name -> {var_name: [line_info]}
        self.cross_scope_relationships = []  # 跨作用域关系缓存

    def set_entry_script(self, entry_script_path: str):
        """设置启动脚本并分析执行流程"""
        self.entry_script_path = entry_script_path
        if entry_script_path:
            try:
                with open(entry_script_path, 'r', encoding='utf-8') as f:
                    self.entry_script_content = f.read()
                self._analyze_entry_script()
            except Exception as e:
                logger.error(f"Failed to read entry script {entry_script_path}: {e}")
                self.entry_script_content = ""

    def _analyze_entry_script(self):
        """分析启动脚本，构建执行流程和变量作用域时间线"""
        if not self.entry_script_content:
            return

        lines = self.entry_script_content.split('\n')
        current_scope = "entry_script"
        self.execution_flow = []
        self.variable_scope_timeline = {}
        self.script_execution_order = []

        # 记录启动脚本中的变量定义
        entry_vars = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # 1. 检测函数调用赋值：var = func 或 var = func(...)
            func_assign_patterns = [
                r'^\s*([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\s*$',      # var = func
                r'^\s*([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\s*\(',     # var = func(...)
            ]

            for pattern in func_assign_patterns:
                match = re.match(pattern, line)
                if match:
                    var_name = match.group(1)
                    func_name = match.group(2)

                    # 记录变量定义
                    entry_vars.add(var_name)
                    self.variable_scope_timeline[var_name] = {
                        'defined_at': line_num,
                        'scope': current_scope,
                        'available_from': line_num,
                        'available_until': float('inf'),
                        'assigned_from_function': func_name,
                        'has_parameters': '(' in line
                    }

                    # 记录函数调用
                    self._record_function_call(func_name, line_num, current_scope, has_params='(' in line)
                    print(f"DEBUG: Detected function call: {var_name} = {func_name} at line {line_num}")
                    break
            else:
                # 2. 检测直接函数调用：func 或 func(...)
                direct_func_patterns = [
                    r'^\s*([a-zA-Z_]\w*)\s*$',      # func
                    r'^\s*([a-zA-Z_]\w*)\s*\(',     # func(...)
                ]

                for pattern in direct_func_patterns:
                    match = re.match(pattern, line)
                    if match:
                        func_name = match.group(1)
                        self._record_function_call(func_name, line_num, current_scope, has_params='(' in line)
                        print(f"DEBUG: Detected direct function call: {func_name} at line {line_num}")
                        break
                else:
                    # 3. 检测脚本调用：script;
                    script_call_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*;', line)
                    if script_call_match:
                        script_name = script_call_match.group(1)
                        self._record_script_call(script_name, line_num, current_scope)
                        continue

                    # 4. 检测run()调用
                    run_call_match = re.search(r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
                    if run_call_match:
                        script_name = run_call_match.group(1).replace('.m', '')
                        self._record_script_call(script_name, line_num, current_scope)
                        continue

                    # 5. 检测普通变量定义
                    var_def_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*=', line)
                    if var_def_match:
                        var_name = var_def_match.group(1)
                        entry_vars.add(var_name)
                        self.variable_scope_timeline[var_name] = {
                            'defined_at': line_num,
                            'scope': current_scope,
                            'available_from': line_num,
                            'available_until': float('inf')
                        }
                        continue

            # 检测变量使用
            for var_name in entry_vars:
                if re.search(r'\b' + re.escape(var_name) + r'\b', line):
                    if var_name in self.variable_scope_timeline:
                        self.variable_scope_timeline[var_name]['usage_lines'] = \
                            self.variable_scope_timeline[var_name].get('usage_lines', []) + [line_num]

        logger.info(f"Entry script analysis: {len(self.execution_flow)} calls, {len(entry_vars)} variables")

    def _record_function_call(self, func_name: str, line_num: int, current_scope: str, has_params: bool):
        """记录函数调用"""
        self.execution_flow.append({
            'type': 'function_call',
            'function_name': func_name,
            'line_number': line_num,
            'scope': current_scope,
            'available_variables': set(self.variable_scope_timeline.keys()),
            'has_parameters': has_params
        })

    def _record_script_call(self, script_name: str, line_num: int, current_scope: str):
        """记录脚本调用"""
        self.execution_flow.append({
            'type': 'script_call',
            'script_name': script_name,
            'line_number': line_num,
            'scope': current_scope,
            'available_variables': set(self.variable_scope_timeline.keys())
        })

        if script_name not in self.script_execution_order:
            self.script_execution_order.append(script_name)

    def register_script_variables(self, script_name: str, script_content: str):
        """注册脚本中的变量定义和使用"""
        lines = script_content.split('\n')

        # 检测变量定义
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # 变量定义模式
            var_def_patterns = [
                r'^\s*([a-zA-Z_]\w*)\s*=\s*',  # 简单赋值
                r'^\s*([a-zA-Z_]\w*)\s*=\s*\[',  # 数组赋值
                r'^\s*([a-zA-Z_]\w*)\s*=\s*{',   # 元胞数组赋值
                r'^\s*([a-zA-Z_]\w*)\s*=\s*struct\(',  # 结构体赋值
            ]

            for pattern in var_def_patterns:
                match = re.match(pattern, line)
                if match:
                    var_name = match.group(1)
                    if len(var_name) >= 2:  # 过滤短变量名
                        self.variable_definitions[script_name][var_name] = {
                            'line': line_num,
                            'pattern': pattern,
                            'full_line': line
                        }
                    break

        # 检测变量使用
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # 变量使用模式（排除函数调用）
            var_use_pattern = r'\b([a-zA-Z_]\w{1,})\b'
            for match in re.finditer(var_use_pattern, line):
                var_name = match.group(1)
                if len(var_name) >= 2 and not self._is_function_call(line, var_name):
                    if var_name not in self.variable_usage[script_name]:
                        self.variable_usage[script_name][var_name] = []
                    self.variable_usage[script_name][var_name].append({
                        'line': line_num,
                        'full_line': line
                    })

    def _is_function_call(self, line: str, var_name: str) -> bool:
        """检查是否为函数调用"""
        # 检查是否跟随括号
        pattern = r'\b' + re.escape(var_name) + r'\s*\('
        return bool(re.search(pattern, line))

    def process_cross_scope_relationships(self, graph: Neo4jGraph) -> Neo4jGraph:
        """处理跨作用域关系，基于启动脚本执行顺序"""
        if not self.entry_script_path:
            logger.warning("No entry script available")
            return graph

        logger.info(f"Processing cross-scope relationships based on entry script: {self.entry_script_path}")
        logger.info(f"Execution flow: {len(self.execution_flow)} calls")
        logger.info(f"Script execution order: {self.script_execution_order}")

        # 构建高效的查找索引
        self._build_lookup_indexes(graph)

        # 1. 处理启动脚本中的函数调用关系
        self._process_entry_script_function_calls(graph)

        # 2. 处理跨作用域变量USES关系
        self._process_cross_scope_variable_uses(graph)

        # 3. 处理跨作用域变量ASSIGNED_TO关系
        self._process_cross_scope_variable_assignments(graph)

        # 4. 处理跨作用域函数调用
        self._process_cross_scope_function_calls(graph)

        # 5. 处理跨作用域脚本调用
        self._process_cross_scope_script_calls(graph)

        logger.info(f"Added {len(self.cross_scope_relationships)} cross-scope relationships")
        return graph

    def _build_lookup_indexes(self, graph: Neo4jGraph):
        """构建高效的查找索引"""
        self.node_by_id = {node.id: node for node in graph.nodes}
        self.script_nodes = {node.id: node for node in graph.nodes if node.label == 'Script'}
        self.function_nodes = {node.id: node for node in graph.nodes if node.label == 'Function'}
        self.variable_nodes = {node.id: node for node in graph.nodes if node.label == 'Variable'}

        # 按名称索引
        self.script_by_name = {}
        self.function_by_name = {}
        self.variables_by_name = defaultdict(list)

        # 获取启动脚本的文件名（不含扩展名）
        entry_script_name = None
        if self.entry_script_path:
            entry_script_name = Path(self.entry_script_path).stem

        print(f"DEBUG: Looking for entry script: {entry_script_name}")
        print(f"DEBUG: Available script nodes:")

        for node in graph.nodes:
            if node.label == 'Script':
                script_name = node.properties.get('name', '')
                print(f"  Script node: {script_name} (id: {node.id})")
                if script_name:
                    self.script_by_name[script_name] = node
                    # 更健壮地识别启动脚本：忽略大小写和首尾空格
                    if entry_script_name and script_name.strip().lower() == entry_script_name.strip().lower():
                        self.script_by_name["entry_script"] = node
                        print(f"Found entry script node: {script_name} -> entry_script")
                    elif script_name.strip().lower() == "entry_script":
                        self.script_by_name["entry_script"] = node
                        print(f"Found entry script node with name 'entry_script'")

            elif node.label == 'Function':
                func_name = node.properties.get('name', '')
                if func_name:
                    self.function_by_name[func_name] = node
            elif node.label == 'Variable':
                var_name = node.properties.get('name', '')
                if var_name:
                    self.variables_by_name[var_name].append(node)

        # 如果没有找到启动脚本节点，创建一个虚拟的启动脚本节点用于关系生成
        if "entry_script" not in self.script_by_name and self.entry_script_path:
            print(f"Warning: No entry script node found, creating virtual entry script node")
            # 这里可以创建一个虚拟节点，或者跳过启动脚本的关系生成
            # 为了简化，我们暂时跳过启动脚本的关系生成

    def _process_entry_script_function_calls(self, graph: Neo4jGraph):
        """处理启动脚本中的函数调用关系"""
        logger.info("Processing entry script function calls")

        if "entry_script" not in self.script_by_name:
            logger.warning("Entry script node not found, skipping entry script function call processing")
            return

        entry_script_node = self.script_by_name["entry_script"]

        # 处理启动脚本执行流程中的函数调用
        for call_info in self.execution_flow:
            if call_info['type'] == 'function_call':
                func_name = call_info['function_name']
                line_num = call_info['line_number']
                has_params = call_info.get('has_params', False)

                # 检查是否为MATLAB内置函数或关键字
                if self._is_matlab_builtin_or_keyword(func_name):
                    continue

                # 智能判断：优先查找函数，如果找不到再查找脚本
                target_node = None
                call_type = None

                if func_name in self.function_by_name:
                    target_node = self.function_by_name[func_name]
                    call_type = 'entry_script_function_call'
                elif func_name in self.script_by_name:
                    target_node = self.script_by_name[func_name]
                    call_type = 'entry_script_script_call'

                if target_node:
                    # 创建启动脚本调用函数/脚本的关系
                    relationship = Neo4jRelationship(
                        start_node_id=entry_script_node.id,
                        end_node_id=target_node.id,
                        type='CALLS',
                        properties={
                            'call_type': call_type,
                            'line_number': line_num,
                            'source_script': 'entry_script',
                            'target_name': func_name,
                            'has_parameters': has_params,
                            'execution_order_based': True,
                            'post_processed': True
                        }
                    )

                    graph.relationships.append(relationship)
                    self.cross_scope_relationships.append(relationship)
                    logger.info(f"Added entry script call: entry_script -> {func_name} ({call_type})")

                # 处理变量赋值自函数的情况
                for var_name, var_info in self.variable_scope_timeline.items():
                    if var_info.get('assigned_from_function') == func_name:
                        # 创建变量USES函数的关系
                        if target_node:
                            # 获取或创建变量节点
                            var_node_id = self._get_or_create_variable_node(graph, var_name, "entry_script")
                            if var_node_id:
                                relationship = Neo4jRelationship(
                                    start_node_id=var_node_id,
                                    end_node_id=target_node.id,
                                    type='USES',
                                    properties={
                                        'usage_type': 'variable_assigned_from_function',
                                        'variable_name': var_name,
                                        'function_name': func_name,
                                        'line_number': line_num,
                                        'has_parameters': has_params,
                                        'execution_order_based': True,
                                        'post_processed': True
                                    }
                                )

                                graph.relationships.append(relationship)
                                self.cross_scope_relationships.append(relationship)
                                logger.info(f"Added variable-function relationship: {var_name} -> {func_name}")

    def _process_cross_scope_variable_uses(self, graph: Neo4jGraph):
        """处理跨作用域变量USES关系"""
        logger.info("Processing cross-scope variable USES relationships")

        # 处理启动脚本中的变量使用
        if "entry_script" in self.script_by_name:
            entry_script_node = self.script_by_name["entry_script"]
            entry_content = entry_script_node.properties.get('code_snippet', '')

            if entry_content:
                # 获取启动脚本中使用的变量
                used_vars = self.variable_usage.get("entry_script", {})

                for var_name, usage_info in used_vars.items():
                    # 查找变量来源（基于执行顺序）
                    source_script = self._find_variable_source(var_name, "entry_script")

                    if source_script and source_script != "entry_script":
                        # 创建跨作用域USES关系
                        self._create_cross_scope_uses_relationship(
                            graph, entry_script_node.id, var_name, source_script
                        )

        # 为每个脚本处理变量使用
        for script_name in self.script_execution_order:
            if script_name not in self.script_by_name:
                continue

            script_node = self.script_by_name[script_name]
            script_content = script_node.properties.get('code_snippet', '')

            if not script_content:
                continue

            # 获取脚本中使用的变量
            used_vars = self.variable_usage.get(script_name, {})

            for var_name, usage_info in used_vars.items():
                # 查找变量来源（基于执行顺序）
                source_script = self._find_variable_source(var_name, script_name)

                if source_script and source_script != script_name:
                    # 创建跨作用域USES关系
                    self._create_cross_scope_uses_relationship(
                        graph, script_node.id, var_name, source_script
                    )

    def _find_variable_source(self, var_name: str, usage_script: str) -> Optional[str]:
        """根据执行顺序查找变量来源"""
        # 检查启动脚本
        if var_name in self.variable_scope_timeline:
            return "entry_script"

        # 根据执行顺序查找
        try:
            usage_index = self.script_execution_order.index(usage_script)
        except ValueError:
            # 如果usage_script不在执行顺序中，可能是启动脚本
            if usage_script == "entry_script":
                # 在启动脚本之前执行的脚本中查找变量定义
                for candidate_script in self.script_execution_order:
                    if (candidate_script in self.variable_definitions and
                        var_name in self.variable_definitions[candidate_script]):
                        return candidate_script
            return None

        # 在usage_script之前执行的脚本中查找变量定义
        for i in range(usage_index):
            candidate_script = self.script_execution_order[i]
            if (candidate_script in self.variable_definitions and
                var_name in self.variable_definitions[candidate_script]):
                return candidate_script

        return None

    def _create_cross_scope_uses_relationship(self, graph: Neo4jGraph, caller_id: str,
                                            var_name: str, source_script: str):
        """创建跨作用域USES关系"""
        # 查找或创建目标变量节点
        target_var_id = self._get_or_create_variable_node(graph, var_name, source_script)

        if target_var_id:
            # 创建USES关系
            relationship = Neo4jRelationship(
                start_node_id=caller_id,
                end_node_id=target_var_id,
                type='USES',
                properties={
                    'usage_type': 'cross_scope_execution_order',
                    'variable_name': var_name,
                    'source_script': source_script,
                    'target_script': self._get_script_name_from_id(caller_id),
                    'execution_order_based': True,
                    'post_processed': True
                }
            )

            graph.relationships.append(relationship)
            self.cross_scope_relationships.append(relationship)

    def _get_or_create_variable_node(self, graph: Neo4jGraph, var_name: str, source_script: str) -> Optional[str]:
        """获取或创建变量节点，确保正确的作用域绑定"""
        # 首先查找现有的变量节点，确保作用域匹配
        for var_node in self.variables_by_name.get(var_name, []):
            node_scope_id = var_node.properties.get('scope_id', '')
            node_scope_type = var_node.properties.get('scope_type', '')

            # 检查作用域是否匹配
            if (node_scope_id == source_script and
                node_scope_type in ['script', 'function']):
                return var_node.id

        # 如果没有找到匹配的变量节点，创建新的
        if source_script == "entry_script":
            var_id = f"var_{var_name}_script_entry_script"
            file_path = self.entry_script_path
            scope_type = 'script'
        else:
            # 检查source_script是脚本还是函数
            if source_script in self.script_by_name:
                var_id = f"var_{var_name}_script_{source_script}"
                file_path = f"{source_script}.m"
                scope_type = 'script'
            elif source_script in self.function_by_name:
                var_id = f"var_{var_name}_function_{source_script}"
                file_path = self.function_by_name[source_script].properties.get('file_path', f"{source_script}.m")
                scope_type = 'function'
            else:
                # 默认为脚本
                var_id = f"var_{var_name}_script_{source_script}"
                file_path = f"{source_script}.m"
                scope_type = 'script'

        # 检查是否已存在
        if var_id in self.node_by_id:
            return var_id

        # 创建新节点，确保正确的作用域绑定
        var_node = Neo4jNode(
            id=var_id,
            label='Variable',
            properties={
                'name': var_name,
                'type': 'cross_scope',
                'scope_id': source_script,
                'scope_type': scope_type,
                'file_path': file_path,
                'created_by_post_processor': True,
                'scope_binding': f"{scope_type}:{source_script}"  # 明确的作用域绑定标识
            }
        )

        graph.nodes.append(var_node)
        self.node_by_id[var_id] = var_node

        # 确保variables_by_name中有该变量名的列表
        if var_name not in self.variables_by_name:
            self.variables_by_name[var_name] = []
        self.variables_by_name[var_name].append(var_node)

        logger.debug(f"Created variable node: {var_id} with scope binding: {scope_type}:{source_script}")
        return var_id

    def _process_cross_scope_variable_assignments(self, graph: Neo4jGraph):
        """处理跨作用域变量ASSIGNED_TO关系"""
        logger.info("Processing cross-scope variable ASSIGNED_TO relationships")

        # 预编译正则表达式
        assignment_pattern = re.compile(r'\s*([a-zA-Z_]\w*)\s*=\s*([^;\n]+);?')
        variable_pattern = re.compile(r'\b([a-zA-Z_]\w{2,})\b')

        # 收集所有有意义的变量名
        meaningful_vars = {name: nodes for name, nodes in self.variables_by_name.items()
                          if len(name) >= 3}

        created_assignments = set()

        # 处理跨作用域变量赋值关系
        for node in graph.nodes:
            if node.label not in ['Script', 'Function']:
                continue

            content = node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找赋值语句
            for match in assignment_pattern.finditer(content):
                target_var = match.group(1)
                rhs_expr = match.group(2)

                # 提取右侧表达式中的变量
                rhs_vars = set()
                for var_match in variable_pattern.finditer(rhs_expr):
                    var_name = var_match.group(1)
                    if var_name in meaningful_vars and var_name != target_var:
                        rhs_vars.add(var_name)

                # 为每个变量对创建ASSIGNED_TO关系
                for rhs_var in rhs_vars:
                    assignment_key = (target_var, rhs_var)
                    if assignment_key not in created_assignments:
                        # 检查是否为跨作用域关系
                        if self._is_cross_scope_assignment(target_var, rhs_var, node):
                            self._create_cross_scope_assignment_relationship(
                                graph, target_var, rhs_var, node
                            )
                            created_assignments.add(assignment_key)

    def _is_cross_scope_assignment(self, target_var: str, source_var: str, node: Neo4jNode) -> bool:
        """检查是否为跨作用域赋值"""
        # 获取节点所属的脚本
        node_script = self._get_script_name_from_id(node.id)

        # 检查源变量是否来自不同的脚本
        source_script = self._find_variable_source(source_var, node_script)

        return source_script and source_script != node_script

    def _create_cross_scope_assignment_relationship(self, graph: Neo4jGraph, target_var: str,
                                                  source_var: str, node: Neo4jNode):
        """创建跨作用域ASSIGNED_TO关系"""
        node_script = self._get_script_name_from_id(node.id)
        source_script = self._find_variable_source(source_var, node_script)

        if not source_script:
            return

        # 获取或创建变量节点
        target_var_id = self._get_or_create_variable_node(graph, target_var, node_script)
        source_var_id = self._get_or_create_variable_node(graph, source_var, source_script)

        if target_var_id and source_var_id:
            relationship = Neo4jRelationship(
                start_node_id=source_var_id,
                end_node_id=target_var_id,
                type='ASSIGNED_TO',
                properties={
                    'assignment_type': 'cross_scope_execution_order',
                    'source_script': source_script,
                    'target_script': node_script,
                    'execution_order_based': True,
                    'post_processed': True
                }
            )

            graph.relationships.append(relationship)
            self.cross_scope_relationships.append(relationship)

    def _process_cross_scope_function_calls(self, graph: Neo4jGraph):
        """处理跨作用域函数调用"""
        logger.info("Processing cross-scope function calls")

        # 预编译正则表达式
        func_call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')

        for node in graph.nodes:
            if node.label not in ['Script', 'Function']:
                continue

            content = node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找函数调用
            for match in func_call_pattern.finditer(content):
                func_name = match.group(1)

                # 检查是否为MATLAB内置函数或关键字
                if self._is_matlab_builtin_or_keyword(func_name):
                    continue

                if func_name in self.function_by_name:
                    func_node = self.function_by_name[func_name]

                    # 检查是否为跨作用域调用
                    if self._is_cross_scope_call(node, func_node):
                        relationship = Neo4jRelationship(
                            start_node_id=node.id,
                            end_node_id=func_node.id,
                            type='CALLS',
                            properties={
                                'call_type': 'cross_scope_function',
                                'source_script': self._get_script_name_from_id(node.id),
                                'target_script': self._get_script_name_from_id(func_node.id),
                                'execution_order_based': True,
                                'post_processed': True
                            }
                        )

                        graph.relationships.append(relationship)
                        self.cross_scope_relationships.append(relationship)

        # 新增：处理函数通过三种方式访问外部变量的情况
        self._process_function_external_variable_access(graph)

    def _process_function_external_variable_access(self, graph: Neo4jGraph):
        """处理函数通过参数、全局变量、evalin/assignin访问外部变量的情况"""
        logger.info("Processing function external variable access")

        for node in graph.nodes:
            if node.label != 'Function':
                continue

            content = node.properties.get('code_snippet', '')
            if not content:
                continue

            # 1. 处理函数参数中的变量使用
            self._process_function_parameter_variables(graph, node, content)

            # 2. 处理全局变量声明和使用
            self._process_global_variables(graph, node, content)

            # 3. 处理evalin/assignin函数调用
            self._process_evalin_assignin_calls(graph, node, content)

    def _process_function_parameter_variables(self, graph: Neo4jGraph, func_node: Neo4jNode, content: str):
        """处理函数参数中的变量使用"""
        # 提取函数签名
        func_signature_match = re.search(r'function\s+(?:\[[^\]]*\]\s*=\s*)?([a-zA-Z_]\w*)\s*\(([^)]*)\)', content)
        if not func_signature_match:
            return

        func_name = func_signature_match.group(1)
        params_str = func_signature_match.group(2).strip()

        if not params_str:
            return

        # 解析参数列表
        params = [param.strip() for param in params_str.split(',') if param.strip()]

        # 为每个参数创建USES关系
        for param in params:
            # 清理参数名（去除默认值等）
            param_name = re.match(r'([a-zA-Z_]\w*)', param)
            if param_name:
                param_name = param_name.group(1)

                # 创建参数变量节点
                param_var_id = self._get_or_create_variable_node(graph, param_name, self._get_script_name_from_id(func_node.id))

                if param_var_id:
                    # 创建函数USES参数的关系
                    relationship = Neo4jRelationship(
                        start_node_id=func_node.id,
                        end_node_id=param_var_id,
                        type='USES',
                        properties={
                            'usage_type': 'function_parameter',
                            'parameter_name': param_name,
                            'function_name': func_name,
                            'access_method': 'parameter_passing',
                            'execution_order_based': True,
                            'post_processed': True
                        }
                    )

                    graph.relationships.append(relationship)
                    self.cross_scope_relationships.append(relationship)
                    logger.info(f"Added function parameter relationship: {func_name} -> {param_name}")

    def _process_global_variables(self, graph: Neo4jGraph, func_node: Neo4jNode, content: str):
        """处理全局变量声明和使用"""
        # 查找global声明
        global_pattern = re.compile(r'global\s+([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*)*)', re.MULTILINE)

        for match in global_pattern.finditer(content):
            global_vars_str = match.group(1)
            global_vars = [var.strip() for var in global_vars_str.split() if var.strip()]

            for global_var in global_vars:
                # 创建全局变量节点
                global_var_id = self._get_or_create_variable_node(graph, global_var, "global_scope")

                if global_var_id:
                    # 创建函数USES全局变量的关系
                    relationship = Neo4jRelationship(
                        start_node_id=func_node.id,
                        end_node_id=global_var_id,
                        type='USES',
                        properties={
                            'usage_type': 'global_variable',
                            'variable_name': global_var,
                            'function_name': self._get_function_name_from_node(func_node),
                            'access_method': 'global_declaration',
                            'execution_order_based': True,
                            'post_processed': True
                        }
                    )

                    graph.relationships.append(relationship)
                    self.cross_scope_relationships.append(relationship)
                    logger.info(f"Added global variable relationship: {self._get_function_name_from_node(func_node)} -> {global_var}")

        # 查找全局变量的使用（在global声明之后）
        lines = content.split('\n')
        global_vars_declared = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 检测global声明
            global_match = re.match(r'global\s+([a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*)*)', line)
            if global_match:
                vars_str = global_match.group(1)
                global_vars_declared.update(var.strip() for var in vars_str.split() if var.strip())
                continue

            # 在global声明之后，查找这些变量的使用
            if global_vars_declared:
                for global_var in global_vars_declared:
                    if re.search(r'\b' + re.escape(global_var) + r'\b', line):
                        # 创建USES关系（如果还没有创建）
                        self._create_global_variable_uses_relationship(graph, func_node, global_var, line_num)

    def _process_evalin_assignin_calls(self, graph: Neo4jGraph, func_node: Neo4jNode, content: str):
        """处理evalin和assignin函数调用"""
        # evalin模式：evalin('base', 'variable_name') 或 evalin('caller', 'variable_name')
        evalin_pattern = re.compile(r'evalin\s*\(\s*[\'"](base|caller)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)')

        for match in evalin_pattern.finditer(content):
            workspace = match.group(1)  # 'base' 或 'caller'
            expression = match.group(2)

            # 尝试从表达式中提取变量名
            var_names = self._extract_variable_names_from_expression(expression)

            for var_name in var_names:
                # 创建USES关系
                self._create_evalin_variable_uses_relationship(graph, func_node, var_name, workspace, 'evalin')

        # assignin模式：assignin('base', 'variable_name', value) 或 assignin('caller', 'variable_name', value)
        assignin_pattern = re.compile(r'assignin\s*\(\s*[\'"](base|caller)[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]\s*,\s*([^)]+)\s*\)')

        for match in assignin_pattern.finditer(content):
            workspace = match.group(1)  # 'base' 或 'caller'
            var_name = match.group(2)
            value_expr = match.group(3)

            # 创建USES关系
            self._create_evalin_variable_uses_relationship(graph, func_node, var_name, workspace, 'assignin')

            # 同时处理value_expr中可能包含的变量
            value_var_names = self._extract_variable_names_from_expression(value_expr)
            for value_var in value_var_names:
                self._create_evalin_variable_uses_relationship(graph, func_node, value_var, workspace, 'assignin_value')

    def _extract_variable_names_from_expression(self, expression: str) -> List[str]:
        """从表达式中提取变量名"""
        # 简单的变量名提取模式
        var_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b')
        variables = []

        for match in var_pattern.finditer(expression):
            var_name = match.group(1)
            # 过滤掉MATLAB关键字和内置函数
            if not self._is_matlab_builtin_or_keyword(var_name) and len(var_name) >= 2:
                variables.append(var_name)

        return variables

    def _create_global_variable_uses_relationship(self, graph: Neo4jGraph, func_node: Neo4jNode,
                                                var_name: str, line_num: int):
        """创建全局变量USES关系"""
        global_var_id = self._get_or_create_variable_node(graph, var_name, "global_scope")

        if global_var_id:
            relationship = Neo4jRelationship(
                start_node_id=func_node.id,
                end_node_id=global_var_id,
                type='USES',
                properties={
                    'usage_type': 'global_variable_usage',
                    'variable_name': var_name,
                    'function_name': self._get_function_name_from_node(func_node),
                    'access_method': 'global_usage',
                    'line_number': line_num,
                    'execution_order_based': True,
                    'post_processed': True
                }
            )

            graph.relationships.append(relationship)
            self.cross_scope_relationships.append(relationship)

    def _create_evalin_variable_uses_relationship(self, graph: Neo4jGraph, func_node: Neo4jNode,
                                                var_name: str, workspace: str, access_type: str):
        """创建evalin/assignin变量USES关系"""
        # 根据workspace确定变量来源
        if workspace == 'base':
            source_scope = "base_workspace"
        elif workspace == 'caller':
            source_scope = "caller_workspace"
        else:
            source_scope = "unknown_workspace"

        var_id = self._get_or_create_variable_node(graph, var_name, source_scope)

        if var_id:
            relationship = Neo4jRelationship(
                start_node_id=func_node.id,
                end_node_id=var_id,
                type='USES',
                properties={
                    'usage_type': 'workspace_variable',
                    'variable_name': var_name,
                    'function_name': self._get_function_name_from_node(func_node),
                    'access_method': access_type,
                    'workspace': workspace,
                    'source_scope': source_scope,
                    'execution_order_based': True,
                    'post_processed': True
                }
            )

            graph.relationships.append(relationship)
            self.cross_scope_relationships.append(relationship)
            logger.info(f"Added {access_type} relationship: {self._get_function_name_from_node(func_node)} -> {var_name} ({workspace})")

    def _get_function_name_from_node(self, func_node: Neo4jNode) -> str:
        """从函数节点中提取函数名"""
        return func_node.properties.get('name', 'unknown_function')

    def _is_matlab_builtin_or_keyword(self, func_name: str) -> bool:
        """检查是否为MATLAB内置函数或关键字"""
        builtin_funcs = {
            'disp', 'fprintf', 'sprintf', 'num2str', 'str2num', 'length', 'size',
            'sum', 'mean', 'max', 'min', 'abs', 'sqrt', 'exp', 'log', 'sin', 'cos',
            'plot', 'figure', 'subplot', 'title', 'xlabel', 'ylabel', 'legend',
            'hold', 'grid', 'axis', 'close', 'clf', 'clc', 'clear', 'whos',
            'exist', 'isempty', 'isnumeric', 'ischar', 'iscell', 'isstruct',
            'cell', 'struct', 'zeros', 'ones', 'eye', 'rand', 'randn', 'linspace',
            'logspace', 'reshape', 'transpose', 'ctranspose', 'inv', 'pinv',
            'eig', 'svd', 'qr', 'lu', 'chol', 'det', 'trace', 'rank', 'cond',
            'norm', 'dot', 'cross', 'kron', 'conv', 'deconv', 'fft', 'ifft',
            'filter', 'filtfilt', 'butter', 'cheby1', 'cheby2', 'ellip',
            'bode', 'nyquist', 'nichols', 'rlocus', 'step', 'impulse', 'lsim'
        }

        keywords = {
            'if', 'for', 'while', 'function', 'end', 'else', 'elseif',
            'return', 'break', 'continue', 'switch', 'case', 'otherwise',
            'try', 'catch', 'classdef', 'properties', 'methods', 'global',
            'persistent', 'nargin', 'nargout', 'varargin', 'varargout'
        }

        return func_name.lower() in builtin_funcs or func_name.lower() in keywords

    def _is_cross_scope_call(self, caller_node: Neo4jNode, callee_node: Neo4jNode) -> bool:
        """检查是否为跨作用域调用"""
        caller_script = self._get_script_name_from_id(caller_node.id)
        callee_script = self._get_script_name_from_id(callee_node.id)

        return caller_script != callee_script

    def _process_cross_scope_script_calls(self, graph: Neo4jGraph):
        """处理跨作用域脚本调用"""
        logger.info("Processing cross-scope script calls")

        # 预编译正则表达式
        script_call_patterns = [
            re.compile(r'^\s*([a-zA-Z_]\w*)\s*;', re.MULTILINE),
            re.compile(r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        ]

        for node in graph.nodes:
            if node.label not in ['Script', 'Function']:
                continue

            content = node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找脚本调用
            for pattern in script_call_patterns:
                for match in pattern.finditer(content):
                    called_script = match.group(1).replace('.m', '')

                    # 检查是否为MATLAB内置函数或关键字
                    if self._is_matlab_builtin_or_keyword(called_script):
                        continue

                    if called_script in self.script_by_name:
                        callee_node = self.script_by_name[called_script]

                        # 检查是否为跨作用域调用
                        if self._is_cross_scope_call(node, callee_node):
                            relationship = Neo4jRelationship(
                                start_node_id=node.id,
                                end_node_id=callee_node.id,
                                type='CALLS',
                                properties={
                                    'call_type': 'cross_scope_script',
                                    'source_script': self._get_script_name_from_id(node.id),
                                    'target_script': called_script,
                                    'execution_order_based': True,
                                    'post_processed': True
                                }
                            )

                            graph.relationships.append(relationship)
                            self.cross_scope_relationships.append(relationship)

    def _get_script_name_from_id(self, node_id: str) -> str:
        """从节点ID中提取脚本名称"""
        if node_id.startswith('script_'):
            parts = node_id.split('_', 2)
            if len(parts) >= 3:
                return parts[1]
        elif node_id.startswith('function_'):
            # 从函数节点ID中提取文件路径，然后提取脚本名
            parts = node_id.split('_', 2)
            if len(parts) >= 3:
                file_path = parts[2]
                return Path(file_path).stem
        return "unknown"

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'entry_script': self.entry_script_path,
            'execution_order': self.script_execution_order,
            'total_scripts': len(self.script_execution_order),
            'variable_definitions': dict(self.variable_definitions),
            'cross_scope_relationships': len(self.cross_scope_relationships)
        }

    def validate_variable_scope_binding(self, graph: Neo4jGraph) -> Dict[str, Any]:
        """验证变量节点的作用域绑定是否正确"""
        logger.info("Validating variable scope binding...")

        validation_results = {
            'total_variables': 0,
            'properly_bound': 0,
            'unbound_variables': [],
            'duplicate_variables': [],
            'scope_issues': []
        }

        # 按变量名分组
        variables_by_name = {}
        for node in graph.nodes:
            if node.label == 'Variable':
                validation_results['total_variables'] += 1
                var_name = node.properties.get('name', '')
                scope_id = node.properties.get('scope_id', '')
                scope_type = node.properties.get('scope_type', '')
                scope_binding = node.properties.get('scope_binding', '')

                # 检查作用域绑定
                if not scope_id or not scope_type:
                    validation_results['unbound_variables'].append({
                        'node_id': node.id,
                        'var_name': var_name,
                        'issue': 'Missing scope_id or scope_type'
                    })
                elif scope_binding != f"{scope_type}:{scope_id}":
                    validation_results['scope_issues'].append({
                        'node_id': node.id,
                        'var_name': var_name,
                        'expected_binding': f"{scope_type}:{scope_id}",
                        'actual_binding': scope_binding
                    })
                else:
                    validation_results['properly_bound'] += 1

                # 检查重复变量
                if var_name not in variables_by_name:
                    variables_by_name[var_name] = []
                variables_by_name[var_name].append(node)

        # 检查同名变量是否在不同作用域中
        for var_name, var_nodes in variables_by_name.items():
            if len(var_nodes) > 1:
                # 检查是否有重复的作用域
                scope_combinations = set()
                for var_node in var_nodes:
                    scope_id = var_node.properties.get('scope_id', '')
                    scope_type = var_node.properties.get('scope_type', '')
                    scope_combinations.add(f"{scope_type}:{scope_id}")

                if len(scope_combinations) < len(var_nodes):
                    validation_results['duplicate_variables'].append({
                        'var_name': var_name,
                        'nodes': [node.id for node in var_nodes],
                        'scopes': list(scope_combinations)
                    })

        # 输出验证结果
        logger.info(f"Variable scope binding validation results:")
        logger.info(f"  Total variables: {validation_results['total_variables']}")
        logger.info(f"  Properly bound: {validation_results['properly_bound']}")
        logger.info(f"  Unbound variables: {len(validation_results['unbound_variables'])}")
        logger.info(f"  Scope issues: {len(validation_results['scope_issues'])}")
        logger.info(f"  Duplicate variables: {len(validation_results['duplicate_variables'])}")

        if validation_results['unbound_variables']:
            logger.warning("Found unbound variables:")
            for issue in validation_results['unbound_variables'][:5]:  # 只显示前5个
                logger.warning(f"  {issue}")

        if validation_results['duplicate_variables']:
            logger.warning("Found duplicate variables:")
            for issue in validation_results['duplicate_variables'][:5]:  # 只显示前5个
                logger.warning(f"  {issue}")

        return validation_results

    def ensure_structural_binding(self, graph: Neo4jGraph) -> Dict[str, Any]:
        """确保所有变量和函数节点都与其作用域节点有结构性绑定"""
        logger.info("Ensuring structural binding for all nodes...")

        binding_results = {
            'variables_fixed': 0,
            'functions_fixed': 0,
            'variables_already_bound': 0,
            'functions_already_bound': 0,
            'orphaned_variables': [],
            'orphaned_functions': []
        }

        # 构建现有的DEFINES关系索引
        defines_relationships = {}
        for rel in graph.relationships:
            if rel.type == 'DEFINES':
                defines_relationships[rel.end_node_id] = rel.start_node_id

        # 1. 修复变量节点的结构性绑定
        for node in graph.nodes:
            if node.label == 'Variable':
                # 检查是否已有DEFINES关系
                if node.id in defines_relationships:
                    binding_results['variables_already_bound'] += 1
                    continue

                # 获取变量的作用域信息
                scope_id = node.properties.get('scope_id', '')
                scope_type = node.properties.get('scope_type', '')

                if not scope_id or not scope_type:
                    binding_results['orphaned_variables'].append({
                        'node_id': node.id,
                        'var_name': node.properties.get('name', ''),
                        'issue': 'Missing scope information'
                    })
                    continue

                # 查找对应的作用域节点
                scope_node = None
                if scope_type == 'script':
                    # 查找脚本节点
                    for script_node in graph.nodes:
                        if (script_node.label == 'Script' and
                            script_node.properties.get('name', '') == scope_id):
                            scope_node = script_node
                            break
                elif scope_type == 'function':
                    # 查找函数节点
                    for func_node in graph.nodes:
                        if (func_node.label == 'Function' and
                            func_node.properties.get('name', '') == scope_id):
                            scope_node = func_node
                            break

                if scope_node:
                    # 创建DEFINES关系
                    relationship = Neo4jRelationship(
                        start_node_id=scope_node.id,
                        end_node_id=node.id,
                        type='DEFINES',
                        properties={
                            'definition_type': 'structural_binding',
                            'scope_type': scope_type,
                            'scope_id': scope_id,
                            'auto_fixed': True,
                            'post_processed': True
                        }
                    )
                    graph.relationships.append(relationship)
                    binding_results['variables_fixed'] += 1
                    logger.debug(f"Fixed variable binding: {scope_node.id} -[DEFINES]-> {node.id}")
                else:
                    binding_results['orphaned_variables'].append({
                        'node_id': node.id,
                        'var_name': node.properties.get('name', ''),
                        'issue': f'Cannot find scope node for {scope_type}:{scope_id}'
                    })

        # 2. 修复函数节点的结构性绑定
        for node in graph.nodes:
            if node.label == 'Function':
                # 检查是否已有DEFINES关系
                if node.id in defines_relationships:
                    binding_results['functions_already_bound'] += 1
                    continue

                # 获取函数的文件路径
                file_path = node.properties.get('file_path', '')
                if not file_path:
                    binding_results['orphaned_functions'].append({
                        'node_id': node.id,
                        'func_name': node.properties.get('name', ''),
                        'issue': 'Missing file_path'
                    })
                    continue

                # 查找对应的脚本节点
                script_node = None
                for script_node_candidate in graph.nodes:
                    if (script_node_candidate.label == 'Script' and
                        script_node_candidate.properties.get('file_path', '') == file_path):
                        script_node = script_node_candidate
                        break

                if script_node:
                    # 创建DEFINES关系
                    relationship = Neo4jRelationship(
                        start_node_id=script_node.id,
                        end_node_id=node.id,
                        type='DEFINES',
                        properties={
                            'definition_type': 'function_in_script',
                            'file_path': file_path,
                            'auto_fixed': True,
                            'post_processed': True
                        }
                    )
                    graph.relationships.append(relationship)
                    binding_results['functions_fixed'] += 1
                    logger.debug(f"Fixed function binding: {script_node.id} -[DEFINES]-> {node.id}")
                else:
                    binding_results['orphaned_functions'].append({
                        'node_id': node.id,
                        'func_name': node.properties.get('name', ''),
                        'issue': f'Cannot find script node for file: {file_path}'
                    })

        # 输出修复结果
        logger.info(f"Structural binding results:")
        logger.info(f"  Variables already bound: {binding_results['variables_already_bound']}")
        logger.info(f"  Variables fixed: {binding_results['variables_fixed']}")
        logger.info(f"  Functions already bound: {binding_results['functions_already_bound']}")
        logger.info(f"  Functions fixed: {binding_results['functions_fixed']}")
        logger.info(f"  Orphaned variables: {len(binding_results['orphaned_variables'])}")
        logger.info(f"  Orphaned functions: {len(binding_results['orphaned_functions'])}")

        if binding_results['orphaned_variables']:
            logger.warning("Found orphaned variables:")
            for issue in binding_results['orphaned_variables'][:3]:  # 只显示前3个
                logger.warning(f"  {issue}")

        if binding_results['orphaned_functions']:
            logger.warning("Found orphaned functions:")
            for issue in binding_results['orphaned_functions'][:3]:  # 只显示前3个
                logger.warning(f"  {issue}")

        return binding_results
