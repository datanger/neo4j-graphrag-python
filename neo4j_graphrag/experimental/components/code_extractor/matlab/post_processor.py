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
from .utils import get_matlab_builtin_functions
from typing import Dict, List, Optional, Any, Tuple, Set

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from .registry import get_global_registry

logger = logging.getLogger(__name__)


class MatlabPostProcessor:
    """MATLAB代码后处理器，处理跨文件关系和变量作用域"""

    def __init__(self):
        self.registry = get_global_registry()
        self.function_nodes = {}  # function_name -> node_id
        self.script_nodes = {}    # script_name -> node_id
        self.variable_nodes = {}  # variable_name -> [node_ids]

    def post_process_cross_file_relationships(self, graph: Neo4jGraph, progress: Optional[Any] = None) -> Neo4jGraph:
        """后处理跨文件关系 - 简化版本，专注于跨作用域关系"""
        logger.info("Starting simplified post-processing of cross-file relationships")

        # 构建高效的查找索引
        self._build_lookup_indexes(graph)

        # 1. 处理跨作用域变量USES关系（基于启动脚本执行顺序）
        self._process_cross_scope_variable_uses(graph)
        if progress: progress.update(1)

        # 2. 处理变量间的ASSIGNED_TO关系
        self._process_variable_assignments(graph)
        if progress: progress.update(1)

        # 3. 处理跨作用域函数调用USES关系
        self._process_cross_scope_function_calls(graph)
        if progress: progress.update(1)

        # 4. 处理跨作用域脚本调用USES关系
        self._process_cross_scope_script_calls(graph)
        if progress: progress.update(1)

        logger.info("Completed simplified post-processing")
        return graph

    def _build_lookup_indexes(self, graph: Neo4jGraph):
        """构建高效的查找索引"""
        # 变量名到节点的映射
        self.var_name_to_nodes = {}
        # 函数名到节点的映射
        self.func_name_to_nodes = {}
        # 脚本名到节点的映射
        self.script_name_to_nodes = {}
        # 节点ID到节点的映射
        self.node_by_id = {}

        for node in graph.nodes:
            self.node_by_id[node.id] = node

            if node.label == 'Variable':
                var_name = node.properties.get('name', '')
                if var_name and len(var_name) >= 2:
                    if var_name not in self.var_name_to_nodes:
                        self.var_name_to_nodes[var_name] = []
                    self.var_name_to_nodes[var_name].append(node)

            elif node.label == 'Function':
                func_name = node.properties.get('name', '')
                if func_name:
                    self.func_name_to_nodes[func_name] = node

            elif node.label == 'Script':
                script_name = node.properties.get('name', '')
                if script_name:
                    self.script_name_to_nodes[script_name] = node

        # 已存在的关系集合（用于去重）
        self.existing_relationships = set()
        for rel in graph.relationships:
            self.existing_relationships.add((rel.start_node_id, rel.end_node_id, rel.type))

    def _process_cross_scope_variable_uses(self, graph: Neo4jGraph):
        """处理跨作用域变量USES关系 - 基于启动脚本执行顺序"""
        if not self.registry.entry_script_execution_flow:
            return

        # 获取执行顺序
        execution_order = self.registry.get_entry_script_execution_order()
        if not execution_order:
            return

        # 为每个脚本处理变量使用
        for script_name in execution_order:
            script_node = self.script_name_to_nodes.get(script_name)
            if not script_node:
                continue

            content = script_node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找脚本中使用的所有变量
            used_vars = self._extract_variables_from_content(content)

            # 限制每个脚本最多处理10个跨作用域变量使用
            processed_count = 0
            for var_name in used_vars:
                if processed_count >= 10:
                    break

                # 查找变量来源（基于执行顺序）
                source_script = self._find_variable_source(var_name, script_name, execution_order)
                if source_script and source_script != script_name:
                    # 创建跨作用域USES关系
                    self._add_cross_scope_uses_relationship(
                        graph, script_node.id, var_name, source_script, 'variable'
                    )
                    processed_count += 1

    def _process_variable_assignments(self, graph: Neo4jGraph):
        """处理跨作用域变量间的ASSIGNED_TO关系"""
        # 预编译正则表达式
        assignment_pattern = re.compile(r'\s*([a-zA-Z_]\w*)\s*=\s*([^;\n]+);?')
        variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b')

        # 只处理有意义的变量名（长度>=3，避免短变量名）
        meaningful_vars = {name: nodes for name, nodes in self.var_name_to_nodes.items()
                          if len(name) >= 3}

        # 收集跨作用域变量使用信息
        cross_scope_uses = self._collect_cross_scope_variable_uses()

        # 用于跟踪已创建的关系，避免重复
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
                target_var = match.group(1)  # 左侧变量
                rhs_expr = match.group(2).strip()  # 右侧表达式

                # 只处理有意义的变量
                if target_var not in meaningful_vars:
                    continue

                # 提取右侧的变量
                rhs_vars = variable_pattern.findall(rhs_expr)

                # 只处理有意义的右侧变量，且避免自赋值
                meaningful_rhs_vars = [var for var in rhs_vars
                                     if var in meaningful_vars and var != target_var]

                # 为跨作用域使用的变量创建ASSIGNED_TO关系
                for rhs_var in meaningful_rhs_vars:
                    # 检查这个变量是否被跨作用域使用
                    if self._is_cross_scope_variable(rhs_var, node, cross_scope_uses):
                        # 为每个变量对只创建一次关系
                        assignment_key = (target_var, rhs_var)
                        if assignment_key not in created_assignments:
                            # 选择第一个目标变量节点和第一个源变量节点
                            target_node = meaningful_vars[target_var][0]
                            source_node = meaningful_vars[rhs_var][0]

                            # 确保是跨作用域的关系
                            if not self._same_scope(source_node.id, target_node.id):
                                self._add_relationship(
                                    graph, source_node.id, target_node.id, 'ASSIGNED_TO',
                                    {
                                        'assignment_type': 'cross_scope_assignment',
                                        'source_script': self._get_script_from_node_id(source_node.id),
                                        'target_script': self._get_script_from_node_id(target_node.id),
                                        'post_processed': True
                                    }
                                )
                                created_assignments.add(assignment_key)

    def _get_script_from_node_id(self, node_id: str) -> str:
        """从节点ID中提取脚本名称"""
        # 解析节点ID格式：var_{var_name}_{scope_type}_{parent_id}
        parts = node_id.split('_')
        if len(parts) >= 4:
            # 提取parent_id部分
            parent_id = '_'.join(parts[3:])
            # 如果是脚本节点，直接返回
            if parent_id.startswith('script_'):
                return parent_id.replace('script_', '')
            # 如果是函数节点，返回文件路径
            elif parent_id.startswith('function_'):
                # 从函数节点ID中提取文件路径
                func_parts = parent_id.split('_')
                if len(func_parts) >= 2:
                    return func_parts[1]  # 返回函数名作为作用域标识
        return "unknown"

    def _collect_cross_scope_variable_uses(self) -> Dict[str, Set[str]]:
        """收集跨作用域变量使用信息"""
        cross_scope_uses = {}

        if not self.registry.entry_script_execution_flow:
            return cross_scope_uses

        # 获取执行顺序
        execution_order = self.registry.get_entry_script_execution_order()
        if not execution_order:
            return cross_scope_uses

        # 为每个脚本收集跨作用域变量使用
        for script_name in execution_order:
            script_node = self.script_name_to_nodes.get(script_name)
            if not script_node:
                continue

            content = script_node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找脚本中使用的所有变量
            used_vars = self._extract_variables_from_content(content)

            for var_name in used_vars:
                # 查找变量来源（基于执行顺序）
                source_script = self._find_variable_source(var_name, script_name, execution_order)
                if source_script and source_script != script_name:
                    # 这是一个跨作用域变量使用
                    if var_name not in cross_scope_uses:
                        cross_scope_uses[var_name] = set()
                    cross_scope_uses[var_name].add(source_script)

        return cross_scope_uses

    def _is_cross_scope_variable(self, var_name: str, current_node, cross_scope_uses: Dict[str, Set[str]]) -> bool:
        """检查变量是否被跨作用域使用"""
        # 如果变量在跨作用域使用列表中，则返回True
        if var_name in cross_scope_uses:
            return True

        # 检查变量是否在入口脚本中定义
        if var_name in self.registry.entry_script_variable_scope:
            return True

        return False

    def _same_scope(self, node_id1: str, node_id2: str) -> bool:
        """检查两个节点是否在同一作用域"""
        node1 = self.node_by_id.get(node_id1)
        node2 = self.node_by_id.get(node_id2)

        if not node1 or not node2:
            return False

        # 获取节点的作用域信息
        scope1 = node1.properties.get('scope_id', '')
        scope2 = node2.properties.get('scope_id', '')

        # 如果作用域相同，则在同一作用域
        return scope1 == scope2 and scope1 != ''

    def _process_cross_scope_function_calls(self, graph: Neo4jGraph):
        """处理跨作用域函数调用USES关系"""
        function_call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
        builtin_funcs = get_matlab_builtin_functions()
        keywords = {'if', 'for', 'while', 'function', 'end', 'else', 'elseif',
                           'return', 'break', 'continue', 'switch', 'case', 'otherwise',
                   'try', 'catch', 'classdef', 'properties', 'methods'}

        for node in graph.nodes:
            if node.label not in ['Script', 'Function']:
                continue

            content = node.properties.get('code_snippet', '')
            if not content:
                continue

            # 查找函数调用
            for match in function_call_pattern.finditer(content):
                func_name = match.group(1)
                # 跳过内置函数（若工程中没有同名自定义实现）
                if func_name.lower() in builtin_funcs and func_name not in self.func_name_to_nodes:
                    continue

                # 跳过关键字
                if func_name in keywords:
                    continue

                # 检查是否是跨作用域调用
                if func_name in self.func_name_to_nodes:
                    func_node = self.func_name_to_nodes[func_name]

                    # 若调用者与被调用函数在同一文件，认为是本地函数，跳过跨作用域处理
                    if func_node.properties.get('file_path') == node.properties.get('file_path'):
                        continue

                    # 如果是函数调用函数，且不是自调用
                    if (node.label == 'Function' and
                        func_name != node.properties.get('name')):
                        self._add_relationship(
                            graph, node.id, func_node.id, 'CALLS',
                            {
                                'call_type': 'function_call',
                                'scope': 'cross_scope',
                                'post_processed': True
                            }
                        )

                    # 如果是脚本调用函数
                    elif node.label == 'Script':
                        self._add_relationship(
                            graph, node.id, func_node.id, 'USES',
                            {
                                'call_type': 'function_call',
                                'scope': 'cross_scope',
                                'post_processed': True
                            }
                        )

    def _process_cross_scope_script_calls(self, graph: Neo4jGraph):
        """处理跨作用域脚本调用USES关系"""
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

                    if called_script in self.script_name_to_nodes:
                        script_node = self.script_name_to_nodes[called_script]

                        # 创建跨作用域USES关系
                        self._add_relationship(
                            graph, node.id, script_node.id, 'USES',
                            {
                                'call_type': 'script_call',
                                'scope': 'cross_scope',
                                'post_processed': True
                            }
                        )

    def _extract_variables_from_content(self, content: str) -> Set[str]:
        """从代码内容中提取变量名 - 正确过滤注释和字符串"""
        keywords = {'if', 'for', 'while', 'function', 'end', 'else', 'elseif',
                   'return', 'break', 'continue', 'switch', 'case', 'otherwise',
                   'try', 'catch', 'classdef', 'properties', 'methods', 'true', 'false',
                   'pi', 'inf', 'nan', 'eps', 'realmax', 'realmin', 'i', 'j'}

        variables = set()
        lines = content.split('\n')

        for line in lines:
            # 跳过注释行
            stripped_line = line.strip()
            if stripped_line.startswith('%') or stripped_line.startswith('//'):
                continue

            # 处理行内注释
            comment_pos = stripped_line.find('%')
            if comment_pos != -1:
                stripped_line = stripped_line[:comment_pos]

            # 跳过空行
            if not stripped_line:
                continue

            # 提取变量名，但排除字符串中的内容
            variables_in_line = self._extract_variables_from_line(stripped_line)

            for var_name in variables_in_line:
                # 只提取有意义的变量名（长度>=3，不是关键字，且在变量映射中存在）
                if (var_name not in keywords and
                    len(var_name) >= 3 and
                    var_name in self.var_name_to_nodes):
                    variables.add(var_name)

        return variables

    def _extract_variables_from_line(self, line: str) -> Set[str]:
        """从单行代码中提取变量名，排除字符串中的内容"""
        variables = set()

        # 移除字符串内容
        line_without_strings = self._remove_strings(line)

        # 使用正则表达式提取变量名
        variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b')
        for match in variable_pattern.finditer(line_without_strings):
            var_name = match.group(1)
            variables.add(var_name)

        return variables

    def _remove_strings(self, line: str) -> str:
        """移除行中的字符串内容"""
        result = ""
        i = 0
        in_single_quote = False
        in_double_quote = False

        while i < len(line):
            char = line[i]

            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                result += " "  # 用空格替换字符串内容
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                result += " "  # 用空格替换字符串内容
            elif not in_single_quote and not in_double_quote:
                result += char

            i += 1

        return result

    def _find_variable_source(self, var_name: str, usage_script: str, execution_order: List[str]) -> Optional[str]:
        """基于执行顺序查找变量来源"""
        # 检查变量是否在入口脚本中定义
        if var_name in self.registry.entry_script_variable_scope:
            return "entry_script"

        # 根据执行顺序查找
        try:
            usage_index = execution_order.index(usage_script)
        except ValueError:
            return None

        # 在usage_script之前执行的脚本中查找变量定义
        for i in range(usage_index):
            candidate_script = execution_order[i]
            if (candidate_script in self.registry.variable_definitions_by_script and
                var_name in self.registry.variable_definitions_by_script[candidate_script]):
                return candidate_script

        return None

    def _add_cross_scope_uses_relationship(self, graph: Neo4jGraph, caller_id: str, var_name: str,
                                         source_script: str, target_type: str):
        """添加跨作用域USES关系"""
        if source_script == "entry_script":
            target_id = f"var_{var_name}_script_entry_script"
        else:
            target_id = f"var_{var_name}_script_{source_script}"

        # 检查目标节点是否存在
        if target_id not in self.node_by_id:
            # 创建缺失的变量节点
            var_node = Neo4jNode(
                id=target_id,
                label='Variable',
                properties={
                    'name': var_name,
                    'type': 'local',
                    'scope_id': f'{source_script}' if source_script != "entry_script" else 'entry_script',
                    'scope_type': 'script',
                    'file_path': f'{source_script}.m' if source_script != "entry_script" else 'entry_script.m'
                }
            )
            graph.nodes.append(var_node)
            self.node_by_id[target_id] = var_node

        # 添加USES关系
        self._add_relationship(
            graph, caller_id, target_id, 'USES',
            {
                'usage_type': 'cross_scope_execution_order',
                'variable_name': var_name,
                'source_script': source_script,
                'target_script': self._get_script_from_node_id(caller_id),
                'execution_order': self._get_execution_order(source_script),
                'source_execution_order': self._get_execution_order(source_script),
                'post_processed': True
            }
        )

    def _get_execution_order(self, script_name: str) -> int:
        """获取脚本的执行顺序"""
        if not self.registry.entry_script_execution_flow:
            return 0

        execution_order = self.registry.get_entry_script_execution_order()
        if not execution_order:
            return 0

        try:
            return execution_order.index(script_name)
        except ValueError:
            return 0

    def _add_relationship(self, graph: Neo4jGraph, start_id: str, end_id: str,
                         rel_type: str, properties: dict = None):
        """添加关系（带去重检查）"""
        rel_key = (start_id, end_id, rel_type)
        if rel_key not in self.existing_relationships:
            rel = Neo4jRelationship(
                start_node_id=start_id,
                end_node_id=end_id,
                type=rel_type,
                properties=properties or {}
            )
            graph.relationships.append(rel)
            self.existing_relationships.add(rel_key)
