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
from typing import Dict, List, Optional, Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship
from .registry import get_global_registry

logger = logging.getLogger(__name__)


class MatlabPostProcessor:
    """MATLAB代码后处理器，处理跨文件关系和变量作用域"""

    def __init__(self):
        self.registry = get_global_registry()

    def post_process_cross_file_relationships(self, graph: Neo4jGraph) -> Neo4jGraph:
        """后处理跨文件关系"""
        logger.info("Starting post-processing of cross-file relationships")
        
        # 处理启动脚本变量访问
        self._process_entry_script_variable_access(graph)
        
        # 处理执行顺序变量访问
        if self.registry.execution_order:
            self._process_execution_order_variable_access(graph, self.registry.execution_order)
        
        # 查找脚本调用
        self._find_script_calls_in_content(graph)
        
        # 查找函数调用
        self._find_function_calls_in_content(graph)
        
        logger.info("Completed post-processing of cross-file relationships")
        return graph

    def _process_entry_script_variable_access(self, graph: Neo4jGraph):
        """处理启动脚本中的变量访问"""
        if not self.registry.entry_script_execution_flow:
            return
        
        # 创建启动脚本节点
        entry_script_node = {
            'id': 'entry_script',
            'labels': ['Script'],
            'properties': {
                'name': 'entry_script',
                'file_path': self.registry.entry_script_path,
                'type': 'entry_script'
            }
        }
        
        # 添加启动脚本节点
        if not any(node['id'] == 'entry_script' for node in graph.nodes):
            graph.nodes.append(entry_script_node)
        
        # 处理启动脚本中的变量定义
        for var_name, var_info in self.registry.entry_script_variable_scope.items():
            var_node = {
                'id': f'variable_{var_name}_entry_script',
                'labels': ['Variable'],
                'properties': {
                    'name': var_name,
                    'type': 'entry_script_variable',
                    'defined_at': var_info['defined_at'],
                    'scope': var_info['scope']
                }
            }
            
            if not any(node['id'] == var_node['id'] for node in graph.nodes):
                graph.nodes.append(var_node)
            
            # 添加定义关系
            def_edge = {
                'source': 'entry_script',
                'target': var_node['id'],
                'type': 'DEFINES',
                'properties': {'line_number': var_info['defined_at']}
            }
            graph.relationships.append(def_edge)
        
        # 处理启动脚本调用的脚本对变量的访问
        for call in self.registry.entry_script_execution_flow:
            script_name = call['script_name']
            script_node_id = f"script_{script_name}_{script_name}.m"
            
            # 检查脚本节点是否存在
            if any(node['id'] == script_node_id for node in graph.nodes):
                # 为每个启动脚本变量创建访问关系
                for var_name in self.registry.entry_script_variable_scope.keys():
                    access_edge = {
                        'source': script_node_id,
                        'target': f'variable_{var_name}_entry_script',
                        'type': 'ACCESSES',
                        'properties': {
                            'access_type': 'entry_script_variable',
                            'line_number': call['line_number']
                        }
                    }
                    graph.relationships.append(access_edge)

    def _process_execution_order_variable_access(self, graph: Neo4jGraph, execution_order: List[str]):
        """根据执行顺序处理变量访问"""
        for i, script_name in enumerate(execution_order):
            script_node_id = f"script_{script_name}_{script_name}.m"
            
            # 检查脚本节点是否存在
            if not any(node['id'] == script_node_id for node in graph.nodes):
                continue
            
            # 获取脚本中使用的变量
            if script_name in self.registry.variable_usage_by_script:
                for var_name, usage_lines in self.registry.variable_usage_by_script[script_name].items():
                    # 查找变量来源
                    source_script = self.registry.get_variable_source_script(var_name, script_name)
                    
                    if source_script and source_script != script_name:
                        # 变量来自其他脚本
                        source_script_node_id = f"script_{source_script}_{source_script}.m"
                        var_node_id = f"variable_{var_name}_{source_script}_{source_script}.m"
                        
                        # 检查变量节点是否存在
                        if any(node['id'] == var_node_id for node in graph.nodes):
                            # 创建跨脚本变量访问关系
                            for usage_line in usage_lines:
                                access_edge = {
                                    'source': script_node_id,
                                    'target': var_node_id,
                                    'type': 'ACCESSES',
                                    'properties': {
                                        'access_type': 'cross_script_variable',
                                        'line_number': usage_line,
                                        'source_script': source_script
                                    }
                                }
                                graph.relationships.append(access_edge)

    def _find_script_calls_in_content(self, graph: Neo4jGraph):
        """在内容中查找脚本调用"""
        node_map = {node['id']: node for node in graph.nodes}
        
        for node in graph.nodes:
            if 'Script' in node['labels'] or 'Function' in node['labels']:
                content = node.get('properties', {}).get('code_snippet', '')
                if not content:
                    continue
                
                # 查找脚本调用
                script_call_patterns = [
                    r'^\s*([a-zA-Z_]\w*)\s*;',  # 直接脚本调用
                    r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",  # run()函数调用
                ]
                
                for pattern in script_call_patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        called_script = match.group(1)
                        called_script_node_id = f"script_{called_script}_{called_script}.m"
                        
                        # 检查被调用的脚本是否存在
                        if called_script_node_id in node_map:
                            call_edge = {
                                'source': node['id'],
                                'target': called_script_node_id,
                                'type': 'CALLS_SCRIPT',
                                'properties': {
                                    'call_type': 'direct' if ';' in match.group(0) else 'run_function',
                                    'line_number': self._get_line_number_from_match(match, content)
                                }
                            }
                            graph.relationships.append(call_edge)

    def _find_function_calls_in_content(self, graph: Neo4jGraph):
        """在内容中查找函数调用"""
        node_map = {node['id']: node for node in graph.nodes}
        
        for node in graph.nodes:
            if 'Script' in node['labels'] or 'Function' in node['labels']:
                content = node.get('properties', {}).get('code_snippet', '')
                if not content:
                    continue
                
                # 查找函数调用
                function_call_pattern = r'\b([a-zA-Z_]\w*)\s*\('
                
                for match in re.finditer(function_call_pattern, content):
                    func_name = match.group(1)
                    
                    # 跳过MATLAB内置函数和关键字
                    if func_name in ['if', 'for', 'while', 'function', 'end', 'else', 'elseif', 'return', 'break', 'continue']:
                        continue
                    
                    # 查找函数节点
                    func_node_id = f"function_{func_name}_{func_name}.m"
                    
                    # 检查函数节点是否存在
                    if func_node_id in node_map:
                        call_edge = {
                            'source': node['id'],
                            'target': func_node_id,
                            'type': 'CALLS_FUNCTION',
                            'properties': {
                                'line_number': self._get_line_number_from_match(match, content)
                            }
                        }
                        graph.relationships.append(call_edge)

    def _get_line_number_from_match(self, match, content: str) -> int:
        """根据匹配位置计算行号"""
        return content[:match.start()].count('\n') + 1

    def cleanup_duplicates(self, graph: Neo4jGraph) -> Neo4jGraph:
        """清理重复的节点和关系"""
        # 清理重复节点
        seen_nodes = set()
        unique_nodes = []
        
        for node in graph.nodes:
            node_key = (node['id'], tuple(sorted(node['labels'])))
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                unique_nodes.append(node)
        
        graph.nodes = unique_nodes
        
        # 清理重复关系
        seen_edges = set()
        unique_edges = []
        
        for edge in graph.relationships:
            edge_key = (edge['source'], edge['target'], edge['type'])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)
        
        graph.relationships = unique_edges
        
        logger.info(f"Cleaned up duplicates: {len(unique_nodes)} nodes, {len(unique_edges)} relationships")
        return graph

    def extract_variables_and_relationships(self, graph: Neo4jGraph) -> Neo4jGraph:
        """提取变量和关系"""
        # 这里可以添加更复杂的变量分析逻辑
        # 例如：变量类型推断、变量作用域分析等
        return graph

    def establish_cross_scope_relationships(self, graph: Neo4jGraph) -> Neo4jGraph:
        """建立跨作用域关系"""
        # 这里可以添加跨作用域关系的建立逻辑
        return graph

    def establish_script_to_script_calls(self, graph: Neo4jGraph) -> Neo4jGraph:
        """建立脚本到脚本的调用关系"""
        # 这里可以添加脚本间调用关系的建立逻辑
        return graph