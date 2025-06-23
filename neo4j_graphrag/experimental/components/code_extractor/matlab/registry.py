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
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GlobalMatlabRegistry:
    """Global registry to track all MATLAB entities across files for post-processing."""

    def __init__(self):
        self.scripts = {}  # script_name -> script_node
        self.functions = {}  # func_name -> func_node
        self.all_nodes = []  # All nodes from all files
        self.all_edges = []  # All edges from all files
        self.call_sites = []  # All call sites for post-processing
        self.file_contents = {}  # file_path -> content for cross-file analysis
        # 新增：执行顺序分析相关
        self.execution_order = []  # 脚本执行顺序
        self.variable_definitions_by_script = {}  # script_name -> {var_name: line_number}
        self.variable_usage_by_script = {}  # script_name -> {var_name: line_number}
        self.script_dependencies = {}  # script_name -> [dependent_scripts]
        # 新增：启动脚本支持
        self.entry_script_path = None  # 启动脚本路径
        self.entry_script_content = None  # 启动脚本内容
        self.entry_script_execution_flow = []  # 启动脚本执行流程
        self.entry_script_variable_scope = {}  # 启动脚本中的变量作用域

    def set_entry_script(self, script_path: str, script_content: str = None):
        """设置启动脚本，用于分析执行流程和变量作用域。
        
        Args:
            script_path: 启动脚本的路径
            script_content: 启动脚本的内容，如果为None则从文件读取
        """
        self.entry_script_path = script_path
        if script_content is None:
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    self.entry_script_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read entry script {script_path}: {e}")
                self.entry_script_content = ""
        else:
            self.entry_script_content = script_content
        
        # 分析启动脚本的执行流程
        self._analyze_entry_script_execution_flow()

    def _analyze_entry_script_execution_flow(self):
        """分析启动脚本的执行流程，确定脚本调用顺序和变量作用域。"""
        if not self.entry_script_content:
            return
        
        lines = self.entry_script_content.split('\n')
        current_scope = "entry_script"
        self.entry_script_execution_flow = []
        self.entry_script_variable_scope = {}
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            # 检测脚本调用
            script_call_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*;', line)
            if script_call_match:
                script_name = script_call_match.group(1)
                self.entry_script_execution_flow.append({
                    'type': 'script_call',
                    'script_name': script_name,
                    'line_number': line_num,
                    'scope': current_scope
                })
                continue
            
            # 检测run()调用
            run_call_match = re.search(r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
            if run_call_match:
                script_name = run_call_match.group(1)
                self.entry_script_execution_flow.append({
                    'type': 'run_call',
                    'script_name': script_name,
                    'line_number': line_num,
                    'scope': current_scope
                })
                continue
            
            # 检测变量定义
            var_def_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*=', line)
            if var_def_match:
                var_name = var_def_match.group(1)
                if var_name not in self.entry_script_variable_scope:
                    self.entry_script_variable_scope[var_name] = {
                        'defined_at': line_num,
                        'scope': current_scope,
                        'usage_lines': []
                    }
                continue
            
            # 检测变量使用
            for var_name in self.entry_script_variable_scope.keys():
                if re.search(r'\b' + re.escape(var_name) + r'\b', line):
                    self.entry_script_variable_scope[var_name]['usage_lines'].append(line_num)
        
        logger.info(f"Entry script execution flow: {len(self.entry_script_execution_flow)} calls, {len(self.entry_script_variable_scope)} variables")

    def get_entry_script_execution_order(self) -> List[str]:
        """根据启动脚本获取执行顺序。"""
        if not self.entry_script_execution_flow:
            return []
        
        execution_order = []
        seen_scripts = set()
        
        # 按照启动脚本中的调用顺序构建执行顺序
        for call in self.entry_script_execution_flow:
            script_name = call['script_name']
            if script_name not in seen_scripts:
                execution_order.append(script_name)
                seen_scripts.add(script_name)
        
        return execution_order

    def get_entry_script_variable_source(self, var_name: str, usage_script: str) -> Optional[str]:
        """根据启动脚本确定变量的来源脚本。
        
        Args:
            var_name: 变量名
            usage_script: 使用该变量的脚本名
            
        Returns:
            定义该变量的脚本名，如果无法确定则返回None
        """
        if not self.entry_script_execution_flow:
            return None
        
        # 检查变量是否在启动脚本中定义
        if var_name in self.entry_script_variable_scope:
            return "entry_script"
        
        # 根据执行顺序查找变量来源
        execution_order = self.get_entry_script_execution_order()
        if usage_script not in execution_order:
            return None
        
        usage_index = execution_order.index(usage_script)
        
        # 在usage_script之前执行的脚本中查找变量定义
        for i in range(usage_index):
            candidate_script = execution_order[i]
            if (candidate_script in self.variable_definitions_by_script and 
                var_name in self.variable_definitions_by_script[candidate_script]):
                return candidate_script
        
        return None

    def register_script(self, script_name: str, script_node: dict):
        """注册脚本节点"""
        self.scripts[script_name] = script_node

    def register_function(self, func_name: str, func_node: dict):
        """注册函数节点"""
        self.functions[func_name] = func_node

    def add_nodes(self, nodes: List[dict]):
        """添加节点到全局列表"""
        self.all_nodes.extend(nodes)

    def add_edges(self, edges: List[dict]):
        """添加边到全局列表"""
        self.all_edges.extend(edges)

    def add_call_sites(self, call_sites: List[dict]):
        """添加调用点"""
        self.call_sites.extend(call_sites)

    def register_file_content(self, file_path: str, content: str):
        """注册文件内容"""
        self.file_contents[file_path] = content

    def register_variable_definition(self, script_name: str, var_name: str, line_number: int):
        """注册变量定义"""
        if script_name not in self.variable_definitions_by_script:
            self.variable_definitions_by_script[script_name] = {}
        self.variable_definitions_by_script[script_name][var_name] = line_number

    def register_variable_usage(self, script_name: str, var_name: str, line_number: int):
        """注册变量使用"""
        if script_name not in self.variable_usage_by_script:
            self.variable_usage_by_script[script_name] = {}
        if var_name not in self.variable_usage_by_script[script_name]:
            self.variable_usage_by_script[script_name][var_name] = []
        self.variable_usage_by_script[script_name][var_name].append(line_number)

    def register_script_dependency(self, caller_script: str, callee_script: str):
        """注册脚本依赖关系"""
        if caller_script not in self.script_dependencies:
            self.script_dependencies[caller_script] = []
        if callee_script not in self.script_dependencies[caller_script]:
            self.script_dependencies[caller_script].append(callee_script)

    def analyze_execution_order(self):
        """分析脚本执行顺序"""
        if not self.script_dependencies:
            return
        
        # 使用拓扑排序确定执行顺序
        in_degree = {}
        for script in self.scripts.keys():
            in_degree[script] = 0
        
        for caller, callees in self.script_dependencies.items():
            for callee in callees:
                if callee in in_degree:
                    in_degree[callee] += 1
        
        queue = [script for script, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            if current in self.script_dependencies:
                for callee in self.script_dependencies[current]:
                    if callee in in_degree:
                        in_degree[callee] -= 1
                        if in_degree[callee] == 0:
                            queue.append(callee)
        
        # 添加没有依赖关系的脚本
        for script in self.scripts.keys():
            if script not in execution_order:
                execution_order.append(script)
        
        self.execution_order = execution_order
        logger.info(f"Execution order: {execution_order}")

    def can_access_variable(self, caller_script: str, var_name: str, usage_line: int) -> bool:
        """检查脚本是否可以访问变量"""
        if caller_script not in self.variable_definitions_by_script:
            return False
        
        if var_name not in self.variable_definitions_by_script[caller_script]:
            return False
        
        def_line = self.variable_definitions_by_script[caller_script][var_name]
        return def_line < usage_line

    def get_variable_source_script(self, var_name: str, caller_script: str) -> Optional[str]:
        """获取变量的来源脚本"""
        # 首先检查调用脚本本身是否定义了该变量
        if (caller_script in self.variable_definitions_by_script and 
            var_name in self.variable_definitions_by_script[caller_script]):
            return caller_script
        
        # 检查启动脚本是否定义了该变量
        if var_name in self.entry_script_variable_scope:
            return "entry_script"
        
        # 根据执行顺序查找变量来源
        if self.execution_order:
            caller_index = -1
            try:
                caller_index = self.execution_order.index(caller_script)
            except ValueError:
                pass
            
            if caller_index >= 0:
                # 在调用脚本之前执行的脚本中查找变量定义
                for i in range(caller_index):
                    candidate_script = self.execution_order[i]
                    if (candidate_script in self.variable_definitions_by_script and 
                        var_name in self.variable_definitions_by_script[candidate_script]):
                        return candidate_script
        
        # 如果没有找到明确的来源，返回None
        return None


# 全局实例
_global_registry = None


def get_global_registry() -> GlobalMatlabRegistry:
    """获取全局注册表实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = GlobalMatlabRegistry()
    return _global_registry


def reset_global_registry():
    """重置全局注册表"""
    global _global_registry
    _global_registry = None