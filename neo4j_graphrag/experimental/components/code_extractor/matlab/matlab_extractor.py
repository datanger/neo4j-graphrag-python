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
重构后的MATLAB代码提取器

这个模块将原来的大型matlab_extractor.py文件拆分为多个更小、更专注的模块：

1. registry.py - 全局注册表，管理跨文件关系
2. parser.py - 代码解析器，处理MATLAB代码解析
3. post_processor.py - 后处理器，处理跨文件关系和变量作用域
4. utils.py - 工具函数，通用辅助函数
5. matlab_extractor_new.py - 主提取器，整合所有功能

主要改进：
- 代码模块化，每个模块职责单一
- 更好的可维护性和可测试性
- 减少代码重复
- 更清晰的代码结构
"""

from .matlab_extractor_refactored import MatlabExtractor, MatlabExtractionResult
from .registry import GlobalMatlabRegistry, get_global_registry, reset_global_registry
from .parser import MatlabCodeParser
from .post_processor import MatlabPostProcessor
from .utils import (
    ensure_neo4j_compatible,
    get_code_snippet,
    extract_variables_from_code,
    extract_function_calls_from_code,
    extract_script_calls_from_code,
    is_matlab_file,
    extract_file_name_from_path,
    create_node_id,
    validate_matlab_syntax,
    extract_function_signature,
    count_code_lines,
    extract_comments_from_code
)

__all__ = [
    'MatlabExtractor',
    'MatlabExtractionResult',
    'GlobalMatlabRegistry',
    'get_global_registry',
    'reset_global_registry',
    'MatlabCodeParser',
    'MatlabPostProcessor',
    'ensure_neo4j_compatible',
    'get_code_snippet',
    'extract_variables_from_code',
    'extract_function_calls_from_code',
    'extract_script_calls_from_code',
    'is_matlab_file',
    'extract_file_name_from_path',
    'create_node_id',
    'validate_matlab_syntax',
    'extract_function_signature',
    'count_code_lines',
    'extract_comments_from_code'
]