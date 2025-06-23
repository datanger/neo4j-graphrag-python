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

import asyncio
import json
import logging
import os
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.types import (
    TextChunk,
    TextChunks,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship
)
from neo4j_graphrag.experimental.pipeline.component import DataModel
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from pydantic import ValidationError

from .registry import get_global_registry, reset_global_registry
from .parser import MatlabCodeParser
from .post_processor import MatlabPostProcessor
from .utils import ensure_neo4j_compatible, get_code_snippet

# Define a custom return type that inherits from DataModel
class MatlabExtractionResult(DataModel):
    """Result model for MATLAB code extraction."""
    graph: Neo4jGraph

logger = logging.getLogger(__name__)


class MatlabExtractor(LLMEntityRelationExtractor):
    """MATLAB代码提取器，重构后的精简版本"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        on_error: OnError = OnError.IGNORE,
        create_lexical_graph: bool = True,
        max_concurrency: int = 5,
        debug: bool = False,
        enable_post_processing: bool = True,
        entry_script_path: Optional[str] = None,
        use_llm: bool = False,
    ):
        super().__init__(
            llm=llm,
            on_error=on_error,
            create_lexical_graph=create_lexical_graph,
            max_concurrency=max_concurrency,
            debug=debug,
        )
        
        self.enable_post_processing = enable_post_processing
        self.entry_script_path = entry_script_path
        self.use_llm = use_llm
        
        # 初始化组件
        self.parser = MatlabCodeParser()
        self.post_processor = MatlabPostProcessor()
        self.registry = get_global_registry()
        
        # 设置启动脚本
        if entry_script_path:
            self.registry.set_entry_script(entry_script_path)
        
        # 状态变量
        self.nodes = []
        self.relationships = []
        self.current_file_path = ""
        self.current_content = ""

    def _node_exists(self, node_id: str) -> bool:
        """检查节点是否已存在"""
        return any(node['id'] == node_id for node in self.nodes)

    def _add_node_if_not_exists(self, node_data: dict) -> None:
        """添加节点（如果不存在）"""
        if not self._node_exists(node_data['id']):
            # 确保属性兼容Neo4j
            if 'properties' in node_data:
                node_data['properties'] = {
                    k: ensure_neo4j_compatible(v) 
                    for k, v in node_data['properties'].items()
                }
            self.nodes.append(node_data)

    def _add_edge(self, source_id: str, target_id: str, label: str, properties: dict = None, line_number: int = None) -> None:
        """添加边"""
        edge_data = {
            'source': source_id,
            'target': target_id,
            'type': label,
            'properties': properties or {}
        }
        
        if line_number:
            edge_data['properties']['line_number'] = line_number
        
        # 确保属性兼容Neo4j
        edge_data['properties'] = {
            k: ensure_neo4j_compatible(v) 
            for k, v in edge_data['properties'].items()
        }
        
        self.relationships.append(edge_data)

    async def extract_for_chunk(
        self, schema: GraphSchema, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """为单个代码块执行提取"""
        try:
            # 重置状态
            self._reset_state()
            
            # 解析MATLAB代码
            self._parse_matlab_code(chunk)
            
            # 生成描述（如果需要）
            if self.use_llm and self.llm:
                await self._generate_descriptions()
            
            # 后处理
            if self.enable_post_processing:
                self._post_process()
            
            # 构建Neo4j图
            graph = Neo4jGraph(nodes=self.nodes, relationships=self.relationships)
            
            return graph
            
        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            if self.on_error == OnError.RAISE:
                raise
            return Neo4jGraph(nodes=[], relationships=[])

    def _parse_matlab_code(self, chunk: TextChunk) -> None:
        """解析MATLAB代码"""
        self.current_file_path = chunk.metadata.get('file_path', 'unknown')
        self.current_content = chunk.text
        
        # 使用解析器解析代码
        nodes, edges = self.parser.parse_matlab_code(self.current_file_path, self.current_content)
        
        # 添加节点和边
        for node in nodes:
            self._add_node_if_not_exists(node)
        
        for edge in edges:
            self._add_edge(
                edge['source'], 
                edge['target'], 
                edge['type'], 
                edge.get('properties', {})
            )
        
        # 注册到全局注册表
        self.registry.register_file_content(self.current_file_path, self.current_content)
        self.registry.add_nodes(nodes)
        self.registry.add_edges(edges)

    async def _generate_descriptions(self) -> None:
        """生成节点描述"""
        if not self.llm:
            return
        
        for node in self.nodes:
            if 'description' not in node.get('properties', {}):
                code_snippet = get_code_snippet(node)
                try:
                    description = await self._generate_node_description(node, code_snippet)
                    node['properties']['description'] = description
                except Exception as e:
                    logger.warning(f"Failed to generate description for node {node['id']}: {e}")

    async def _generate_node_description(self, node: Dict[str, Any], code_snippet: str) -> str:
        """生成单个节点的描述"""
        if not self.llm:
            return ""
        
        node_type = node['labels'][0] if node['labels'] else 'Unknown'
        node_name = node.get('properties', {}).get('name', 'Unknown')
        
        prompt = f"""
        Please provide a brief description for this MATLAB {node_type.lower()} named '{node_name}'.
        
        Code snippet:
        {code_snippet[:300]}...
        
        Description:"""
        
        try:
            response = await self.llm.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"MATLAB {node_type.lower()}: {node_name}"

    def _post_process(self) -> None:
        """后处理"""
        # 清理重复项
        self.post_processor.cleanup_duplicates(Neo4jGraph(nodes=self.nodes, relationships=self.relationships))
        
        # 提取变量和关系
        self.post_processor.extract_variables_and_relationships(Neo4jGraph(nodes=self.nodes, relationships=self.relationships))
        
        # 建立跨作用域关系
        self.post_processor.establish_cross_scope_relationships(Neo4jGraph(nodes=self.nodes, relationships=self.relationships))
        
        # 建立脚本到脚本调用
        self.post_processor.establish_script_to_script_calls(Neo4jGraph(nodes=self.nodes, relationships=self.relationships))

    def _reset_state(self) -> None:
        """重置状态"""
        self.nodes = []
        self.relationships = []
        self.current_file_path = ""
        self.current_content = ""

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[Any] = None,
        lexical_graph_config: Optional[Any] = None,
        schema: Optional[GraphSchema] = None,
        examples: str = "",
        enable_post_processing: Optional[bool] = None,
        rebuild_data: bool = False,
        **kwargs: Any,
    ) -> MatlabExtractionResult:
        """运行MATLAB代码提取"""
        # 重置全局注册表（如果需要）
        if rebuild_data:
            reset_global_registry()
            self.registry = get_global_registry()
            if self.entry_script_path:
                self.registry.set_entry_script(self.entry_script_path)
        
        # 设置后处理选项
        if enable_post_processing is not None:
            self.enable_post_processing = enable_post_processing
        
        # 执行提取
        if self.use_llm and self.llm:
            # 使用LLM进行提取
            graph = await super().run(
                chunks=chunks,
                document_info=document_info,
                lexical_graph_config=lexical_graph_config,
                schema=schema,
                examples=examples,
                **kwargs
            )
        else:
            # 使用纯Python实现
            all_nodes = []
            all_relationships = []
            
            # 处理每个代码块
            for chunk in chunks:
                try:
                    chunk_graph = await self.extract_for_chunk(schema or GraphSchema(), examples, chunk)
                    all_nodes.extend(chunk_graph.nodes)
                    all_relationships.extend(chunk_graph.relationships)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    if self.on_error == OnError.RAISE:
                        raise
            
            # 后处理跨文件关系
            if self.enable_post_processing:
                temp_graph = Neo4jGraph(nodes=all_nodes, relationships=all_relationships)
                processed_graph = self.post_processor.post_process_cross_file_relationships(temp_graph)
                all_nodes = processed_graph.nodes
                all_relationships = processed_graph.relationships
            
            graph = Neo4jGraph(nodes=all_nodes, relationships=all_relationships)
        
        return MatlabExtractionResult(graph=graph)

    @classmethod
    def reset_global_registry(cls):
        """重置全局注册表"""
        reset_global_registry()