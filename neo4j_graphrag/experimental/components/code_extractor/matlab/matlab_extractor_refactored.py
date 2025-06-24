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
        enable_post_processing: bool = True,
        entry_script_path: Optional[str] = None,
        use_llm: bool = False,
    ):
        super().__init__(
            llm=llm,
            on_error=on_error,
            create_lexical_graph=create_lexical_graph,
            max_concurrency=max_concurrency,
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
        return any(node.id == node_id for node in self.nodes)

    def _add_node_if_not_exists(self, node_data: dict) -> None:
        """添加节点（如果不存在）"""
        if not self._node_exists(node_data['id']):
            # 转换节点数据为Neo4jNode格式
            label = node_data.get('labels', ['Unknown'])[0] if 'labels' in node_data else node_data.get('label', 'Unknown')
            properties = node_data.get('properties', {})

            # 确保属性兼容Neo4j
            properties = {
                k: ensure_neo4j_compatible(v)
                for k, v in properties.items()
            }

            neo4j_node = Neo4jNode(
                id=node_data['id'],
                label=label,
                properties=properties
            )
            self.nodes.append(neo4j_node)

    def _add_edge(self, source_id: str, target_id: str, label: str, properties: dict = None, line_number: int = None) -> None:
        """添加边"""
        if properties is None:
            properties = {}

        if line_number:
            properties['line_number'] = line_number

        # 确保属性兼容Neo4j
        properties = {
            k: ensure_neo4j_compatible(v)
            for k, v in properties.items()
        }

        # 确定节点类型
        start_node_type = self._get_node_type(source_id)
        end_node_type = self._get_node_type(target_id)

        neo4j_relationship = Neo4jRelationship(
            start_node_id=source_id,
            end_node_id=target_id,
            type=label,
            start_node_type=start_node_type,
            end_node_type=end_node_type,
            properties=properties
        )

        self.relationships.append(neo4j_relationship)

    def _get_node_type(self, node_id: str) -> str:
        """根据节点ID确定节点类型"""
        if node_id.startswith('script_'):
            return 'Script'
        elif node_id.startswith('func_') or node_id.startswith('function_'):
            return 'Function'
        elif node_id.startswith('var_') or node_id.startswith('variable_use_'):
            return 'Variable'
        else:
            return 'Unknown'

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
            if 'description' not in node.properties:
                code_snippet = get_code_snippet(node)
                try:
                    description = await self._generate_node_description(node, code_snippet)
                    node.properties['description'] = description
                except Exception as e:
                    logger.warning(f"Failed to generate description for node {node.id}: {e}")

    async def _generate_node_description(self, node: Neo4jNode, code_snippet: str) -> str:
        """为节点生成描述"""
        prompt = f"""
        Generate a brief description for this MATLAB code element:

        Type: {node.label}
        Name: {node.properties.get('name', 'Unknown')}
        Code: {code_snippet[:200]}...

        Description:
        """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Auto-generated description"

    def _post_process(self) -> None:
        """后处理：处理跨文件关系和变量作用域"""
        try:
            # 创建临时图进行后处理
            temp_graph = Neo4jGraph(nodes=self.nodes, relationships=self.relationships)

            # 应用后处理
            processed_graph = self.post_processor.post_process_cross_file_relationships(temp_graph)

            # 更新状态
            self.nodes = processed_graph.nodes
            self.relationships = processed_graph.relationships

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")

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
        """运行提取器"""
        if enable_post_processing is not None:
            self.enable_post_processing = enable_post_processing

        # 重置全局注册表
        reset_global_registry()

        # 处理所有块
        all_nodes = []
        all_relationships = []

        for chunk in chunks.chunks:
            try:
                # 只进行解析，不进行后处理
                result = await self.extract_for_chunk_without_post_processing(schema or GraphSchema(), examples, chunk)
                all_nodes.extend(result.nodes)
                all_relationships.extend(result.relationships)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                if self.on_error == OnError.RAISE:
                    raise

        # 创建临时图进行后处理
        temp_graph = Neo4jGraph(nodes=all_nodes, relationships=all_relationships)

        # 应用后处理（只调用一次）
        if self.enable_post_processing:
            processed_graph = self.post_processor.post_process_cross_file_relationships(temp_graph)
            all_nodes = processed_graph.nodes
            all_relationships = processed_graph.relationships

        # 创建最终图
        final_graph = Neo4jGraph(nodes=all_nodes, relationships=all_relationships)

        return MatlabExtractionResult(graph=final_graph)

    async def extract_for_chunk_without_post_processing(
        self, schema: GraphSchema, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """为单个代码块执行提取（不进行后处理）"""
        try:
            # 重置状态
            self._reset_state()

            # 解析MATLAB代码
            self._parse_matlab_code(chunk)

            # 生成描述（如果需要）
            if self.use_llm and self.llm:
                await self._generate_descriptions()

            # 不进行后处理 - 这是关键区别

            # 构建Neo4j图
            graph = Neo4jGraph(nodes=self.nodes, relationships=self.relationships)

            return graph

        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            if self.on_error == OnError.RAISE:
                raise
            return Neo4jGraph(nodes=[], relationships=[])

    @classmethod
    def reset_global_registry(cls):
        """重置全局注册表"""
        reset_global_registry()
