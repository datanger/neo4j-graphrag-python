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

# Define a custom return type that inherits from DataModel
class MatlabExtractionResult(DataModel):
    """Result model for MATLAB code extraction."""
    graph: Neo4jGraph

logger = logging.getLogger(__name__)

# Constants for MATLAB code analysis
MATLAB_KEYWORDS = {
    "if", "else", "elseif", "end", "for", "while", "switch", "case", "otherwise",
    "try", "catch", "function", "return", "global", "persistent", "classdef",
    "properties", "methods", "events", "parfor", "spmd", "is"
}

# Load MATLAB built-in functions
with open(Path(__file__).parent / "matlab_builtin_functions.json", "r") as f:
    MATLAB_BUILTINS = json.load(f)

COMMON_WORDS_TO_IGNORE = {
    "on", "off", "assign", "assigns", "define", "defines", "use", "uses", 
    "variable", "variables", "parameter", "parameters", "input", "inputs", 
    "output", "outputs", "script", "local", "global", "calculation", "value", 
    "test", "file", "path", "content", "line", "range", "preview", 
    "dependencies", "generates", "description", "type", "id", "source", 
    "target", "label", "node", "edge", "graph", "element", "elements",
    "start", "if", "for", "while", "loop", "iter", "count", "index", 
    "idx", "step", "endfor", "endif", "is", "i", "j", "k", "x", "y", "z"
}

IDENTIFIERS_TO_EXCLUDE = set(MATLAB_KEYWORDS).union(set(MATLAB_BUILTINS)).union(COMMON_WORDS_TO_IGNORE)

class MatlabExtractor(LLMEntityRelationExtractor):
    """Extracts entities and relationships from MATLAB code using LLM for description generation."""
    
    def __init__(
        self,
        llm: LLMInterface,
        on_error: OnError = OnError.IGNORE,
        create_lexical_graph: bool = True,
        max_concurrency: int = 5,
    ):
        """Initialize the MATLAB code extractor.
        
        Args:
            llm: The language model to use for generating descriptions
            on_error: What to do when an error occurs during extraction
            create_lexical_graph: Whether to include text chunks in the graph
            max_concurrency: Maximum number of concurrent LLM requests
        """
        # Use the default ERExtractionTemplate which is already configured for entity-relation extraction
        prompt_template = ERExtractionTemplate()
        
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            create_lexical_graph=create_lexical_graph,
            on_error=on_error,
            max_concurrency=max_concurrency,
        )
        
        # Initialize MATLAB-specific attributes
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.known_function_names: Set[str] = set()
        self.script_level_vars: Set[str] = set()
        self.function_parameters: Dict[str, List[str]] = defaultdict(list)
        self.variable_occurrences: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self.variable_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.script_calls: Set[Tuple[str, str]] = set()
        self.current_function: Optional[str] = None
        self.text: str = ""
        self.path: str = ""
        self.lines: List[str] = []

        # Reset state for new chunk
        self._reset_state()

    async def extract_for_chunk(
        self, schema: GraphSchema, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """Extract entities and relationships from a single chunk of MATLAB code."""
        self.text = chunk.text
        self.path = chunk.metadata.get("file_path", "")
        self.lines = self.text.splitlines()
        
        # Reset state for this chunk
        self._reset_state()
        
        # Parse the MATLAB code
        self._parse_matlab_code()
        
        # Format the result according to Neo4jGraph requirements
        result = {
            "nodes": [],
            "relationships": []
        }
        
        # Add nodes with proper types and properties
        for node in self.nodes:
            node_type = node.pop("label")
            node_props = {k: v for k, v in node.items() if k not in ["id", "type"]}
            
            # Handle line_range formatting based on node type
            if node_type == "Variable" and "line_range" in node_props and isinstance(node_props["line_range"], str):
                # Convert string line_range to list of tuples for Variables
                try:
                    start, end = map(int, node_props["line_range"].split('-'))
                    node_props["line_range"] = [("", start, end)]
                except:
                    node_props["line_range"] = []
            
            result["nodes"].append({
                "type": node_type,
                "id": node["id"],
                "properties": node_props
            })
        
        # Add relationships with proper source/target types
        for edge in self.edges:
            source_node = next((n for n in self.nodes if n["id"] == edge["source"]), None)
            target_node = next((n for n in self.nodes if n["id"] == edge["target"]), None)
            
            if source_node and target_node:
                result["relationships"].append({
                    "type": edge["label"],
                    "source_id": edge["source"],
                    "source_type": source_node["label"],
                    "target_id": edge["target"],
                    "target_type": target_node["label"],
                    "properties": edge.get("properties", {})
                })
        
        # Validate and return the graph
        try:
            chunk_graph = Neo4jGraph.model_validate(result)
            logger.debug(f"Successfully created graph with {len(result['nodes'])} nodes and {len(result['relationships'])} relationships")
            return chunk_graph
        except ValidationError as e:
            logger.error(f"Failed to validate graph: {e}")
            if self.on_error == OnError.RAISE:
                raise
            return Neo4jGraph(nodes=[], relationships=[])
            

    def _reset_state(self) -> None:
        """Reset the internal state for processing a new chunk."""
        self.nodes = []
        self.edges = []
        self.known_function_names.clear()
        self.script_level_vars.clear()
        self.function_parameters.clear()
        self.variable_occurrences.clear()
        self.variable_dependencies.clear()
        self.script_calls.clear()
        self.current_function = None

    def _parse_matlab_code(self) -> None:
        """Parse MATLAB code to extract entities and relationships."""
        # This would contain the original parsing logic from the MatlabExtractor
        # For brevity, I'm including a simplified version
        if "function " in self.text:
            self._parse_function()
        else:
            self._parse_script()
            
        # Extract variables and relationships
        self._extract_variables_and_relationships()

    def _parse_function(self) -> None:
        """Parse a MATLAB function definition."""
        # Extract function signature
        func_match = re.search(
            r'^\s*function\s+(?:\[?([\w\s,]*)\]?\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?',
            self.text,
            re.MULTILINE
        )
        
        if not func_match:
            self._parse_script()
            return
            
        return_vars = [v.strip() for v in func_match.group(1).split(',')] if func_match.group(1) else []
        func_name = func_match.group(2)
        params = [p.strip() for p in func_match.group(3).split(',')] if func_match.group(3) else []
        
        # Get line range
        line_range = f"{self.text.count('\n', 0, func_match.start()) + 1}-{self.text.count('\n') + 1}"
        
        # Add function node
        func_id = f"func_{func_name}_{len(self.nodes)}"
        self.nodes.append({
            "id": func_id,
            "label": "Function",
            "name": func_name,
            "file_path": self.path,
            "line_range": line_range,
            "description": "",  # Can be filled by LLM later
            "parameters": ", ".join(params) if params else "",
            "returns": ", ".join(return_vars) if return_vars else "",
        })
        
        # Store function parameters for later relationship creation
        self.function_parameters[func_id] = params
        self.current_function = func_id
        
        self.current_function = func_id
        self.known_function_names.add(func_name)

    def _parse_script(self) -> None:
        """Parse a MATLAB script file."""
        script_name = os.path.basename(self.path)
        if not script_name.endswith('.m'):
            return
            
        # Add script node
        script_id = f"script_{script_name}"
        self.nodes.append({
            "id": script_id,
            "label": "Script",
            "name": script_name,
            "file_path": self.path,
            "description": ""  # Will be filled by LLM
        })
        self.current_function = None

    def _extract_variables_and_relationships(self) -> None:
        """Extract variables and their relationships from the code."""
        if not self.nodes:
            return
            
        for node in self.nodes:
            if node["label"] == "Function":
                self._process_function_variables(node)
            elif node["label"] == "Script":
                self._process_script_variables(node)
    
    def _process_function_variables(self, func_node: Dict[str, Any]) -> None:
        """Process variables within a function."""
        func_id = func_node["id"]
        func_name = func_node["name"]
        
        # Add DEFINES relationships for parameters
        for param in self.function_parameters.get(func_id, []):
            if not param:
                continue
                
            # Add variable node if it doesn't exist
            var_id = f"var_{param}_{len(self.nodes)}"
            self.nodes.append({
                "id": var_id,
                "label": "Variable",
                "name": param,
                "file_path": func_node["file_path"],
                "line_range": func_node["line_range"]
            })
            
            # Add DEFINES relationship
            self.edges.append({
                "source": func_id,
                "target": var_id,
                "label": "DEFINES",
                "properties": {}
            })
    
    def _process_script_variables(self, script_node: Dict[str, Any]) -> None:
        """Process variables within a script."""
        script_id = script_node["id"]
        
        # Find all variable assignments in the script
        var_pattern = r'(\w+)\s*=[^=]'
        for match in re.finditer(var_pattern, self.text):
            var_name = match.group(1).strip()
            if var_name in IDENTIFIERS_TO_EXCLUDE:
                continue
                
            # Add variable node if not exists
            var_id = f"{len(self.nodes)}"
            if not any(n.get('name') == var_name for n in self.nodes if n.get('label') == 'Variable'):
                self.nodes.append({
                    "id": var_id,
                    "label": "Variable",
                    "name": var_name,
                    "file_path": self.path,
                    "line_range": str(self.text.count('\n', 0, match.start()) + 1)
                })
            
            # Add DEFINES relationship if in a function/script
            if self.current_function:
                source_id = self.current_function
                self.edges.append({
                    "source": source_id,
                    "target": var_id,
                    "type": "DEFINES"
                })
        
        # Process usages
        for match in usages:
            var_name = match.group(1)
            if var_name in IDENTIFIERS_TO_EXCLUDE or var_name in self.known_function_names:
                continue
                
            # Check if this is a variable we're tracking
            var_node = next((n for n in self.nodes if n.get('name') == var_name and n.get('label') == 'Variable'), None)
            if not var_node:
                continue
                
            # Add USES relationship if in a function/script
            if self.current_function:
                source_id = self.current_function
                # Check if relationship already exists
                if not any(e.get('source') == source_id and e.get('target') == var_node['id'] and e.get('type') == 'USES' 
                          for e in self.edges):
                    self.edges.append({
                        "source": source_id,
                        "target": var_node['id'],
                        "type": "USES"
                    })

    async def _generate_descriptions(self) -> None:
        """Generate descriptions for functions and scripts using LLM."""
        for node in self.nodes:
            if node['label'] in ['Function', 'Script']:
                try:
                    # Generate a description using the LLM
                    prompt = f"Generate a brief description for this {node['label'].lower()}:"
                    if node['label'] == 'Function':
                        prompt += f"\nFunction: {node['name']}"
                        if node.get('parameters'):
                            prompt += f"\nParameters: {node['parameters']}"
                        if node.get('returns'):
                            prompt += f"\nReturns: {node['returns']}"
                    else:  # Script
                        prompt += f"\nScript: {node['name']}"
                    
                    # Add context from the code
                    prompt += f"\n\nCode:\n{self.text}"
                    
                    # Get description from LLM
                    description = await self.llm.generate(prompt)
                    node['description'] = description.strip()
                    
                except Exception as e:
                    logger.warning(f"Failed to generate description for {node['name']}: {str(e)}")
                    node['description'] = "Description not available"

    def _get_code_snippet(self, node: Dict[str, Any]) -> str:
        """Get the code snippet for a node."""
        if node["type"] == "Function":
            # Extract function code
            func_name = node["name"]
            func_match = re.search(
                rf'^\s*function\s+(?:\[?[\w\s,]*\]?\s*=\s*)?{re.escape(func_name)}\s*\([^)]*\)[^{{]*{{?',
                self.text,
                re.MULTILINE
            )
            if func_match:
                start = func_match.start()
                # Find the matching end
                brace_count = 0
                for i, char in enumerate(self.text[start:], start=start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return self.text[start:i+1]
        
        # Default to first 20 lines for scripts or if function parsing fails
        return '\n'.join(self.lines[:20])

    async def _generate_node_description(self, node: Dict[str, Any], code_snippet: str) -> str:
        """Generate a description for a node using the LLM."""
        prompt = f"""
        Analyze the following MATLAB {node["type"].lower()} and generate a concise description:
        
        {code_snippet}
        
        Description:
        """
        
        try:
            response = await self.llm.generate(prompt)
            return response.strip()
        except Exception as e:
            if self.on_error == OnError.RAISE:
                raise
            logger.warning(f"Failed to generate description for {node['name']}: {str(e)}")
            return ""

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[Any] = None,
        lexical_graph_config: Optional[Any] = None,
        schema: Optional[GraphSchema] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> MatlabExtractionResult:
        """Run the extraction pipeline with MATLAB-specific processing.
        
        Returns:
            MatlabExtractionResult: The result containing the extracted graph.
        """
        # Ensure we have a schema
        schema = schema or GraphSchema(node_types=[])
        
        # Run the parent class's run method
        result = await super().run(
            chunks=chunks,
            document_info=document_info,
            lexical_graph_config=lexical_graph_config,
            schema=schema,
            examples=examples,
            **kwargs,
        )
        
        # Ensure we return a properly typed result
        if not isinstance(result, Neo4jGraph):
            if isinstance(result, dict):
                # Convert dict to Neo4jGraph
                nodes = result.get('nodes', [])
                relationships = result.get('relationships', [])
                result = Neo4jGraph(nodes=nodes, relationships=relationships)
            else:
                # Convert the result to a Neo4jGraph if it's not already one
                result = Neo4jGraph(
                    nodes=getattr(result, 'nodes', []),
                    relationships=getattr(result, 'relationships', [])
                )
        
        # Ensure nodes and relationships are lists
        if not hasattr(result, 'nodes') or result.nodes is None:
            result.nodes = []
        if not hasattr(result, 'relationships') or result.relationships is None:
            result.relationships = []
        
        # Wrap in our custom result type
        return MatlabExtractionResult(graph=result)