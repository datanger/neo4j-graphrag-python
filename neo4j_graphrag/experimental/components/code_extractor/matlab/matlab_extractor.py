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

# Define a custom return type that inherits from DataModel
class MatlabExtractionResult(DataModel):
    """Result model for MATLAB code extraction."""
    graph: Neo4jGraph

logger = logging.getLogger(__name__)

# Global registry for cross-file relationships
class GlobalMatlabRegistry:
    """Global registry to track all MATLAB entities across files for post-processing."""
    
    def __init__(self):
        self.scripts = {}  # script_name -> script_node
        self.functions = {}  # func_name -> func_node
        self.all_nodes = []  # All nodes from all files
        self.all_edges = []  # All edges from all files
        self.call_sites = []  # All call sites for post-processing
        self.file_contents = {}  # file_path -> content for cross-file analysis
        
    def register_script(self, script_name: str, script_node: dict):
        """Register a script node."""
        self.scripts[script_name] = script_node
        
    def register_function(self, func_name: str, func_node: dict):
        """Register a function node."""
        self.functions[func_name] = func_node
        
    def add_nodes(self, nodes: List[dict]):
        """Add nodes from a file."""
        self.all_nodes.extend(nodes)
        
    def add_edges(self, edges: List[dict]):
        """Add edges from a file."""
        self.all_edges.extend(edges)
        
    def add_call_sites(self, call_sites: List[dict]):
        """Add call sites for post-processing."""
        self.call_sites.extend(call_sites)
        
    def register_file_content(self, file_path: str, content: str):
        """Register file content for cross-file analysis."""
        self.file_contents[file_path] = content

# Global instance
_global_registry = GlobalMatlabRegistry()

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
        debug: bool = False,
        enable_post_processing: bool = True,
    ):
        """Initialize the MATLAB code extractor.
        
        Args:
            llm: The language model to use for generating descriptions
            on_error: What to do when an error occurs during extraction
            create_lexical_graph: Whether to include text chunks in the graph
            max_concurrency: Maximum number of concurrent LLM requests
            debug: Enable debug output if True
            enable_post_processing: Enable cross-file relationship post-processing
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
        
        # Debug and optimization flags
        self.debug = debug
        self.enable_post_processing = enable_post_processing
        self.function_calls = set()  # Track function calls for deduplication
        self.processed_nodes = set()  # Track processed nodes for deduplication
        self.call_sites = {}  # Track call sites for better debugging
        self.variable_assignments = set()  # Track variable assignments
        self._line_offsets = None  # Cache for line number calculations
        
        # Track variable definitions and modifications
        self.variable_definitions = {}  # var_name -> (node_id, scope_type, scope_id)
        self.variable_scopes = []  # Stack of (scope_type, scope_id) for nested scopes
        self.current_scope = None  # Current scope (function/script) ID
        self.func_call_pattern = r'(?<![\w.])([a-zA-Z_]\w*)\s*(?=\()'

        # Reset state for new chunk
        self._reset_state()

    def _ensure_neo4j_compatible(self, value):
        """Ensure value is Neo4j-compatible type"""
        if value is None:
            return None
            
        # Handle numpy types
        if hasattr(value, 'item') and hasattr(value, 'dtype'):
            try:
                value = value.item()
            except:
                return str(value)
        
        # Handle standard Python types
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        elif isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, (float, np.floating)):
            if float(value).is_integer():
                return int(value)
            return float(value)
        elif isinstance(value, (list, tuple)):
            return [self._ensure_neo4j_compatible(x) for x in value]
        elif isinstance(value, dict):
            return {str(k): self._ensure_neo4j_compatible(v) for k, v in value.items()}
        elif isinstance(value, str):
            return value
        else:
            return str(value)
    
    async def extract_for_chunk(
        self, schema: GraphSchema, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        """Extract entities and relationships from a single chunk of MATLAB code."""
        self.text = chunk.text
        self.path = chunk.metadata.get("file_path", "")
        self.lines = self.text.splitlines()
        # Reset state for this chunk
        self._reset_state()
        
        # Register file content for cross-file analysis
        if self.enable_post_processing:
            _global_registry.register_file_content(self.path, self.text)
        
        # --- 第一遍：只创建脚本节点 ---
        script_name = os.path.basename(self.path)
        if script_name.endswith('.m'):
            script_id = f"script_{script_name}"
            if not any(n.get('id') == script_id for n in self.nodes):
                script_node = {
                    "id": script_id,
                    "label": "Script",
                    "name": script_name,
                    "file_path": self.path,
                    "description": "",
                    "line_range": ("", 1, len(self.lines))
                }
                self.nodes.append(script_node)
                
                # Register script in global registry
                if self.enable_post_processing:
                    _global_registry.register_script(script_name, script_node)
        
        # --- 第二遍：正常处理内容和关系 ---
        self._parse_matlab_code()
        
        # Register nodes and edges in global registry for post-processing
        if self.enable_post_processing:
            _global_registry.add_nodes(self.nodes)
            _global_registry.add_edges(self.edges)
            
            # Register functions found in this file
            for node in self.nodes:
                if node.get('label') == 'Function':
                    func_name = node.get('name')
                    if func_name:
                        _global_registry.register_function(func_name, node)
        
        # Create Neo4jGraph instance
        graph = Neo4jGraph()
        # Add nodes with proper types and properties
        for node in self.nodes:
            try:
                node_label = node.get("label")
                node_id = node.get("id")
                properties = {
                    str(k): self._ensure_neo4j_compatible(v) 
                    for k, v in node.items() 
                    if k not in ["id", "label", "type"]
                }
                if node_label == "Variable" and "line_range" in properties:
                    line_range = properties["line_range"]
                    if isinstance(line_range, str):
                        try:
                            start, end = map(int, line_range.split('-'))
                            properties["line_range"] = [("", start, end)]
                        except:
                            properties["line_range"] = []
                graph.nodes.append(
                    Neo4jNode(
                        id=str(node_id),
                        label=str(node_label),
                        properties=properties
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing node {node.get('id')}: {str(e)}")
                continue
        for edge in self.edges:
            try:
                source_id = edge.get("source")
                target_id = edge.get("target")
                rel_type = edge.get("label")
                if not all([source_id, target_id, rel_type]):
                    continue
                graph.relationships.append(
                    Neo4jRelationship(
                        start_node_id=str(source_id),
                        end_node_id=str(target_id),
                        type=str(rel_type),
                        properties={
                            k: self._ensure_neo4j_compatible(v) 
                            for k, v in edge.get("properties", {}).items()
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing edge {edge.get('source')} -> {edge.get('target')}: {str(e)}")
                continue
        logger.debug(f"Successfully created graph with {len(graph.nodes)} nodes and {len(graph.relationships)} relationships")
        return graph

    @classmethod
    def post_process_cross_file_relationships(cls, graph: Neo4jGraph) -> Neo4jGraph:
        """Post-process the graph to add cross-file script-to-script and script-to-function relationships.
        
        This method should be called after all files have been processed to establish
        cross-file relationships that couldn't be detected during single-file processing.
        
        Args:
            graph: The Neo4jGraph to post-process
            
        Returns:
            Neo4jGraph: The updated graph with cross-file relationships
        """
        if not _global_registry.scripts and not _global_registry.functions:
            logger.warning("No scripts or functions registered for post-processing")
            return graph
            
        logger.info(f"Post-processing cross-file relationships with {len(_global_registry.scripts)} scripts and {len(_global_registry.functions)} functions")
        
        # Create a mapping of node IDs to nodes for quick lookup
        node_map = {node.id: node for node in graph.nodes}
        
        # Process each file's content to find cross-file calls
        for file_path, content in _global_registry.file_contents.items():
            script_name = os.path.basename(file_path)
            if not script_name.endswith('.m'):
                continue
                
            # Find the script node for this file
            script_node = None
            for node in graph.nodes:
                if (node.label == "Script" and 
                    node.properties.get("name") == script_name):
                    script_node = node
                    break
                    
            if not script_node:
                continue
                
            # Find script calls in this file
            cls._find_script_calls_in_content(content, script_node, node_map, graph)
            
            # Find function calls in this file
            cls._find_function_calls_in_content(content, script_node, node_map, graph)
        
        logger.info(f"Post-processing complete. Graph now has {len(graph.relationships)} relationships")
        return graph
    
    @classmethod
    def _find_script_calls_in_content(cls, content: str, caller_node, node_map: dict, graph: Neo4jGraph):
        """Find script calls in the given content and add relationships."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
                
            # Pattern for direct script calls: script_name;
            script_call_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*;', line)
            if script_call_match:
                script_name = script_call_match.group(1)
                
                # Skip if this is a known keyword or built-in
                if script_name.lower() in IDENTIFIERS_TO_EXCLUDE:
                    continue
                
                # Find the called script (support both script_name and script_name.m)
                called_script = None
                for node in graph.nodes:
                    if (node.label == "Script" and 
                        (node.properties.get("name") == script_name or node.properties.get("name") == script_name + ".m")):
                        called_script = node
                        break
                
                if called_script:
                    # Check if this relationship already exists
                    existing_edge = next(
                        (rel for rel in graph.relationships 
                         if (rel.start_node_id == caller_node.id and 
                             rel.end_node_id == called_script.id and 
                             rel.type == "CALLS" and
                             rel.properties.get("line_number") == line_num)),
                        None
                    )
                    
                    if not existing_edge:
                        # Add CALLS relationship
                        graph.relationships.append(
                            Neo4jRelationship(
                                start_node_id=caller_node.id,
                                end_node_id=called_script.id,
                                type="CALLS",
                                properties={
                                    "file": caller_node.properties.get("file_path", ""),
                                    "line_number": line_num,
                                    "call_type": "script_call",
                                    "post_processed": True
                                }
                            )
                        )
                        logger.debug(f"Added cross-file script call: {caller_node.properties.get('name')} -> {script_name} at line {line_num}")
    
    @classmethod
    def _find_function_calls_in_content(cls, content: str, caller_node, node_map: dict, graph: Neo4jGraph):
        """Find function calls in the given content and add relationships."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
                
            # Pattern for function calls: func_name(...)
            func_call_match = re.search(r'(?<![\w.])([a-zA-Z_]\w*)\s*\([^)]*\)', line)
            if func_call_match:
                func_name = func_call_match.group(1)
                
                # Skip if this is a known keyword or built-in
                if func_name.lower() in IDENTIFIERS_TO_EXCLUDE:
                    continue
                    
                # Find the called function
                called_func = None
                for node in graph.nodes:
                    if (node.label == "Function" and 
                        node.properties.get("name") == func_name):
                        called_func = node
                        break
                        
                if called_func:
                    # Check if this relationship already exists
                    existing_edge = next(
                        (rel for rel in graph.relationships 
                         if (rel.start_node_id == caller_node.id and 
                             rel.end_node_id == called_func.id and 
                             rel.type == "CALLS" and
                             rel.properties.get("line_number") == line_num)),
                        None
                    )
                    
                    if not existing_edge:
                        # Add CALLS relationship
                        graph.relationships.append(
                            Neo4jRelationship(
                                start_node_id=caller_node.id,
                                end_node_id=called_func.id,
                                type="CALLS",
                                properties={
                                    "file": caller_node.properties.get("file_path", ""),
                                    "line_number": line_num,
                                    "call_type": "function_call",
                                    "post_processed": True
                                }
                            )
                        )
                        logger.debug(f"Added cross-file function call: {caller_node.properties.get('name')} -> {func_name} at line {line_num}")

    @classmethod
    def reset_global_registry(cls):
        """Reset the global registry. Useful for testing or when processing new sets of files."""
        global _global_registry
        _global_registry = GlobalMatlabRegistry()
        logger.info("Global MATLAB registry reset")

    def _reset_state(self) -> None:
        """Reset the internal state for processing a new chunk."""
        self.nodes = []
        self.edges = []
        self.known_function_names = set()
        self.script_level_vars = set()
        self.function_parameters = defaultdict(list)
        self.variable_occurrences = defaultdict(list)
        self.variable_dependencies = defaultdict(set)
        self.script_calls = set()
        self.current_function = None
        
        # Reset optimization and tracking variables
        self.function_calls = set()
        self.processed_nodes = set()
        self.call_sites = {}
        self.variable_assignments = set()
        self._line_offsets = None
        
        # Reset variable tracking state
        self.variable_definitions = {}
        self.variable_scopes = []
        self.current_scope = None

    def _parse_matlab_code(self) -> None:
        """Parse MATLAB code to extract entities and relationships."""
        if not self.text.strip():
            return
            
        # First, find all function definitions in the file
        function_pattern = re.compile(
            r'^\s*function\s+(?:\[?([\w\s,]*)\]?\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?',
            re.MULTILINE
        )
        
        # Find all function matches
        function_matches = list(function_pattern.finditer(self.text))
        
        if not function_matches:
            # No functions found, treat as a script
            self._parse_script()
        else:
            # Process each function
            for i, match in enumerate(function_matches):
                # Get the function's code block
                start_pos = match.start()
                end_pos = function_matches[i+1].start() if i+1 < len(function_matches) else len(self.text)
                func_code = self.text[start_pos:end_pos]
                
                # Save current text and set to function code
                original_text = self.text
                self.text = func_code
                
                try:
                    # Parse the function
                    self._parse_function()
                except Exception as e:
                    print(f"Error parsing function at position {start_pos}: {str(e)}")
                finally:
                    # Restore original text
                    self.text = original_text
        
        # Extract variables and relationships
        self._extract_variables_and_relationships()

    def _extract_variables_and_relationships(self) -> None:
        """Extract variables and their relationships from the parsed code.
        
        This method processes the collected variable definitions, occurrences, and dependencies
        to create appropriate nodes and edges in the graph.
        """
        if self.debug:
            print("Extracting variables and relationships...")
            print(f"Found {len(self.variable_definitions)} variable definitions")
        
        # Process each variable definition
        for var_name, (var_id, scope_type, scope_id) in self.variable_definitions.items():
            if var_name in IDENTIFIERS_TO_EXCLUDE:
                continue
                
            # Create variable node if it doesn't exist
            if not any(n.get('id') == var_id for n in self.nodes):
                self.nodes.append({
                    'id': var_id,
                    'label': 'Variable',
                    'name': var_name,
                    'scope_type': scope_type.lower(),
                    'scope_id': scope_id,
                    'file_path': self.path,
                })
            
            # Add DEFINES or ASSIGNED_TO relationship from scope to variable
            if scope_type.lower() == 'function':
                self._add_edge(
                    source_id=scope_id,
                    target_id=var_id,
                    label='DEFINES',
                    properties={'type': 'parameter' if var_name in self.function_parameters.get(scope_id, []) else 'local'}
                )
            elif scope_type.lower() == 'script':  # Only for script-level variables
                self._add_edge(
                    source_id=scope_id,
                    target_id=var_id,
                    label='DEFINES',
                    properties={'type': 'variable_definition'}
                )
            # Do not create ASSIGNED_TO for global variables
        
        # Process variable dependencies (USES relationships)
        for var_name, deps in self.variable_dependencies.items():
            if var_name not in self.variable_definitions:
                continue
                
            var_id = self.variable_definitions[var_name][0]
            for dep_var in deps:
                if dep_var in self.variable_definitions and dep_var not in IDENTIFIERS_TO_EXCLUDE:
                    dep_id = self.variable_definitions[dep_var][0]
                    self._add_edge(
                        source_id=var_id,
                        target_id=dep_id,
                        label='USES',
                        properties={'type': 'dependency'}
                    )
        
        # Process script calls
        for caller_id, script_name in self.script_calls:
            # Find the script node
            script_node = next((n for n in self.nodes 
                             if n.get('name') == script_name 
                             and n.get('label') == 'Script'), None)
            
            if script_node:
                self._add_edge(
                    source_id=caller_id,
                    target_id=script_node['id'],
                    label='CALLS',
                    properties={'type': 'script_call'}
                )
                
        if self.debug:
            print(f"Extracted {len(self.nodes)} nodes and {len(self.edges)} edges")

    def _parse_function(self) -> None:
        """Parse a MATLAB function definition.
        
        Extracts the function signature, parameters, return values, and code snippet.
        Creates a function node with appropriate metadata and stores it in the graph.
        """
        # Extract function signature using regex
        func_match = re.search(
            r'^\s*function\s+(?:\[?([\w\s,]*)\]?\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?',
            self.text,
            re.MULTILINE
        )
        
        if not func_match:
            # If no function signature is found, treat as a script
            self._parse_script()
            return
            
        # Extract return variables, function name, and parameters
        return_vars = [v.strip() for v in func_match.group(1).split(',')] if func_match.group(1) else []
        func_name = func_match.group(2)
        params = [p.strip() for p in func_match.group(3).split(',')] if func_match.group(3) else []
        
        # Get line range and code snippet
        start_line = self.text.count('\n', 0, func_match.start()) + 1
        end_line = self.text.count('\n') + 1
        func_code = self._get_function_code_snippet(func_match)
        
        # Create function ID
        func_id = f"func_{func_name}_{len(self.nodes)}"
        
        # Store function parameters for later use
        self.function_parameters[func_id] = params
        
        # Add function node
        self.nodes.append({
            "id": func_id,
            "label": "Function",
            "name": func_name,
            "file_path": self.path,
            "line_range": [
                (func_code.split('\n')[0].strip(), f"{start_line}-{end_line}")
            ],
            "parameters": params,
            "returns": return_vars,
            "description": ""  # Will be filled in by LLM later
        })
        
        # Set up scope tracking
        prev_scope = self.current_scope
        self.current_scope = func_id
        self.variable_scopes.append(('function', func_id))
        
        try:
            # Process function parameters as variables in this scope
            for param in params:
                if param and param not in IDENTIFIERS_TO_EXCLUDE:
                    # Create variable node for the parameter
                    var_id = f"var_{param}_{len(self.variable_definitions)}"
                    self.variable_definitions[param] = (var_id, 'Function', func_id)
                    
                    # Add the variable node
                    self.nodes.append({
                        "id": var_id,
                        "label": "Variable",
                        "name": param,
                        "file_path": self.path,
                        "line_range": [
                            (f"parameter: {param}", f"{start_line}-{start_line}")
                        ]
                    })
                    
                    # Add DEFINES relationship from function to parameter
                    self._add_edge(
                        source_id=func_id,
                        target_id=var_id,
                        label="DEFINES",
                        properties={
                            "file": self.path,
                            "line": start_line,
                            "type": "parameter_definition"
                        }
                    )
            
            # Process function body
            self.current_function = func_id
            self.known_function_names.add(func_name)
            
            # Process variables in the function body
            self._process_variables_in_code(
                func_code, 
                func_id, 
                self.path,
                start_line - 1  # Convert to 0-based line number
            )
            
        finally:
            # Restore previous scope
            self.current_scope = prev_scope
            if self.variable_scopes and self.variable_scopes[-1][1] == func_id:
                self.variable_scopes.pop()

    def _get_function_code_snippet(self, func_match) -> str:
        """Extract the full function code snippet."""
        start = func_match.start()
        brace_count = 0
        in_string = False
        string_char = None
        
        for i, char in enumerate(self.text[start:], start=start):
            # Handle string literals to avoid counting braces inside strings
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    # Check if this is an escaped quote
                    if i > 0 and self.text[i-1] == '\\':
                        continue
                    in_string = False
                continue
                    
            if in_string:
                continue
                
            # Count braces to find matching end
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return self.text[start:i+1].strip()
        
        # If we get here, return the rest of the text
        return self.text[start:].strip()

    def _parse_script(self) -> None:
        """Parse a MATLAB script file."""
        script_name = os.path.basename(self.path)
        if not script_name.endswith('.m'):
            return
            
        # Add script node
        script_id = f"script_{script_name}"
        
        # Get line range and code snippet (first 20 lines or all if less)
        code_lines = self.text.splitlines()
        end_line = min(20, len(code_lines))
        code_snippet = '\n'.join(code_lines[:end_line])
        line_range = (code_snippet, 1, end_line)
        
        # Set up script scope
        prev_scope = self.current_scope
        self.current_scope = script_id
        self.variable_scopes.append(('script', script_id))
        
        try:
            # Add script node to the graph
            self.nodes.append({
                "id": script_id,
                "label": "Script",
                "name": script_name,
                "file_path": self.path,
                "description": "",  # Will be filled by LLM later
                "line_range": line_range
            })
            
            # Process variables in the script
            self._process_variables_in_code(
                self.text, 
                script_id, 
                self.path,
                0  # Line offset is 0 for scripts
            )
            
        finally:
            # Restore previous scope
            self.current_scope = prev_scope
            if self.variable_scopes and self.variable_scopes[-1][1] == script_id:
                self.variable_scopes.pop()

    def _process_variables_in_code(self, code: str, parent_id: str, file_path: str, line_offset: int = 0) -> None:
        current_scope_vars = set()
        current_scope_type = 'Script' if 'script_' in parent_id else 'Function'
        current_scope = self.current_scope or parent_id
        func_params = self.function_parameters.get(parent_id, [])
        for param in func_params:
            if not param or param in IDENTIFIERS_TO_EXCLUDE:
                continue
            current_scope_vars.add(param)
            if param not in self.variable_definitions:
                var_node_id = f"var_{param}_{len(self.variable_definitions)}"
                self.variable_definitions[param] = (var_node_id, current_scope_type, current_scope)
                self.nodes.append({
                    "id": var_node_id,
                    "label": "Variable",
                    "name": param,
                    "file_path": file_path,
                    "line_range": [("parameter", f"{line_offset+1}-{line_offset+1}")]
                })
                self._add_edge(
                    source_id=parent_id,
                    target_id=var_node_id,
                    label="DEFINES",
                    properties={
                        "file": file_path,
                        "line": line_offset + 1,
                        "type": "parameter_definition"
                    }
                )
                if self.debug:
                    print(f"  [+] Defined parameter: {param} in {parent_id}")
        lines = code.split('\n')
        global_node_added = any(n.get('id') == 'global' for n in self.nodes)
        global_vars = set()
        
        # First pass: detect script calls to avoid treating them as variables
        script_calls_in_lines = {}
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            # Detect script calls like run('script.m'), eval('script.m'), or script_name;
            script_call_patterns = [
                r'(?:run\s*\(\s*[\'\"]|eval\s*\(\s*[\'\"])([^\'\"]+\.m)[\'\"]\s*\)',  # run('script.m') or eval('script.m')
                r'(?<![\w.])([a-zA-Z_]\w*)\s*;'  # script_name; (direct script call)
            ]
            for pattern in script_call_patterns:
                script_call_match = re.match(pattern, line)
                if script_call_match:
                    script_name = script_call_match.group(1) if script_call_match.lastindex >= 1 else script_call_match.group(0)
                    # Mark this line as a script call
                    script_calls_in_lines[line_num] = script_name
                    break
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            actual_line = line_offset + line_num
            
            # Skip if this line is a script call (will be handled by _process_script_calls)
            if line_num in script_calls_in_lines:
                continue
                
            # Detect global variable declarations
            global_match = re.match(r'^global\s+([a-zA-Z_][\w\s]*)', line)
            if global_match:
                if not global_node_added:
                    self.nodes.append({
                        "id": "global",
                        "label": "GlobalScope",
                        "name": "global"
                    })
                    global_node_added = True
                for gvar in global_match.group(1).split():
                    global_vars.add(gvar)
                    if gvar not in self.variable_definitions:
                        var_node_id = f"var_{gvar}_{len(self.variable_definitions)}"
                        self.variable_definitions[gvar] = (var_node_id, 'global', 'global')
                        self.nodes.append({
                            "id": var_node_id,
                            "label": "Variable",
                            "name": gvar,
                            "file_path": file_path,
                            "line_range": [(line.split('%')[0].strip(), f"{actual_line}")]
                        })
                        self._add_edge(
                            source_id=parent_id,
                            target_id=var_node_id,
                            label="DEFINES",
                            properties={
                                "file": file_path,
                                "line": actual_line,
                                "type": "global_variable_definition"
                            }
                        )
                        if self.debug:
                            print(f"  [+] Defined global variable: {gvar}")
                continue
            var_assign_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.+)$')
            assign_match = var_assign_pattern.match(line)
            assigned_var = None
            rhs_expr = None
            if assign_match:
                assigned_var = assign_match.group(1)
                rhs_expr = assign_match.group(2)
                if assigned_var in IDENTIFIERS_TO_EXCLUDE:
                    continue
                # If this is a global variable, always treat as global scope
                is_global = assigned_var in global_vars or (
                    assigned_var in self.variable_definitions and self.variable_definitions[assigned_var][1] == 'global')
                if assigned_var not in self.variable_definitions:
                    var_node_id = f"var_{assigned_var}_{len(self.variable_definitions)}"
                    scope_type = 'global' if is_global else current_scope_type
                    scope_id = 'global' if is_global else current_scope
                    self.variable_definitions[assigned_var] = (var_node_id, scope_type, scope_id)
                    self.nodes.append({
                        "id": var_node_id,
                        "label": "Variable",
                        "name": assigned_var,
                        "file_path": file_path,
                        "line_range": [(line.split('%')[0].strip(), f"{actual_line}")]
                    })
                    self._add_edge(
                        source_id=scope_id,
                        target_id=var_node_id,
                        label="DEFINES",
                        properties={
                            "file": file_path,
                            "line": actual_line,
                            "type": "variable_definition"
                        }
                    )
                    if self.debug:
                        print(f"  [+] Defined variable: {assigned_var} in {scope_id}")
                else:
                    var_node_id, def_scope_type, def_scope_id = self.variable_definitions[assigned_var]
                    is_global = def_scope_type == 'global' or assigned_var in global_vars
                    # MODIFIES for global or cross-scope
                    if (is_global and def_scope_id != current_scope) or (not is_global and def_scope_id != parent_id):
                        self._add_edge(
                            source_id=parent_id,
                            target_id=var_node_id,
                            label="MODIFIES",
                            properties={
                                "file": file_path,
                                "line": actual_line,
                                "original_scope_type": def_scope_type,
                                "original_scope_id": def_scope_id,
                                "modified_in_scope": current_scope,
                                "type": "cross_scope_modification"
                            }
                        )
                        if self.debug:
                            print(f"[DEBUG] Added MODIFIES relationship for '{assigned_var}' from {current_scope} (originally defined in {def_scope_type} {def_scope_id})")
                        # Update the scope where the variable was last modified
                        self.variable_definitions[assigned_var] = (var_node_id, def_scope_type if is_global else current_scope_type, current_scope if not is_global else 'global')
                current_scope_vars.add(assigned_var)
                if not parent_id.startswith('script_'):
                    for rhs_var in re.findall(r'\b([a-zA-Z_]\w*)\b', rhs_expr):
                        if (rhs_var in self.variable_definitions and 
                            rhs_var not in IDENTIFIERS_TO_EXCLUDE and
                            rhs_var != assigned_var):
                            rhs_node_id = self.variable_definitions[rhs_var][0]
                            # Only create ASSIGNED_TO if rhs_node_id is not 'global'
                            if rhs_node_id != 'global':
                                self._add_edge(
                                    source_id=rhs_node_id,
                                    target_id=var_node_id,
                                    label="ASSIGNED_TO",
                                    properties={
                                        "file": file_path,
                                        "line": actual_line,
                                        "type": "variable_assignment"
                                    }
                                )
                                if self.debug:
                                    print(f"  [=] {rhs_var} ASSIGNED_TO {assigned_var} at line {actual_line}")
            if not any(line.lstrip().startswith(keyword) for keyword in 
                      ['function ', 'if ', 'for ', 'while ', 'switch ', 'case ', 'else', 'end', 'global ', 'persistent ']):
                clean_line = re.sub(r"'.*?'", "''", line)
                clean_line = re.sub(r'".*?"', '""', clean_line)
                search_line = rhs_expr if rhs_expr is not None else clean_line
                for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', search_line):
                    var_name = match.group(1)
                    if assigned_var and var_name == assigned_var:
                        continue
                    if (var_name in IDENTIFIERS_TO_EXCLUDE or 
                        not var_name.isidentifier() or 
                        var_name.isdigit() or
                        var_name in MATLAB_KEYWORDS):
                        continue
                    if var_name in self.variable_definitions:
                        var_node_id, var_scope_type, var_scope_id = self.variable_definitions[var_name]
                        self._add_edge(
                            source_id=parent_id,
                            target_id=var_node_id,
                            label="USES",
                            properties={
                                "file": file_path,
                                "line": actual_line,
                                "type": "variable_usage"
                            }
                        )
                        if self.debug:
                            print(f"[DEBUG] {parent_id} USES variable '{var_name}' at line {actual_line}")
                    elif var_name not in current_scope_vars and rhs_expr is None:
                        var_node_id = f"var_{var_name}_{len(self.variable_definitions)}"
                        self.variable_definitions[var_name] = (
                            var_node_id, 
                            current_scope_type, 
                            current_scope
                        )
                        if not any(n.get('id') == var_node_id for n in self.nodes):
                            self.nodes.append({
                                "id": var_node_id,
                                "label": "Variable",
                                "name": var_name,
                                "file_path": file_path,
                                "line_range": [(line.split('%')[0].strip(), f"{actual_line}-{actual_line}")]
                            })
                        self._add_edge(
                            source_id=parent_id,
                            target_id=var_node_id,
                            label="DEFINES",
                            properties={
                                "file": file_path,
                                "line": actual_line,
                                "type": "variable_definition"
                            }
                        )
                        current_scope_vars.add(var_name)
                        if self.debug:
                            print(f"[DEBUG] New variable definition: {var_name} in {current_scope} "
                                 f"(type: {current_scope_type}, line: {actual_line})")
        self._process_function_calls(code, parent_id, file_path, line_offset)

    def _process_function_calls(self, code: str, caller_id: str, file_path: str, line_offset: int) -> None:
        """Process function calls in the given code."""
        # Initialize call sites if not already done
        if not hasattr(self, 'call_sites'):
            self.call_sites = {}
            
        # Get the current line number
        current_line = self._get_line_number(code, line_offset)
            
        # Skip if we've already processed this code block
        code_hash = hash(code.strip())
        if (caller_id, code_hash) in self.processed_nodes:
            return
                
        # Mark this code block as processed
        self.processed_nodes.add((caller_id, code_hash))
            
        # Debug output
        if self.debug:
            print(f"Processing function calls in {caller_id} at line {current_line}")
            
        # Process script calls first (run, eval, etc.)
        self._process_script_calls(code, caller_id, file_path, line_offset)
        
        # Enhanced function call pattern to handle various MATLAB function call formats
        func_call_patterns = [
            # Standard function calls: func()
            r'(?<![\w.])([a-zA-Z_]\w*)\s*\([^)]*\)',
            
            # Method calls: obj.method()
            r'([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)\s*\([^)]*\)',
            
            # System/command style: !command
            r'!\s*([a-zA-Z_]\w*)'
        ]
        
        for pattern in func_call_patterns:
            for match in re.finditer(pattern, code):
                # Handle different match groups based on pattern
                if match.lastindex == 1:
                    func_name = match.group(1)
                elif match.lastindex == 2:
                    # For method calls, use both object and method name
                    func_name = f"{match.group(1)}.{match.group(2)}"
                else:
                    continue
                
                # Skip if this is a known MATLAB function or keyword
                if func_name.lower() in MATLAB_KEYWORDS or func_name.lower() in ['disp', 'num2str']:
                    continue
                    
                # Find the function node (check both simple and fully qualified names)
                called_func = None
                for node in self.nodes:
                    if (node.get('label') == 'Function' and 
                        (node.get('name') == func_name or 
                         node.get('properties', {}).get('name') == func_name)):
                        called_func = node
                        break
                
                if called_func:
                    # Get the line number where this function is called
                    match_line = current_line + code[:match.start()].count('\n')
                    
                    # Store call site information
                    call_site = {
                        'caller': caller_id,
                        'callee': called_func['id'],
                        'line': match_line,
                        'file': file_path
                    }
                    
                    # Check if we've already created this call relationship
                    call_key = (caller_id, called_func['id'], match_line)
                    if call_key in self.call_sites:
                        continue
                        
                    self.call_sites[call_key] = call_site
                    
                    # Add CALLS relationship with line number
                    self._add_edge(
                        source_id=caller_id,
                        target_id=called_func['id'],
                        label='CALLS',
                        properties={
                            'file': file_path,
                            'line_number': match_line,
                            'call_type': 'function_call'
                        }
                    )
                    
                    if self.debug:
                        print(f"Added function call: {caller_id} -> {called_func['id']} at line {match_line}")

    def _process_script_calls(self, code: str, caller_id: str, file_path: str, line_offset: int) -> None:
        """Process script calls (run, eval, etc.) in the given code."""
        if self.debug:
            print(f"[DEBUG] Processing script calls in {caller_id}")
            
        # Pattern to match run('script.m'), run "script.m", or just script_name (implicit call)
        script_call_patterns = [
            r'(?:run\s*\(\s*[\'\"]|eval\s*\(\s*[\'\"])([^\'\"]+\.m)[\'\"]\s*\)',  # run('script.m') or eval('script.m')
            r'(?<![\w.])([a-zA-Z_]\w*)(?=\s*\()',  # script_name(...)
            r'(?<![\w.])([a-zA-Z_]\w*)\s*;'  # script_name; (direct script call)
        ]
        
        for pattern in script_call_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                script_name = match.group(1) if match.lastindex >= 1 else match.group(0)
                
                if self.debug:
                    print(f"[DEBUG] Found script call pattern: '{script_name}' with pattern: {pattern}")
                    
                # Clean up the script name - keep .m if present, otherwise add it
                if script_name.endswith('.m'):
                    clean_script_name = script_name
                    base_script_name = script_name[:-2]  # Remove .m
                else:
                    clean_script_name = script_name + '.m'
                    base_script_name = script_name
                    
                if self.debug:
                    print(f"[DEBUG] Clean script name: '{clean_script_name}', base name: '{base_script_name}'")
                    
                # Skip MATLAB keywords and built-ins
                if base_script_name.lower() in IDENTIFIERS_TO_EXCLUDE or base_script_name.lower() in self.known_function_names:
                    if self.debug:
                        print(f"[DEBUG] Skipping '{base_script_name}' - in exclude list or known functions")
                    continue
                
                # Find the called script in our nodes (try both with and without .m)
                called_script = next((n for n in self.nodes 
                                   if (n.get('name', '').lower() == clean_script_name.lower() or
                                       n.get('name', '').lower() == base_script_name.lower() or
                                       n.get('id', '').lower() == f"script_{clean_script_name.lower()}" or
                                       n.get('id', '').lower() == f"script_{base_script_name.lower()}")
                                   and n.get('label') == 'Script'), None)
                
                if self.debug:
                    if called_script:
                        print(f"[DEBUG] Found called script: {called_script.get('name')} (id: {called_script.get('id')})")
                    else:
                        print(f"[DEBUG] No script found for '{clean_script_name}' or '{base_script_name}'")
                        print(f"[DEBUG] Available script nodes: {[n.get('name') for n in self.nodes if n.get('label') == 'Script']}")
                    
                if called_script:
                    # Calculate the line number where this call occurs
                    match_line = self._get_line_number(code[:match.start()], line_offset) + 1
                        
                    # Check if this call relationship already exists
                    call_exists = any(
                        e.get('source') == caller_id and 
                        e.get('target') == called_script['id'] and 
                        e.get('label') == 'CALLS' and
                        e.get('properties', {}).get('line_number') == match_line
                        for e in self.edges
                    )
                        
                    if not call_exists:
                        # Add CALLS relationship with line number
                        self._add_edge(
                            source_id=caller_id,
                            target_id=called_script['id'],
                            label='CALLS',
                            properties={
                                'file': file_path,
                                'line_number': match_line,
                                'call_type': 'script_call'
                            }
                        )
                            
                        if self.debug:
                            print(f"Added script call: {caller_id} -> {called_script['id']} at line {match_line}")
                    else:
                        if self.debug:
                            print(f"[DEBUG] Script call already exists: {caller_id} -> {called_script['id']}")

    def _get_line_number(self, text: str, offset: int = 0) -> int:
        """Get the 1-based line number for the given character offset.
        
        Args:
            text: The text to search in (unused, kept for backward compatibility)
            offset: Character offset to start searching from
            
        Returns:
            int: 1-based line number, or -1 if not found
        """
        # Initialize line offsets if not already done
        if self._line_offsets is None:
            self._line_offsets = [0]  # Line 1 starts at position 0
            for i, c in enumerate(self.text):
                if c == '\n':
                    self._line_offsets.append(i + 1)  # Next line starts after newline
        
        # Handle case where offset is out of range
        if offset < 0 or offset >= len(self.text):
            return -1
                    
        # Find the line number for the given offset using binary search
        left, right = 0, len(self._line_offsets)
        while left < right:
            mid = (left + right) // 2
            if self._line_offsets[mid] > offset:
                right = mid
            else:
                left = mid + 1
                
        # Line numbers are 1-based, so return left as is (since we did left = mid + 1)
        return left

    def _add_edge(self, source_id: str, target_id: str, label: str, properties: dict = None, line_number: int = None) -> None:
        """Add an edge to the graph if it doesn't already exist with the same properties."""
        # Skip if source or target is None
        if not source_id or not target_id:
            return
            
        # Create a unique key for this edge
        edge_key = (source_id, target_id, label)
        
        # Initialize properties if not provided
        if properties is None:
            properties = {}
            
        # Add line number if provided
        if line_number is not None:
            properties['line_number'] = line_number
            
        # Check if we already have this exact edge
        for edge in self.edges:
            if (edge['source'] == source_id and 
                edge['target'] == target_id and 
                edge['label'] == label and 
                edge['properties'].get('line_number') == properties.get('line_number')):
                # Update properties if needed
                edge['properties'].update(properties)
                return
                
        # Add new edge
        self.edges.append({
            'source': source_id,
            'target': target_id,
            'label': label,
            'properties': properties or {}
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
        enable_post_processing: Optional[bool] = None,
        **kwargs: Any,
    ) -> MatlabExtractionResult:
        """Run the extraction pipeline with MATLAB-specific processing.
        
        Args:
            chunks: Text chunks to process
            document_info: Document information
            lexical_graph_config: Lexical graph configuration
            schema: Graph schema
            examples: Example text
            enable_post_processing: Override post-processing setting
            **kwargs: Additional arguments
            
        Returns:
            MatlabExtractionResult: The result containing the extracted graph.
        """
        # Override post-processing setting if provided
        if enable_post_processing is not None:
            self.enable_post_processing = enable_post_processing
        
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
        
        # Apply post-processing if enabled
        if self.enable_post_processing:
            logger.info("Applying post-processing for cross-file relationships...")
            result = self.post_process_cross_file_relationships(result)
        
        # Wrap in our custom result type
        return MatlabExtractionResult(graph=result)