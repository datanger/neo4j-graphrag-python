#!/usr/bin/env python3
#
#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
from typing import List, Optional, Union
import sys

# Add comprehensive logging to see all steps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

import asyncio
import neo4j
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import (
    MatlabExtractor, MatlabExtractionResult
)
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    TextChunk,
    TextChunks,
    DocumentInfo,
    LexicalGraphConfig,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.component import DataModel
from neo4j_graphrag.generation.prompts import PromptTemplate



# Schema definitions for MATLAB code analysis
SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Function",
            description="A code function definition",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the function"),
                PropertyType(name="file_path", type="STRING", description="Path to the file containing the function"),
                PropertyType(name="line_range", type="STRING", description="Line range where the function is defined"),
                PropertyType(name="description", type="STRING", description="Function description from docstring"),
                PropertyType(name="parameters", type="STRING", description="List of function parameters"),
                PropertyType(name="returns", type="STRING", description="List of return values"),
            ],
        ),
        NodeType(
            label="Variable",
            description="A variable used in the code",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the variable"),
                PropertyType(name="file_path", type="STRING", description="Path to the file where the variable is defined"),
                PropertyType(name="scope_id", type="STRING", description="ID of the scope (script or function) where this variable is defined"),
                PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' or 'function'"),
                PropertyType(name="line_range", type="LIST", description="List of tuples containing variable usage in script and corresponding line range, each tuple element is like (context, start_line-end_line)"),
            ],
        ),
        NodeType(
            label="Script",
            description="A code script file",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the script"),
                PropertyType(name="file_path", type="STRING", description="Path to the script file"),
                PropertyType(name="description", type="STRING", description="Script description"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(
            label="CALLS",
            description="A function or script calls another function or script"
        ),
        RelationshipType(
            label="USES",
            description="A function or script uses a variable which is defined in another function or script",
        ),
        RelationshipType(
            label="DEFINES",
            description="A script defines a variable or a function and a function defines a variable",
        ),
        RelationshipType(
            label="ASSIGNED_TO",
            description="A variable is assigned to another variable",
        ),
        RelationshipType(
            label="MODIFIES",
            description="A function or script modifies a variable that was defined in another scope",
        ),
    ],
    patterns=[
        ("Function", "CALLS", "Function"),
        ("Function", "CALLS", "Script"),
        ("Script", "CALLS", "Function"),
        ("Script", "CALLS", "Script"),
        ("Function", "USES", "Variable"),
        ("Script", "USES", "Variable"),
        ("Function", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Function"),
        ("Function", "MODIFIES", "Variable"),
        ("Script", "MODIFIES", "Variable"),
        ("Variable", "ASSIGNED_TO", "Variable"),
    ],
)

EXAMPLES = """

**关系模式说明：**

1. **CALLS 关系**：
   - `Script -> Function`: 脚本调用函数 (如 main_script.m 调用 helper_function)
   - `Script -> Script`: 脚本调用其他脚本 (如 main_script.m 调用 helper_script.m)
   - `Function -> Function`: 函数调用其他函数

2. **DEFINES 关系**：
   - `Script -> Variable`: 脚本定义变量 (如 main_script.m 定义 x, y, z)
   - `Function -> Variable`: 函数定义参数或局部变量 (如 modify_variables 定义 in1, out1)
   - `Script -> Function`: 脚本定义函数 (如脚本文件中定义的函数)

3. **USES 关系**：
   - `Script -> Variable`: 脚本使用外部变量 (如 main_script.m 使用 x, y)
   - `Function -> Variable`: 函数使用外部变量 (如 modify_variables 使用 in1)

4. **MODIFIES 关系**：
   - `Script -> Variable`: 脚本修改变量 (如 main_script.m 修改 x, y)
   - `Function -> Variable`: 函数修改变量

5. **ASSIGNED_TO 关系**：
   - `Variable -> Variable`: 变量赋值给其他变量 (如 x 赋值给 y, y 赋值给 z)

**变量作用域处理说明：**

**重要改进：每个脚本/函数中的变量都单独生成节点，即使变量名相同**

1. **变量ID格式**：`var_{变量名}_{作用域ID}`
   - 例如：`var_x_script_main_script` 表示 main_script.m 中的变量 x
   - 例如：`var_x_func_modify_variables` 表示 modify_variables 函数中的变量 x

2. **作用域隔离**：
   - 不同脚本中的同名变量被视为不同的节点
   - 不同函数中的同名变量被视为不同的节点
   - 每个变量节点包含 `scope_id` 和 `scope_type` 属性

3. **跨作用域关系**：
   - **USES 关系**：当一个脚本/函数使用另一个作用域中定义的变量时
   - **ASSIGNED_TO 关系**：当变量值从一个作用域传递到另一个作用域时

**实际案例说明：**

1. **同变量名在不同作用域**：
   - `main_script.m` 中定义变量 `x` → 节点 `var_x_script_main_script`
   - `modify_variables` 函数中也有变量 `x` → 节点 `var_x_func_modify_variables`
   - 这两个变量是完全独立的节点

2. **跨作用域变量使用**：
   - `main_script.m` 调用 `modify_variables(x, y)` 时
   - 创建关系：`script_main_script -[USES]-> var_x_script_main_script`
   - 创建关系：`script_main_script -[USES]-> var_y_script_main_script`

3. **变量赋值传递**：
   - 在 `main_script.m` 中：`y = x + 5`
   - 创建关系：`var_x_script_main_script -[ASSIGNED_TO]-> var_y_script_main_script`

4. **函数调用和变量传递**：
   - `main_script.m` 调用 `modify_variables(x, y)`
   - 函数内部处理：`out1 = in1 * 2`
   - 创建关系：`var_in1_func_modify_variables -[ASSIGNED_TO]-> var_out1_func_modify_variables`

**优势：**
- 避免变量名冲突
- 清晰的作用域边界
- 便于可视化展示
- 支持复杂的跨文件变量依赖分析


**输出格式：**
```json
{
  "nodes": [
    {
      "type": "Script",
      "id": "script_main_script",
      "properties": {
        "name": "main_script.m",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "description": "Main script that demonstrates various relationship patterns"
      }
    },
    {
      "type": "Script",
      "id": "script_helper_script",
      "properties": {
        "name": "helper_script.m",
        "file_path": "tests/matlab_test/examples/helper_script.m",
        "description": "Helper script called by main script"
      }
    },
    {
      "type": "Function",
      "id": "func_helper_function",
      "properties": {
        "name": "helper_function",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "parameters": "input_value",
        "returns": "result",
        "description": "Helper function that processes input"
      }
    },
    {
      "type": "Function",
      "id": "func_modify_variables",
      "properties": {
        "name": "modify_variables",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "parameters": "in1, in2",
        "returns": "out1, out2",
        "description": "Function that modifies its input variables"
      }
    },
    {
      "type": "Variable",
      "id": "var_x_script_main_script",
      "properties": {
        "name": "x",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 8, 8]]
      }
    },
    {
      "type": "Variable",
      "id": "var_y_script_main_script",
      "properties": {
        "name": "y",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 9, 9]]
      }
    },
    {
      "type": "Variable",
      "id": "var_z_script_main_script",
      "properties": {
        "name": "z",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 10, 10]]
      }
    },
    {
      "type": "Variable",
      "id": "var_result1_script_main_script",
      "properties": {
        "name": "result1",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 5, 5]]
      }
    },
    {
      "type": "Variable",
      "id": "var_in1_func_modify_variables",
      "properties": {
        "name": "in1",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 2, 2]]
      }
    },
    {
      "type": "Variable",
      "id": "var_out1_func_modify_variables",
      "properties": {
        "name": "out1",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 3, 3]]
      }
    },
    {
      "type": "Variable",
      "id": "var_x_func_modify_variables",
      "properties": {
        "name": "x",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 3, 3]]
      }
    },
    {
      "type": "Variable",
      "id": "var_y_func_modify_variables",
      "properties": {
        "name": "y",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 4, 4]]
      }
    }
  ],
  "relationships": [
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "func_helper_function",
      "target_type": "Function",
      "properties": {
        "call_type": "function_call",
        "line_number": 5
      }
    },
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "script_helper_script",
      "target_type": "Script",
      "properties": {
        "call_type": "script_call",
        "line_number": 8
      }
    },
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "func_modify_variables",
      "target_type": "Function",
      "properties": {
        "call_type": "function_call",
        "line_number": 13
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 8
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 9
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_z_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 10
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_result1_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 5
      }
    },
    {
      "type": "DEFINES",
      "source_id": "func_modify_variables",
      "source_type": "Function",
      "target_id": "var_in1_func_modify_variables",
      "target_type": "Variable",
      "properties": {
        "type": "parameter",
        "line_number": 2
      }
    },
    {
      "type": "DEFINES",
      "source_id": "func_modify_variables",
      "source_type": "Function",
      "target_id": "var_out1_func_modify_variables",
      "target_type": "Variable",
      "properties": {
        "type": "local",
        "line_number": 3
      }
    },
    {
      "type": "USES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "usage_type": "external_variable",
        "variable_name": "x",
        "line_number": 9
      }
    },
    {
      "type": "USES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "usage_type": "external_variable",
        "variable_name": "y",
        "line_number": 10
      }
    },
    {
      "type": "USES",
      "source_id": "func_modify_variables",
      "source_type": "Function",
      "target_id": "var_in1_func_modify_variables",
      "target_type": "Variable",
      "properties": {
        "usage_type": "external_variable",
        "variable_name": "in1",
        "line_number": 3
      }
    },
    {
      "type": "MODIFIES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_modification",
        "line_number": 13
      }
    },
    {
      "type": "MODIFIES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_modification",
        "line_number": 13
      }
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var_x_script_main_script",
      "source_type": "Variable",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_assignment",
        "line_number": 9
      }
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var_y_script_main_script",
      "source_type": "Variable",
      "target_id": "var_z_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_assignment",
        "line_number": 10
      }
    }
  ]
}
```

"""


class Neo4jGraphResult(DataModel):
    graph: Neo4jGraph

class CodeExtractionTemplate(PromptTemplate):
    DEFAULT_TEMPLATE = """
You are a top-tier algorithm designed for extracting a labeled property graph schema in
structured formats.

Generate a generalized graph schema based on the input text. Identify key entity types,
their relationship types, and property types.

IMPORTANT RULES:
1. Return only abstract schema information, not concrete instances.
2. Use singular PascalCase labels for entity types (e.g., Person, Company, Product).
3. Use UPPER_SNAKE_CASE for relationship types (e.g., WORKS_FOR, MANAGES).
4. Include property definitions only when the type can be confidently inferred, otherwise omit them.
5. When defining potential_schema, ensure that every entity and relation mentioned exists in your entities and relations lists.
6. Do not create entity types that aren't clearly mentioned in the text.
7. Keep your schema minimal and focused on clearly identifiable patterns in the text.

Accepted property types are: BOOLEAN, DATE, DURATION, FLOAT, INTEGER, LIST,
LOCAL_DATETIME, LOCAL_TIME, POINT, STRING, ZONED_DATETIME, ZONED_TIME.

Return a valid JSON object that follows this precise structure:
{schema}

Return Examples:
{examples}

Please return a JSON object according to the above instructions based on the following file path and input text.

File path:
{file_path}

Input text:
```{code_type}
{text}
```
"""
    EXPECTED_INPUTS = ["text"]

    def format(
        self,
        text: str = "",
        schema: str = "",
        file_path: str = "",
        examples: str = "",
        code_type: str = "matlab",
    ) -> str:
        return super().format(text=text, schema=schema, file_path=file_path, examples=examples, code_type=code_type)

class MockLLM:
    """Mock LLM for demonstration purposes."""

    async def generate(self, prompt: str) -> str:
        """Return a mock description."""
        return "Mock description generated by LLM"

def convert_to_native(value):
    """Recursively convert value and all its contents to native Python types."""
    try:
        if value is None:
            return None

        # Handle numpy and other numeric types
        if hasattr(value, 'item') and hasattr(value, 'dtype'):
            value = value.item()  # Convert numpy scalar to Python native

        # Handle basic types
        if isinstance(value, bool):
            return value

        # Handle string type
        if isinstance(value, str):
            # Try to convert to number if it looks like a number
            try:
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                if '.' in value or 'e' in value.lower():
                    fval = float(value)
                    return int(fval) if fval.is_integer() else fval
                return int(value)
            except (ValueError, TypeError):
                return value

        # Handle numeric types - ensure they're native Python types
        if isinstance(value, (int, float)):
            # Convert to int if it's a whole number, otherwise float
            if float(value).is_integer():
                return int(value)
            return float(value)

        # Handle numpy numeric types
        if 'numpy' in str(type(value)) and hasattr(value, 'item'):
            try:
                return convert_to_native(value.item())
            except:
                pass

        # Handle lists and tuples
        if isinstance(value, (list, tuple)):
            return [convert_to_native(x) for x in value]

        # Handle dictionaries
        if isinstance(value, dict):
            return {str(k): convert_to_native(v) for k, v in value.items()}

        # Handle datetime objects
        if hasattr(value, 'isoformat'):
            return value.isoformat()

        # Try to convert to a basic type
        try:
            # Try to get a primitive representation
            if hasattr(value, '__dict__'):
                return {str(k): convert_to_native(v) for k, v in value.__dict__.items()}

            # Try to convert to string as last resort
            str_val = str(value)
            if str_val != str(type(value)):  # Only return if it's a meaningful string representation
                return str_val

            # If we get here, the default string representation isn't helpful
            return None

        except Exception as e:
            print(f"Warning: Could not convert value {value} of type {type(value)}: {e}")
            return None

    except Exception as e:
        print(f"Error in convert_to_native for value {value} of type {type(value)}: {e}")
        return None

def convert_value(value):
    """Convert value to a Neo4j-compatible type with detailed logging."""
    try:
        original_type = type(value).__name__
        converted = convert_to_native(value)

        # Log conversion if the type changed
        if converted is not None and str(converted) != str(value):
            print(f"  Converted {original_type}: {value!r} -> {type(converted).__name__}: {converted!r}")

        return converted
    except Exception as e:
        print(f"Error converting value {value!r} of type {type(value)}: {e}")
        return None

def convert_item(item):
    """Convert a single item to a Neo4j-compatible type."""
    return convert_value(item)

def validate_properties(properties, context):
    """Validate and convert properties to Neo4j-compatible types."""
    if not properties:
        return {}

    valid_props = {}
    for key, value in properties.items():
        try:
            # Skip None values
            if value is None:
                continue

            # Convert the value
            converted = convert_value(value)

            # Check for problematic types
            if converted is not None:
                if isinstance(converted, (list, tuple, dict)):
                    # Check nested structures
                    def check_nested(v):
                        if isinstance(v, (list, tuple)):
                            return all(check_nested(x) for x in v)
                        if isinstance(v, dict):
                            return all(isinstance(k, str) and check_nested(x) for k, x in v.items())
                        return isinstance(v, (str, int, float, bool, type(None)))

                    if not check_nested(converted):
                        print(f"  WARNING: {context} has non-serializable nested type in property '{key}': {converted}")
                        converted = str(converted)

                valid_props[key] = converted

        except Exception as e:
            print(f"  ERROR: Failed to convert {context} property '{key}': {e}")
            try:
                valid_props[key] = str(value)
            except:
                print(f"  ERROR: Could not stringify property '{key}', skipping")

    return valid_props

def ensure_neo4j_compatible(value, path=''):
    """Ensure a value is Neo4j-compatible, converting if necessary."""
    if value is None:
        return None

    # Handle numpy and other numeric types
    if hasattr(value, 'item') and hasattr(value, 'dtype'):
        try:
            return ensure_neo4j_compatible(value.item(), path)
        except Exception:
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
    elif isinstance(value, str):
        return value
    elif isinstance(value, (list, tuple)):
        return [ensure_neo4j_compatible(x, f"{path}[{i}]") for i, x in enumerate(value)]
    elif isinstance(value, dict):
        return {str(k): ensure_neo4j_compatible(v, f"{path}.{k}" if path else k) for k, v in value.items()}
    else:
        print(f"Converting unsupported type {type(value).__name__} to string at {path}")
        return str(value)

async def process_matlab_files(directory: str, llm: LLMInterface) -> 'MatlabExtractionResult':
    """Process MATLAB files in the given directory using the MatlabExtractor."""
    # Initialize the extractor with the LLM
    # extractor = LLMEntityRelationExtractor(llm=llm, prompt_template=CodeExtractionTemplate())
    extractor = MatlabExtractor(llm=llm)

    # Find all .m files in the directory
    matlab_files = list(Path(directory).rglob("*.m"))

    # Initialize result with empty graph
    result_graph = Neo4jGraph()

    # Process each file individually
    for file_path in matlab_files:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create a text chunk with required fields
        chunk = TextChunk(
            text=content,
            index=0,
            metadata={"file_path": str(file_path), "file_name": file_path.name, "code_type": "matlab"}
        )

        # Create document info with required path field for this file
        doc_info = DocumentInfo(
            path=str(file_path),
            metadata={"name": file_path.name}
        )

        # Process the current file, but disable post-processing inside the loop
        file_result = await extractor.run(
            chunks=TextChunks(chunks=[chunk]),
            schema=SCHEMA,
            document_info=doc_info,
            lexical_graph_config=LexicalGraphConfig(),
            examples=EXAMPLES,
            enable_post_processing=False,
        )

        # Merge the results
        if file_result and hasattr(file_result, 'graph') and file_result.graph:
            result_graph.nodes.extend(file_result.graph.nodes)
            result_graph.relationships.extend(file_result.graph.relationships)

    # After processing all files, run the post-processing step once on the aggregated graph
    print("Applying final cross-file relationship post-processing...")
    final_graph = MatlabExtractor.post_process_cross_file_relationships(result_graph)

    # Return the combined result graph
    result = type('Result', (), {'graph': final_graph})()

    # Process all nodes with detailed validation
    for i, node in enumerate(result.graph.nodes):
        if not hasattr(node, 'properties'):
            print(f"Node {i} has no properties")
            node.properties = {}
            continue

        node_id = getattr(node, 'id', f'node_{i}')
        node_label = getattr(node, 'label', 'UNKNOWN')
        print(f"\nValidating node {i} ({node_label}): {node_id}")

        # Convert line_range if it exists
        if 'line_range' in node.properties:
            try:
                line_ranges = node.properties['line_range']
                if isinstance(line_ranges, list) and line_ranges and isinstance(line_ranges[0], (list, tuple)) and len(line_ranges[0]) == 2:
                    # Convert list of tuples to a string representation
                    line_ranges_str = '; '.join(
                        f'"{str(code)}" at lines {str(line_range)}'
                        for code, line_range in line_ranges
                    )
                    node.properties['line_range'] = line_ranges_str
                elif line_ranges is not None:
                    print(f"  Warning: Unexpected line_range format: {type(line_ranges)}")
            except Exception as e:
                print(f"  Error processing line_range: {e}")

        # Validate and convert all properties
        node.properties = validate_properties(node.properties, f"node {node_id}")

    # Process all relationships with detailed validation
    for i, rel in enumerate(result.graph.relationships):
        if not hasattr(rel, 'properties'):
            print(f"Relationship {i} has no properties")
            rel.properties = {}
            continue

        rel_type = getattr(rel, 'type', 'UNKNOWN')
        rel_id = f"{getattr(rel, 'start_node_id', '?')} -[{rel_type}]-> {getattr(rel, 'end_node_id', '?')}"
        print(f"\nValidating relationship {i}: {rel_id}")

        # Validate and convert all properties
        rel.properties = validate_properties(rel.properties, f"relationship {i}")

    return result

async def main():
    # Initialize LLM
    llm = MockLLM()

    # Directory containing MATLAB files
    matlab_dir = "D:\\Work\\transformer-models"

    # Neo4j connection settings
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    NEO4J_DB = "neo4j"

    driver = None  # Initialize driver to None
    try:
        # Initialize Neo4j driver
        driver = neo4j.GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # --- Clear the database before writing ---
        print("Connecting to Neo4j to clear the database...")
        with driver.session(database=NEO4J_DB) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared successfully.")

        # Process MATLAB files and get the extraction result
        result = await process_matlab_files(matlab_dir, llm)
        graph = result.graph  # Access the Neo4jGraph from the result
        print(f"Extracted graph with {len(graph.nodes)} nodes and {len(graph.relationships)} relationships")

        # Initialize Neo4j writer
        writer = Neo4jWriter(
            driver=driver,
            neo4j_database=NEO4J_DB
        )

        # Convert graph nodes and relationships to dictionaries
        nodes = []
        relationships = []

        # Process nodes with property validation
        for node in graph.nodes:
            try:
                node_dict = dict(node)
                # Convert properties to native types
                properties = {}
                for k, v in node_dict.get('properties', {}).items():
                    try:
                        properties[k] = ensure_neo4j_compatible(v, f"node.{node_dict.get('id', '?')}.{k}")
                    except Exception as e:
                        print(f"Error processing node {node_dict.get('id', '?')} property '{k}': {e}")

                # Create a clean node dictionary
                clean_node = {
                    'id': str(node_dict.get('id', '')),
                    'label': str(node_dict.get('label', '')),
                    'properties': properties
                }
                nodes.append(clean_node)
            except Exception as e:
                print(f"Error processing node {getattr(node, 'id', '?')}: {e}")

        # Process relationships with property validation
        for rel in graph.relationships:
            try:
                rel_dict = dict(rel)
                # Convert properties to native types
                properties = {}
                for k, v in rel_dict.get('properties', {}).items():
                    try:
                        properties[k] = ensure_neo4j_compatible(v, f"rel.{rel_dict.get('start_node_id', '?')}-{rel_dict.get('type', '?')}->{rel_dict.get('end_node_id', '?')}.{k}")
                    except Exception as e:
                        print(f"Error processing relationship property '{k}': {e}")

                # Create a clean relationship dictionary
                clean_rel = {
                    'start_node_id': str(rel_dict.get('start_node_id', '')),
                    'end_node_id': str(rel_dict.get('end_node_id', '')),
                    'type': str(rel_dict.get('type', '')),
                    'properties': properties
                }
                relationships.append(clean_rel)
            except Exception as e:
                print(f"Error processing relationship: {e}")

        print(f"Processed {len(nodes)} nodes and {len(relationships)} relationships with Neo4j-compatible types")

        # Create a clean Neo4jGraph object with validated data
        from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship

        # Create Neo4jNode objects
        print("Creating Neo4j nodes and relationships...")
        neo4j_nodes = []
        for node in nodes:
            try:
                neo4j_node = Neo4jNode(
                    id=node['id'],
                    label=node['label'],
                    properties=node.get('properties', {})
                )
                neo4j_nodes.append(neo4j_node)
            except Exception as e:
                print(f"Error creating Neo4j node {node.get('id', '?')}: {e}")

        # Create Neo4jRelationship objects
        neo4j_relationships = []
        for rel in relationships:
            try:
                neo4j_rel = Neo4jRelationship(
                    start_node_id=rel['start_node_id'],
                    end_node_id=rel['end_node_id'],
                    type=rel['type'],
                    properties=rel.get('properties', {})
                )
                neo4j_relationships.append(neo4j_rel)
            except Exception as e:
                rel_id = f"{rel.get('start_node_id', '?')} -[{rel.get('type', '?')}]-> {rel.get('end_node_id', '?')}"
                print(f"Error creating relationship {rel_id}: {e}")

        # Create the final graph
        neo4j_graph = Neo4jGraph(nodes=neo4j_nodes, relationships=neo4j_relationships)

        # Write to Neo4j
        print(f"Writing {len(neo4j_nodes)} nodes and {len(neo4j_relationships)} relationships to Neo4j...")
        writer_result = await writer.run(neo4j_graph)
        print(f"Successfully wrote to Neo4j: {writer_result.status}")

    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the driver when done
        if driver:
            driver.close()
            print("Neo4j driver closed.")

if __name__ == "__main__":
    asyncio.run(main())
