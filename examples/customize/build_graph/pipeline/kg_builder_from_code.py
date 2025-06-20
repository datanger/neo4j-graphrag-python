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
import re
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import asyncio
import neo4j
from neo4j_graphrag.llm import OllamaLLM, LLMInterface
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import (
    MatlabExtractor,
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

# Add type annotation for the return type of the run method
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
                PropertyType(name="line_range", type="LIST", description="List of tuples containing variable usage in script and corresponding line range, each tuple element is like (context, start_line, end_line)"),
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
            description="A function or script defines a variable",
        ),
        RelationshipType(
            label="ASSIGNED_TO",
            description="A variable is assigned to another variable which is defined in the same function or script",
        ),
    ],
    patterns=[
        ("Function", "CALLS", "Function"),
        ("Function", "CALLS", "Script"),
        ("Script", "CALLS", "Function"),
        ("Script", "CALLS", "Script"),
        ("Function", "USES", "Variable"),
        ("Function", "DEFINES", "Variable"),
        ("Variable", "ASSIGNED_TO", "Variable"),
    ],
)

EXAMPLES = """
```json
{
  "nodes": [
    {
      "type": "Function",
      "id": "func1",
      "properties": {
        "name": "calculateSum",
        "file_path": "/path/to/matlab/functions/calc.m",
        "line_range": "10-25",
        "description": "Calculates the sum of two numbers",
        "parameters": "a, b",
        "returns": "result"
      }
    },
    {
      "type": "Variable",
      "id": "var1",
      "properties": {
        "name": "result",
        "file_path": "/path/to/matlab/functions/calc.m",
        "line_range": [
          ["function calculateSum", 10, 25]
        ]
      }
    },
    {
      "type": "Script",
      "id": "script1",
      "properties": {
        "name": "main_analysis",
        "file_path": "/path/to/matlab/scripts/main_analysis.m",
        "description": "Main analysis script that calls various functions"
      }
    },
    {
      "type": "Variable",
      "id": "var2",
      "properties": {
        "name": "data",
        "file_path": "/path/to/matlab/scripts/main_analysis.m",
        "line_range": [
          ["main script", 5, 5]
        ]
      }
    }
  ],
  "relationships": [
    {
      "type": "CALLS",
      "source_id": "script1",
      "source_type": "Script",
      "target_id": "func1",
      "target_type": "Function"
    },
    {
      "type": "USES",
      "source_id": "script1",
      "source_type": "Script",
      "target_id": "var2",
      "target_type": "Variable"
    },
    {
      "type": "DEFINES",
      "source_id": "func1",
      "source_type": "Function",
      "target_id": "var1",
      "target_type": "Variable"
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var2",
      "source_type": "Variable",
      "target_id": "var1",
      "target_type": "Variable"
    }
  ]
}
```
"""


async def process_matlab_files(directory: str, llm: LLMInterface) -> Dict[str, Any]:
    """Process MATLAB files in the given directory using the MatlabExtractor."""
    # Initialize the extractor with the LLM
    # extractor = LLMEntityRelationExtractor(llm=llm, prompt_template=CodeExtractionTemplate())
    extractor = MatlabExtractor(llm=llm)
    
    # Initialize graph data
    graph_data = {"nodes": [], "relationships": []}
    
    # Find all .m files in the directory
    matlab_files = list(Path(directory).rglob("*.m"))
    
    chunks = []
    for file_path in matlab_files:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a text chunk with required fields
        chunk = TextChunk(
            text=content,  # Use 'text' instead of 'content'
            index=0,       # Add index
            metadata={"file_path": str(file_path.relative_to(directory)), "file_name": file_path.name, "code_type": "matlab"}
        )
        chunks.append(chunk)
    
    # Create document info with required path field
    doc_info = DocumentInfo(
        path=str(file_path),
        metadata={"name": file_path.name}
    )
    
    # Process the chunk
    result = await extractor.run(
        chunks=TextChunks(chunks=chunks),
        schema=SCHEMA,
        document_info=doc_info,
        lexical_graph_config=LexicalGraphConfig(),
        examples=EXAMPLES,
    )
    
    # Add nodes and relationships to the graph
    if hasattr(result, 'nodes') and result.nodes:
        graph_data["nodes"].extend(result.nodes)
    if hasattr(result, 'relationships') and result.relationships:
        graph_data["relationships"].extend(result.relationships)
            
    return graph_data

async def main():
    # Initialize LLM
    llm = OllamaLLM(model_name="deepseek-r1:14b")
    
    # Directory containing MATLAB files
    matlab_dir = "/home/niejie/work/Code/tools/transformer-models"
    
    # Process MATLAB files
    graph_data = await process_matlab_files(matlab_dir, llm)
    
    # Create a Neo4jGraph object
    graph = Neo4jGraph(
        nodes=graph_data["nodes"],
        relationships=graph_data["relationships"]
    )
    
    # Neo4j connection settings
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    NEO4J_DB = "neo4j"
    
    try:
        # Initialize Neo4j driver
        driver = neo4j.GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        # Test the connection and verify database exists
        with driver.session(database="system") as session:
            result = session.run(
                "SHOW DATABASES WHERE name = $db_name", 
                {"db_name": NEO4J_DB}
            )
            if not result.single():
                print(f"Database '{NEO4J_DB}' does not exist. Creating it...")
                session.run(f"CREATE DATABASE {NEO4J_DB}")
        
        # Initialize Neo4j writer
        writer = Neo4jWriter(
            driver=driver,
            neo4j_database=NEO4J_DB
        )
        
        # Write to Neo4j
        print("Writing graph to Neo4j...")
        result = await writer.run(graph=graph, lexical_graph_config=LexicalGraphConfig())
        print(f"Successfully wrote graph to Neo4j: {result.status}")
        print("result", result)
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
        raise
    finally:
        # Close the driver when done (synchronously)
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    asyncio.run(main())