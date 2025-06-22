#!/usr/bin/env python3
"""
Debug script to test function parameter extraction
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor
from neo4j_graphrag.llm import LLMInterface

class MockLLM(LLMInterface):
    def __init__(self):
        super().__init__(model_name="mock")

    def invoke(self, input, message_history=None, system_instruction=None):
        from neo4j_graphrag.llm import LLMResponse
        return LLMResponse(content="Mock description")

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        from neo4j_graphrag.llm import LLMResponse
        return LLMResponse(content="Mock description")

async def debug_function_params():
    """Debug function parameter extraction."""

    # Create extractor
    llm = MockLLM()
    extractor = MatlabExtractor(llm=llm, debug=True)

    # Test file path
    test_file = Path(__file__).parent / "examples" / "helper_function.m"

    print(f"Testing file: {test_file}")
    print(f"File exists: {test_file.exists()}")

    if not test_file.exists():
        print("Test file not found!")
        return

    # Read file content
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"File content:\n{content}")

    # Create text chunk
    from neo4j_graphrag.experimental.components.types import TextChunk
    chunk = TextChunk(
        text=content,
        metadata={"file_path": str(test_file)},
        index=0
    )

    # Extract
    from neo4j_graphrag.experimental.components.schema import GraphSchema
    schema = GraphSchema(node_types=[])

    try:
        result = await extractor.extract_for_chunk(schema, "", chunk)
    except Exception as e:
        import traceback
        print("Exception during extraction:")
        traceback.print_exc()
        return

    print(f"\nExtraction result:")
    print(f"Nodes: {len(result.nodes)}")
    print(f"Relationships: {len(result.relationships)}")

    # Find function node
    function_nodes = [n for n in result.nodes if n.label == "Function"]
    print(f"\nFunction nodes: {len(function_nodes)}")

    for func_node in function_nodes:
        print(f"Function: {func_node.properties.get('name')}")
        print(f"Parameters: {func_node.properties.get('parameters')}")
        print(f"Function ID: {func_node.id}")

        # Find DEFINES relationships
        defines_rels = [r for r in result.relationships
                       if r.start_node_id == func_node.id and r.type == "DEFINES"]
        print(f"DEFINES relationships: {len(defines_rels)}")

        for rel in defines_rels:
            target_node = next((n for n in result.nodes if n.id == rel.end_node_id), None)
            if target_node:
                print(f"  - {target_node.properties.get('name')} (scope_id: {target_node.properties.get('scope_id')})")

        # Find variable nodes with matching scope_id
        param_vars = [n for n in result.nodes
                     if n.label == "Variable" and
                     n.properties.get('scope_id') == func_node.id]
        print(f"Variables with matching scope_id: {len(param_vars)}")

        for var in param_vars:
            print(f"  - {var.properties.get('name')} (scope_id: {var.properties.get('scope_id')})")

if __name__ == "__main__":
    asyncio.run(debug_function_params())
