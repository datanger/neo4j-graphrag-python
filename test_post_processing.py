#!/usr/bin/env python3
"""
Test script to demonstrate the post-processing functionality for cross-file relationships
in the MATLAB extractor.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Mock LLM for testing
class MockLLM:
    async def generate(self, prompt):
        return "Mock description"

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import (
    MatlabExtractor, 
    TextChunk,
    TextChunks
)

async def test_post_processing():
    """Test the post-processing functionality with multiple MATLAB files."""
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create script1.m that calls script2 and function1
        script1_content = """
% Script 1 - calls other scripts and functions
x = 10;
y = 20;
script2;  % Direct script call
result = function1(x, y);  % Function call
disp(result);
"""
        
        # Create script2.m
        script2_content = """
% Script 2 - simple script
a = 5;
b = 15;
c = a + b;
disp(c);
"""
        
        # Create function1.m
        function1_content = """
function result = function1(x, y)
    % Function that adds two numbers
    result = x + y;
end
"""
        
        # Write test files
        script1_path = os.path.join(temp_dir, "script1.m")
        script2_path = os.path.join(temp_dir, "script2.m")
        function1_path = os.path.join(temp_dir, "function1.m")
        
        with open(script1_path, 'w') as f:
            f.write(script1_content)
        with open(script2_path, 'w') as f:
            f.write(script2_content)
        with open(function1_path, 'w') as f:
            f.write(function1_content)
        
        # Create extractor with post-processing enabled
        llm = MockLLM()
        extractor = MatlabExtractor(
            llm=llm,
            debug=True,
            enable_post_processing=True
        )
        
        # Create text chunks for all files
        chunks_list = []
        for i, file_path in enumerate([script1_path, script2_path, function1_path]):
            with open(file_path, 'r') as f:
                content = f.read()
            chunk = TextChunk(
                text=content,
                index=i,
                metadata={"file_path": file_path}
            )
            chunks_list.append(chunk)
        
        chunks = TextChunks(chunks=chunks_list)
        
        print("Processing MATLAB files...")
        
        # Process all chunks
        result = await extractor.run(chunks=chunks)
        graph = result.graph
        
        print(f"\nExtraction complete!")
        print(f"Total nodes: {len(graph.nodes)}")
        print(f"Total relationships: {len(graph.relationships)}")
        
        # Print nodes
        print("\nNodes:")
        for node in graph.nodes:
            print(f"  - {node.label}: {node.properties.get('name', node.id)}")
        
        # Print relationships
        print("\nRelationships:")
        for rel in graph.relationships:
            source_name = next((n.properties.get('name', n.id) for n in graph.nodes if n.id == rel.start_node_id), rel.start_node_id)
            target_name = next((n.properties.get('name', n.id) for n in graph.nodes if n.id == rel.end_node_id), rel.end_node_id)
            post_processed = rel.properties.get('post_processed', False)
            print(f"  - {source_name} --[{rel.type}]--> {target_name} (post_processed: {post_processed})")
        
        # Count different types of relationships
        calls_relationships = [r for r in graph.relationships if r.type == "CALLS"]
        script_to_script = [r for r in calls_relationships if 
                           any(n.label == "Script" for n in graph.nodes if n.id == r.start_node_id) and
                           any(n.label == "Script" for n in graph.nodes if n.id == r.end_node_id)]
        script_to_function = [r for r in calls_relationships if 
                             any(n.label == "Script" for n in graph.nodes if n.id == r.start_node_id) and
                             any(n.label == "Function" for n in graph.nodes if n.id == r.end_node_id)]
        
        print(f"\nRelationship counts:")
        print(f"  - Total CALLS: {len(calls_relationships)}")
        print(f"  - Script-to-Script: {len(script_to_script)}")
        print(f"  - Script-to-Function: {len(script_to_function)}")
        
        # Verify that cross-file relationships were created
        if len(script_to_script) > 0 or len(script_to_function) > 0:
            print("\n✅ Post-processing successful! Cross-file relationships detected.")
        else:
            print("\n❌ No cross-file relationships detected. Post-processing may not be working.")
        
        # Reset global registry for clean state
        MatlabExtractor.reset_global_registry()

if __name__ == "__main__":
    asyncio.run(test_post_processing()) 