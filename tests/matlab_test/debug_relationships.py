#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor_refactored import MatlabExtractor
from neo4j_graphrag.experimental.components.code_extractor.matlab.schema import SCHEMA
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks

async def debug_relationships():
    """Debug script to check all relationships and their properties."""
    
    # Create test files directory
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test file
    test_file = test_dir / "test_script.m"
    with open(test_file, 'w') as f:
        f.write("""
% Test script
shared_var = 10;
result = shared_var * 2;
disp(result);
        """)
    
    # Create extractor
    extractor = MatlabExtractor(enable_post_processing=True)
    
    # Create chunk
    with open(test_file, 'r') as f:
        content = f.read()
    
    chunk = TextChunk(
        text=content,
        index=0,
        metadata={"file_path": str(test_file)}
    )
    
    # Extract
    result = await extractor.run(TextChunks(chunks=[chunk]), schema=SCHEMA)
    
    # Check all nodes
    print(f"Total nodes: {len(result.graph.nodes)}")
    for i, node in enumerate(result.graph.nodes):
        print(f"\nNode {i+1}:")
        print(f"  ID: {node.id}")
        print(f"  Label: {node.label}")
        print(f"  Properties: {node.properties}")
    
    # Check all relationships
    print(f"\nTotal relationships: {len(result.graph.relationships)}")
    
    for i, rel in enumerate(result.graph.relationships):
        print(f"\nRelationship {i+1}:")
        print(f"  Type: {rel.type}")
        print(f"  Start node: {rel.start_node_id}")
        print(f"  End node: {rel.end_node_id}")
        print(f"  Has start_node_type: {hasattr(rel, 'start_node_type')}")
        print(f"  Has end_node_type: {hasattr(rel, 'end_node_type')}")
        if hasattr(rel, 'start_node_type'):
            print(f"  Start node type: {rel.start_node_type}")
        if hasattr(rel, 'end_node_type'):
            print(f"  End node type: {rel.end_node_type}")
        print(f"  Properties: {rel.properties}")
    
    # Also test without post-processing
    print("\n" + "="*50)
    print("Testing without post-processing:")
    
    extractor_no_post = MatlabExtractor(enable_post_processing=False)
    result_no_post = await extractor_no_post.run(TextChunks(chunks=[chunk]), schema=SCHEMA)
    
    print(f"Total nodes (no post): {len(result_no_post.graph.nodes)}")
    print(f"Total relationships (no post): {len(result_no_post.graph.relationships)}")
    
    for i, rel in enumerate(result_no_post.graph.relationships):
        print(f"\nRelationship {i+1} (no post):")
        print(f"  Type: {rel.type}")
        print(f"  Start node: {rel.start_node_id}")
        print(f"  End node: {rel.end_node_id}")
        print(f"  Has start_node_type: {hasattr(rel, 'start_node_type')}")
        print(f"  Has end_node_type: {hasattr(rel, 'end_node_type')}")
        if hasattr(rel, 'start_node_type'):
            print(f"  Start node type: {rel.start_node_type}")
        if hasattr(rel, 'end_node_type'):
            print(f"  End node type: {rel.end_node_type}")
        print(f"  Properties: {rel.properties}")

if __name__ == "__main__":
    asyncio.run(debug_relationships()) 