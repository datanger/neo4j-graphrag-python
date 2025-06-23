 #!/usr/bin/env python3
"""
Test script for refactored MATLAB extractor modules.
This script tests the individual modules created during refactoring.
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_registry_module():
    """Test the registry module."""
    print("Testing registry module...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.registry import (
            GlobalMatlabRegistry, get_global_registry, reset_global_registry
        )
        
        # Test registry creation
        registry = GlobalMatlabRegistry()
        assert registry is not None
        print("✓ Registry creation successful")
        
        # Test global registry
        global_reg = get_global_registry()
        assert global_reg is not None
        print("✓ Global registry access successful")
        
        # Test reset
        reset_global_registry()
        print("✓ Registry reset successful")
        
        return True
    except Exception as e:
        print(f"✗ Registry module test failed: {e}")
        return False

def test_parser_module():
    """Test the parser module."""
    print("\nTesting parser module...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.parser import MatlabCodeParser
        
        parser = MatlabCodeParser()
        assert parser is not None
        print("✓ Parser creation successful")
        
        # Test parsing simple code
        test_code = """
function result = test_function(input)
    result = input * 2;
    x = result + 5;
end
"""
        
        nodes, edges = parser.parse_matlab_code("test.m", test_code)
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        print("✓ Parser parsing successful")
        
        return True
    except Exception as e:
        print(f"✗ Parser module test failed: {e}")
        return False

def test_post_processor_module():
    """Test the post processor module."""
    print("\nTesting post processor module...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.post_processor import MatlabPostProcessor
        from neo4j_graphrag.experimental.components.types import Neo4jGraph
        
        processor = MatlabPostProcessor()
        assert processor is not None
        print("✓ Post processor creation successful")
        
        # Test with empty graph
        empty_graph = Neo4jGraph(nodes=[], relationships=[])
        result = processor.post_process_cross_file_relationships(empty_graph)
        assert result is not None
        print("✓ Post processor processing successful")
        
        return True
    except Exception as e:
        print(f"✗ Post processor module test failed: {e}")
        return False

def test_utils_module():
    """Test the utils module."""
    print("\nTesting utils module...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.utils import (
            ensure_neo4j_compatible, extract_variables_from_code, 
            extract_function_calls_from_code, is_matlab_file
        )
        
        # Test utility functions
        result = ensure_neo4j_compatible("test")
        assert result == "test"
        print("✓ Neo4j compatibility function works")
        
        result = is_matlab_file("test.m")
        assert result == True
        print("✓ MATLAB file detection works")
        
        code = "x = 1; y = x + 2;"
        vars = extract_variables_from_code(code)
        assert "x" in vars
        assert "y" in vars
        print("✓ Variable extraction works")
        
        return True
    except Exception as e:
        print(f"✗ Utils module test failed: {e}")
        return False

def test_refactored_extractor():
    """Test the refactored extractor."""
    print("\nTesting refactored extractor...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor_refactored import MatlabExtractor
        
        extractor = MatlabExtractor(use_llm=False)
        assert extractor is not None
        print("✓ Refactored extractor creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Refactored extractor test failed: {e}")
        return False

async def test_integration():
    """Test integration of all modules."""
    print("\nTesting integration...")
    
    try:
        from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor_refactored import MatlabExtractor
        from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
        from neo4j_graphrag.experimental.components.schema import GraphSchema
        
        # Create test code
        test_code = """
function result = test_function(input)
    result = input * 2;
    x = result + 5;
end
"""
        
        # Create extractor
        extractor = MatlabExtractor(use_llm=False)
        
        # Create chunk
        chunk = TextChunk(
            text=test_code,
            index=0,
            metadata={"file_path": "test.m"}
        )
        
        # Create schema
        schema = GraphSchema()
        
        # Extract
        result = await extractor.extract_for_chunk(schema, "", chunk)
        assert result is not None
        print("✓ Integration test successful")
        
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing refactored MATLAB extractor modules...\n")
    
    tests = [
        test_registry_module,
        test_parser_module,
        test_post_processor_module,
        test_utils_module,
        test_refactored_extractor,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # Run async integration test
    try:
        if asyncio.run(test_integration()):
            passed += 1
        total += 1
    except Exception as e:
        print(f"✗ Async integration test failed: {e}")
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring is successful!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)