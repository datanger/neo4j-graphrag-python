#!/usr/bin/env python3
"""
Debug script for testing MATLAB function parameter extraction.
This script helps debug issues with function parameter handling.
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor
from neo4j_graphrag.experimental.components.code_extractor.matlab.schema import SCHEMA
from neo4j_graphrag.experimental.components.types import TextChunk

def test_function_parameter_extraction():
    """Test function parameter extraction with various function signatures."""
    print("Testing function parameter extraction...")
    
    # Test cases with different function signatures
    test_cases = [
        {
            "name": "Simple function",
            "code": """
function result = simple_function(input)
    result = input * 2;
end
""",
            "expected_params": ["input"],
            "expected_returns": ["result"]
        },
        {
            "name": "Multiple parameters",
            "code": """
function [out1, out2] = multi_param_function(in1, in2, in3)
    out1 = in1 + in2;
    out2 = in2 * in3;
end
""",
            "expected_params": ["in1", "in2", "in3"],
            "expected_returns": ["out1", "out2"]
        },
        {
            "name": "No parameters",
            "code": """
function result = no_param_function()
    result = 42;
end
""",
            "expected_params": [],
            "expected_returns": ["result"]
        },
        {
            "name": "Complex function",
            "code": """
function [output1, output2, output3] = complex_function(input1, input2, input3, input4)
    % This is a complex function with many parameters
    output1 = input1 + input2;
    output2 = input2 * input3;
    output3 = input3 / input4;
    
    % Local variables
    temp1 = output1 + output2;
    temp2 = temp1 * output3;
end
""",
            "expected_params": ["input1", "input2", "input3", "input4"],
            "expected_returns": ["output1", "output2", "output3"]
        }
    ]
    
    # Create extractor
    extractor = MatlabExtractor(use_llm=False)
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        # Create chunk
        chunk = TextChunk(
            text=test_case['code'],
            index=0,
            metadata={"file_path": f"test_{test_case['name'].lower().replace(' ', '_')}.m"}
        )
        
        # Extract using predefined SCHEMA
        result = asyncio.run(extractor.extract_for_chunk(SCHEMA, "", chunk))
        
        if result and hasattr(result, 'nodes'):
            nodes = result.nodes
        else:
            nodes = []
            
        if result and hasattr(result, 'relationships'):
            relationships = result.relationships
        else:
            relationships = []
        
        # Find function node
        function_nodes = [node for node in nodes if node.label == 'Function']
        
        if function_nodes:
            func_node = function_nodes[0]
            func_id = func_node.id
            properties = func_node.properties
            
            print(f"Function ID: {func_id}")
            print(f"Function name: {properties.get('name', 'N/A')}")
            print(f"Parameters: {properties.get('input_parameters', 'N/A')}")
            print(f"Returns: {properties.get('output_variables', 'N/A')}")
            
            # Check parameters
            actual_params = properties.get('input_parameters', [])
            expected_params = test_case['expected_params']
            
            if actual_params == expected_params:
                print("✓ Parameters match expected")
            else:
                print(f"✗ Parameter mismatch: expected {expected_params}, got {actual_params}")
            
            # Check returns
            actual_returns = properties.get('output_variables', [])
            expected_returns = test_case['expected_returns']
            
            if actual_returns == expected_returns:
                print("✓ Returns match expected")
            else:
                print(f"✗ Return mismatch: expected {expected_returns}, got {actual_returns}")
            
            # Check variable nodes for parameters
            var_nodes = [node for node in nodes if node.label == 'Parameter']
            param_vars = [var for var in var_nodes if var.properties.get('function_name') == properties.get('name')]
            
            print(f"Variables in function scope: {len(param_vars)}")
            for var in param_vars:
                var_props = var.properties
                print(f"  - {var.id}: {var_props.get('name', 'N/A')} (type: {var_props.get('type', 'N/A')})")
            
            # Check that parameters are defined as variables
            for param in expected_params:
                param_found = any(var.properties.get('name') == param for var in param_vars)
                if param_found:
                    print(f"✓ Parameter '{param}' found as variable")
                else:
                    print(f"✗ Parameter '{param}' not found as variable")
            
        else:
            print("✗ No function node found")
        
        # Check relationships
        print(f"Total relationships: {len(relationships)}")
        for rel in relationships:
            rel_type = rel.type
            source_id = rel.start_node_id
            target_id = rel.end_node_id
            print(f"  - {rel_type}: {source_id} -> {target_id}")

def test_variable_id_format():
    """Test that variables follow the required ID format."""
    print("\nTesting variable ID format...")
    
    test_code = """
function [out1, out2] = test_function(in1, in2)
    % Function with parameters and local variables
    out1 = in1 * 2;
    out2 = in2 + 10;
    
    % Local variable with same name as parameter
    temp = out1 + out2;
end
"""
    
    # Create extractor
    extractor = MatlabExtractor(use_llm=False)
    
    # Create chunk
    chunk = TextChunk(
        text=test_code,
        index=0,
        metadata={"file_path": "test_variable_format.m"}
    )
    
    # Extract using predefined SCHEMA
    result = asyncio.run(extractor.extract_for_chunk(SCHEMA, "", chunk))
    
    if result and hasattr(result, 'nodes'):
        nodes = result.nodes
    else:
        nodes = []
    
    # Find variable nodes
    var_nodes = [node for node in nodes if node.label == 'Variable']
    
    print(f"Found {len(var_nodes)} variable nodes:")
    
    for var in var_nodes:
        var_id = var.id
        properties = var.properties
        name = properties.get('name', '')
        scope_id = properties.get('scope_id', '')
        scope_type = properties.get('scope_type', '')
        
        # Check ID format
        expected_id = f"var_{name}_{scope_id}"
        if var_id == expected_id:
            print(f"✓ {var_id} (scope: {scope_type})")
        else:
            print(f"✗ {var_id} != {expected_id} (scope: {scope_type})")
        
        # Check required properties
        if 'scope_id' in properties and 'scope_type' in properties:
            print(f"  ✓ Has required properties")
        else:
            print(f"  ✗ Missing required properties")

def main():
    """Run all debug tests."""
    print("Debugging MATLAB function parameter extraction...\n")
    
    try:
        test_function_parameter_extraction()
        test_variable_id_format()
        print("\n✓ All debug tests completed")
        return True
    except Exception as e:
        print(f"\n✗ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
