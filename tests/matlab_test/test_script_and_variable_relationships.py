import os
import sys
import asyncio
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from unittest.mock import MagicMock

# Import the extractor from the main package
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor
from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType
from neo4j_graphrag.experimental.components.types import TextChunk

# Create a mock LLM for testing
class MockLLM:
    def generate(self, *args, **kwargs):
        return "Mock description"

# Create a mock LLMInterface for testing
class MockLLMInterface:
    def __init__(self):
        pass
    
    def generate(self, *args, **kwargs):
        return "Mock description"

class TestScriptAndVariableRelationships(unittest.TestCase):
    @classmethod
    def get_edge_source(cls, edge):
        """Get the source node ID from an edge, handling different field names."""
        return edge.get('source') or edge.get('start_node_id')
    
    @classmethod
    def get_edge_target(cls, edge):
        """Get the target node ID from an edge, handling different field names."""
        return edge.get('target') or edge.get('end_node_id')
    
    @classmethod
    def get_edge_label(cls, edge):
        """Get the edge label/type, handling different field names."""
        return edge.get('label') or edge.get('type')
    
    @classmethod
    def print_edge_info(cls, edge, all_nodes, idx):
        """Print detailed information about an edge for debugging."""
        try:
            src = cls.get_edge_source(edge)
            tgt = cls.get_edge_target(edge)
            lbl = cls.get_edge_label(edge)
            src_node = next((n for n in all_nodes if n.get('id') == src), {'name': '?', 'label': '?'})
            tgt_node = next((n for n in all_nodes if n.get('id') == tgt), {'name': '?', 'label': '?'})
            print(f"{idx}. {src_node.get('label', '?')} {src_node.get('name', '?')} --{lbl}--> {tgt_node.get('label', '?')} {tgt_node.get('name', '?')}")
            if 'properties' in edge:
                print(f"   Properties: {edge['properties']}")
        except Exception as e:
            print(f"{idx}. Error processing edge: {edge}")
            print(f"Error: {e}")
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all tests."""
        # Get the directory where this test file is located
        test_dir = Path(__file__).parent
        
        # Set the paths to the test files in the examples directory
        cls.test_script = test_dir / 'examples' / 'test_script_and_variable_relationships.m'
        cls.external_script = test_dir / 'examples' / 'external_script.m'
        
        # Print the paths for debugging
        print(f"Test script path: {cls.test_script}")
        print(f"External script path: {cls.external_script}")
        print(f"Test script exists: {cls.test_script.exists()}")
        print(f"External script exists: {cls.external_script.exists()}")

    def setUp(self):
        """Set up before each test."""
        self.mock_llm = MockLLMInterface()
        self.extractor = MatlabExtractor(
            llm=self.mock_llm,
            on_error='ignore',
            create_lexical_graph=True,
            max_concurrency=5,
            debug=True
        )
    
    def get_edges_by_type(self, edges: List[Dict], edge_type: str) -> List[Dict]:
        """Helper to get edges of a specific type."""
        return [e for e in edges if e.get('label') == edge_type]
    
    async def _extract_script(self, extractor, script_path):
        """Helper method to extract nodes and edges from a script."""
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\nProcessing script: {script_path}")
        
        # Create a text chunk with metadata and required fields
        chunk = TextChunk(
            text=content,
            index=0,  # Required field, using 0 as the default index
            metadata={"file_path": str(script_path)}
        )
        
        # Create a schema with required node types for MATLAB code extraction
        schema = GraphSchema(
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
                    description="A function or script defines a variable",
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
                ("Function", "MODIFIES", "Variable"),
                ("Script", "MODIFIES", "Variable"),
                ("Variable", "ASSIGNED_TO", "Variable"),
            ]
        )
        
        # Extract nodes and edges
        graph = await extractor.extract_for_chunk(schema, "", chunk)
        
        # Convert graph to nodes and edges lists
        nodes = [dict(node) for node in graph.nodes]
        edges = [dict(rel) for rel in graph.relationships]
        
        return nodes, edges
    
    def test_script_to_script_calls(self):
        """Test that script-to-script calls are correctly detected."""
        # Create a new extractor instance for the external script
        extractor = MatlabExtractor(
            llm=self.mock_llm,
            on_error='ignore',
            create_lexical_graph=True,
            max_concurrency=5,
            debug=True
        )
        
        # Process the external script
        nodes, edges = asyncio.run(self._extract_script(extractor, self.external_script))
        
        # Debug output for external script
        print("\n=== EXTERNAL SCRIPT NODES ===")
        for node in nodes:
            print(f"- {node.get('label')} {node.get('name', '?')} (id: {node.get('id')})")
        
        # Process the test script
        test_nodes, test_edges = asyncio.run(self._extract_script(self.extractor, self.test_script))
        
        # Combine nodes and edges from both extractions
        all_nodes = nodes + test_nodes
        all_edges = edges + test_edges
        
        # Debug output
        print("\n=== ALL NODES ===")
        for idx, node in enumerate(all_nodes):
            print(f"{idx}. {node.get('label')} {node.get('name', '?')} (id: {node.get('id')})")
        
        print("\n=== ALL EDGES ===")
        for idx, edge in enumerate(all_edges):
            # Skip edges that don't have source or target
            if 'source' not in edge or 'target' not in edge:
                print(f"{idx}. Invalid edge (missing source or target): {edge}")
                continue
                
            try:
                source = next((n for n in all_nodes if n.get('id') == edge['source']), {'name': '?', 'label': '?'})
                target = next((n for n in all_nodes if n.get('id') == edge['target']), {'name': '?', 'label': '?'})
                print(f"{idx}. {source.get('label', '?')} {source.get('name', '?')} --{edge.get('label', '?')}--> {target.get('label', '?')} {target.get('name', '?')}")
            except Exception as e:
                print(f"{idx}. Error processing edge: {edge}")
                print(f"Error: {e}")
        
        # Verify nodes were created for both scripts and the function
        script_nodes = [n for n in all_nodes if n.get('label') == 'Script']
        function_nodes = [n for n in all_nodes if n.get('label') == 'Function']
        
        print("\n=== SCRIPT NODES ===")
        for node in script_nodes:
            print(f"- {node.get('name')} (id: {node.get('id')})")
            
        print("\n=== FUNCTION NODES ===")
        for node in function_nodes:
            print(f"- {node.get('name')} (id: {node.get('id')})")
        
        # Verify script and function nodes were created
        self.assertGreater(len(script_nodes), 0, "No script nodes found")
        self.assertGreater(len(function_nodes), 0, "No function nodes found")
        
        # Find all CALLS edges (case-insensitive)
        call_edges = [e for e in all_edges 
                     if self.get_edge_label(e) and self.get_edge_label(e).upper() == 'CALLS'
                     and self.get_edge_source(e) and self.get_edge_target(e)]
        
        # Also look for direct script calls using run()
        run_edges = [e for e in all_edges 
                    if self.get_edge_label(e) and 'run' in self.get_edge_label(e).lower()
                    and self.get_edge_source(e) and self.get_edge_target(e)]
        
        # Combine all potential call edges
        all_call_edges = call_edges + run_edges
        
        # Debug output for call edges
        print("\n=== CALL EDGES ===")
        for idx, edge in enumerate(all_call_edges):
            self.print_edge_info(edge, all_nodes, idx)
        
        # Verify call edges exist
        self.assertGreater(len(all_call_edges), 0, 
                         "No script call edges found. Expected CALLS or run() edges between scripts.")
        
        # Check for direct script-to-script calls (run command or script_call)
        script_call_edges = [e for e in all_call_edges 
                          if 'properties' in e and 
                          (e['properties'].get('type') in ['script_call', 'run_command'] or
                           'run' in (self.get_edge_label(e) or '').lower())]
        
        # Find specific nodes
        test_script_node = next((n for n in all_nodes 
                              if n.get('name') == 'test_script_and_variable_relationships'), None)
        external_script_node = next((n for n in all_nodes 
                                  if n.get('name') == 'external_script'), None)
        
        # Verify nodes exist
        self.assertIsNotNone(test_script_node, "Test script node not found")
        self.assertIsNotNone(external_script_node, "External script node not found")
        
        # Verify script calls external script (either directly or via run command)
        call_found = False
        call_details = []
        
        for edge in all_call_edges:
            src = self.get_edge_source(edge)
            tgt = self.get_edge_target(edge)
            if src == test_script_node['id'] and tgt == external_script_node['id']:
                call_found = True
                call_details.append(f"Found call from {test_script_node['name']} to {external_script_node['name']} with type: {self.get_edge_label(edge)}")
        
        self.assertTrue(call_found, 
                      f"No call edge found from {test_script_node['name']} to {external_script_node['name']}. "
                      f"Searched through {len(all_call_edges)} call edges.")
        
        # Print details of the found calls
        for detail in call_details:
            print(detail)
        
        # Verify function calls external script
        call_function = next((n for n in function_nodes 
                           if n.get('name') == 'call_external_script'), None)
        self.assertIsNotNone(call_function, "call_external_script function not found")
        
        func_call_found = False
        for edge in all_call_edges:
            if (self.get_edge_source(edge) == call_function['id'] and 
                self.get_edge_target(edge) == external_script_node['id']):
                func_call_found = True
                print(f"Found call from function {call_function['name']} to {external_script_node['name']}")
                break
                
        self.assertTrue(func_call_found,
                     f"No call edge found from call_external_script to {external_script_node['name']}. "
                     f"Make sure the function contains a call to the external script.")
    
    def test_variable_assignments(self):
        """Test that variable assignments are correctly tracked."""
        # Process the external script
        nodes, edges = asyncio.run(self._extract_script(self.extractor, self.external_script))
        
        # Process the test script
        test_nodes, test_edges = asyncio.run(self._extract_script(self.extractor, self.test_script))
        
        # Combine nodes and edges from both extractions
        all_nodes = nodes + test_nodes
        all_edges = edges + test_edges
        
        # Debug output
        print("\n=== ALL NODES ===")
        for idx, node in enumerate(all_nodes):
            print(f"{idx}. {node.get('label')} {node.get('name', '?')} (id: {node.get('id')})")
            print(f"   Properties: {', '.join(f'{k}={v}' for k, v in node.items() if k not in ['id', 'label', 'name'])}")
        
        print("\n=== ALL EDGES ===")
        for idx, edge in enumerate(all_edges):
            if 'source' not in edge or 'target' not in edge:
                print(f"{idx}. Invalid edge (missing source or target): {edge}")
                continue
                
            try:
                source = next((n for n in all_nodes if n.get('id') == edge['source']), {'name': '?', 'label': '?'})
                target = next((n for n in all_nodes if n.get('id') == edge['target']), {'name': '?', 'label': '?'})
                print(f"{idx}. {source.get('label', '?')} {source.get('name', '?')} --{edge.get('label', '?')}--> {target.get('label', '?')} {target.get('name', '?')}")
                if 'properties' in edge:
                    print(f"   Properties: {edge['properties']}")
            except Exception as e:
                print(f"{idx}. Error processing edge: {edge}")
                print(f"Error: {e}")
        
        # Find variable nodes and DEFINES relationships
        variable_nodes = [n for n in all_nodes if n.get('label') in ['Variable', 'variable']]
        defines_edges = []
        
        # Look for DEFINES edges with different case variations
        for e in all_edges:
            edge_label = self.get_edge_label(e) or ''
            if (edge_label.upper() == 'DEFINES' or 
                'define' in edge_label.lower() or
                (isinstance(e.get('properties'), dict) and 
                 e['properties'].get('type') in ['variable_definition', 'defines'])):
                if self.get_edge_source(e) and self.get_edge_target(e):
                    defines_edges.append(e)
        
        # Debug output for DEFINES edges
        print("\n=== DEFINES EDGES ===")
        for idx, edge in enumerate(defines_edges):
            self.print_edge_info(edge, all_nodes, idx)
        
        # Verify we have variables and definitions
        self.assertGreater(len(variable_nodes), 0, "No variable nodes found")
        self.assertGreater(len(defines_edges), 0, "No DEFINES relationships found")
        
        # Find specific variable nodes
        var_a = next((n for n in variable_nodes if n.get('name') == 'a'), None)
        var_b = next((n for n in variable_nodes if n.get('name') == 'b'), None)
        
        # Verify variable nodes exist
        self.assertIsNotNone(var_a, "Variable 'a' not found")
        self.assertIsNotNone(var_b, "Variable 'b' not found")
        
        # Find the function node
        func_node = next((n for n in all_nodes 
                         if n.get('label') == 'Function' 
                         and n.get('name') == 'call_external_script'), None)
        self.assertIsNotNone(func_node, "Function 'call_external_script' not found")
        
        # Verify DEFINES relationships from function to variables
        # Verify DEFINES relationships from function to variables
        def check_defines_edge(source_id, target_id, var_name):
            for e in defines_edges:
                if (self.get_edge_source(e) == source_id and 
                    self.get_edge_target(e) == target_id):
                    return True
            return False
        
        # Check for variable 'a' definition
        var_a_defined = check_defines_edge(func_node['id'], var_a['id'], 'a')
        self.assertTrue(var_a_defined, 
                      f"No DEFINES edge found from {func_node['name']} to variable 'a'. "
                      f"Make sure the variable is defined in the function.")
        
        # Check for variable 'b' definition
        var_b_defined = check_defines_edge(func_node['id'], var_b['id'], 'b')
        self.assertTrue(var_b_defined, 
                      f"No DEFINES edge found from {func_node['name']} to variable 'b'. "
                      f"Make sure the variable is defined in the function.")
        
        for var in variable_nodes:
            print(f"- {var.get('name')} (id: {var.get('id')})")
        
        print("\n=== DEFINES RELATIONSHIPS ===")
        for edge in defines_edges:
            try:
                source = next((n for n in all_nodes if n.get('id') == edge.get('source')), {'name': '?', 'label': '?'})
                target = next((n for n in all_nodes if n.get('id') == edge.get('target')), {'name': '?', 'label': '?'})
                print(f"- {source.get('label', '?')} {source.get('name', '?')} defines {target.get('name', '?')}")
            except Exception as e:
                print(f"Error processing defines edge: {edge}")
                print(f"Error: {e}")
        
        # For now, just verify we have some variables and defines relationships
        self.assertGreater(len(variable_nodes), 0, "No variable nodes were created")
        
        # Temporarily disable this assertion to see other test results
        # self.assertGreater(len(defines_edges), 0, "No DEFINES relationships were created")

if __name__ == '__main__':
    unittest.main()
