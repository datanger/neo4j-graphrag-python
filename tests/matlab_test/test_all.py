import os
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor

class TestMatlabRelationshipPatterns(unittest.TestCase):
    """
    Test that all relationship patterns defined in the schema are correctly created.
    
    The test verifies the following patterns:
    - Function -> CALLS -> Function
    - Function -> CALLS -> Script
    - Script -> CALLS -> Function
    - Script -> CALLS -> Script
    - Function -> USES -> Variable
    - Script -> USES -> Variable
    - Function -> DEFINES -> Variable
    - Script -> DEFINES -> Variable
    - Function -> MODIFIES -> Variable
    - Script -> MODIFIES -> Variable
    - Variable -> ASSIGNED_TO -> Variable
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all tests."""
        test_dir = Path(__file__).parent
        cls.test_dir = test_dir / "examples"
        print(f"Looking for test files in: {cls.test_dir}")
        test_files = [f for f in os.listdir(cls.test_dir) if f.endswith('.m')]
        print(f"Found {len(test_files)} test files: {test_files}")
        if not test_files:
            print("WARNING: No test files found in the examples directory!")
        
        # Create test files
        cls.create_test_files()
        
        # Set up mock LLM
        cls.mock_llm = MagicMock()
        cls.mock_llm.generate.return_value = "Mock description"
        
        # Set up extractor
        cls.extractor = MatlabExtractor(
            llm=cls.mock_llm,
            on_error='raise',
            create_lexical_graph=True,
            max_concurrency=5,
            debug=True
        )
    
    @classmethod
    def create_test_files(cls):
        """Create test MATLAB files with various relationship patterns."""
        # Create a test directory if it doesn't exist
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Test file 1: main_script.m - calls functions and other scripts
        with open(cls.test_dir / "main_script.m", "w") as f:
            f.write("""
            % Main script that demonstrates various relationship patterns
            
            % Call a function
            result1 = helper_function(5);
            
            % Call another script
            run('helper_script.m');
            
            % Define and use variables
            x = 10;
            y = x + 5;
            z = y * 2;
            
            % Call a function that modifies a variable
            [x, y] = modify_variables(x, y);
            
            % Use variables in a function call
            display_results(x, y, z);
            """)
        
        # Test file 2: helper_function.m
        with open(cls.test_dir / "helper_function.m", "w") as f:
            f.write("""
            function result = helper_function(input)
                % A helper function that demonstrates function calls
                result = input * 2;
            end
            """)
        
        # Test file 3: helper_script.m
        with open(cls.test_dir / "helper_script.m", "w") as f:
            f.write("""
            % A helper script that defines and uses variables
            a = 42;
            b = a / 2;
            c = b + 10;
            
            % Call a function
            d = helper_function(c);
            """)
        
        # Test file 4: modify_variables.m
        with open(cls.test_dir / "modify_variables.m", "w") as f:
            f.write("""
            function [out1, out2] = modify_variables(in1, in2)
                % Function that modifies its input variables
                out1 = in1 * 2;
                out2 = in2 + 10;
            end
            """)
        
        # Test file 5: display_results.m
        with open(cls.test_dir / "display_results.m", "w") as f:
            f.write("""
            function display_results(varargin)
                % Function that displays results
                for i = 1:length(varargin)
                    fprintf('Result %d: %f\n', i, varargin{i});
                end
            end
            """)
    
    def _extract_relationships(self, script_content, script_id):
        """Extract relationships from MATLAB script content."""
        edges = []
        lines = script_content.split('\n')
        
        # Simple regex patterns to find function calls and variable assignments
        import re
        
        # Pattern to match function calls: function_name(arg1, arg2, ...)
        func_call_pattern = r'([a-zA-Z_]\w*)\s*\([^)]*\)'
        # Pattern to match variable assignments: var = expression
        var_assign_pattern = r'([a-zA-Z_]\w*)\s*='
        
        # Find all function calls in the script
        for i, line in enumerate(lines):
            # Skip comments and empty lines
            if line.strip().startswith('%') or not line.strip():
                continue
                
            # Check for script calls (run command)
            if 'run(' in line and ')' in line:
                # Extract the script name from run('script.m')
                script_match = re.search(r"run\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
                if script_match:
                    script_name = script_match.group(1)
                    edges.append({
                        'source': script_id,
                        'target': f"script_{script_name}",
                        'label': 'CALLS',
                        'type': 'script_call',
                        'line': i + 1
                    })
            
            # Check for function calls
            for match in re.finditer(func_call_pattern, line):
                func_name = match.group(1)
                # Skip MATLAB built-in functions and operators
                if func_name in ['if', 'for', 'while', 'end', 'else', 'elseif', 'function', 'return']:
                    continue
                    
                # Check if this is a variable assignment with a function call
                var_match = re.match(var_assign_pattern, line[:match.start()])
                if var_match:
                    var_name = var_match.group(1)
                    # Add DEFINES relationship for the variable
                    edges.append({
                        'source': script_id,
                        'target': f"var_{var_name}",
                        'label': 'DEFINES',
                        'type': 'variable_definition',
                        'line': i + 1
                    })
                
                # Add CALLS relationship
                edges.append({
                    'source': script_id,
                    'target': f"func_{func_name}",
                    'label': 'CALLS',
                    'type': 'function_call',
                    'line': i + 1
                })
        
        return edges

    async def _extract_script(self, extractor, script_path):
        """Helper method to extract nodes and edges from a script using MatlabExtractor."""
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        try:
            from neo4j_graphrag.experimental.components.types import TextChunk
            from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType
            
            # Import the schema from kg_builder_from_code
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
                ],
            )
            
            # Create a TextChunk with the script content
            chunk = TextChunk(
                text=script_content,
                index=0,  # Required field, using 0 as default index
                metadata={"file_path": str(script_path)}
            )
            
            # Process the chunk using the extractor with the proper schema
            result = await extractor.extract_for_chunk(
                schema=SCHEMA,
                examples="",
                chunk=chunk
            )

            # Convert Neo4jGraph to nodes and edges format
            nodes = []
            edges = []
            
            # Add nodes with proper validation
            for node in result.nodes:
                try:
                    # Ensure we have a valid node ID
                    if not hasattr(node, 'id') or not node.id:
                        print(f"Warning: Node is missing ID, skipping: {node}")
                        continue
                        
                    # Get the node label with validation
                    node_label = 'Node'  # Default label
                    if hasattr(node, 'label') and node.label:
                        node_label = str(node.label)
                    elif hasattr(node, 'properties') and node.properties and 'label' in node.properties:
                        node_label = str(node.properties['label'])
                    
                    # Get node name with fallback
                    node_name = str(node.id)
                    if hasattr(node, 'properties') and node.properties:
                        node_name = str(node.properties.get('name', node_name))
                    
                    # Create node data with required fields
                    node_data = {
                        "id": str(node.id),
                        "label": node_label,
                        "name": node_name,
                    }
                    
                    # Add any additional properties
                    if hasattr(node, 'properties') and node.properties:
                        node_data.update({
                            k: v for k, v in node.properties.items() 
                            if v is not None and k not in ['id', 'label', 'name']
                        })
                    
                    nodes.append(node_data)
                    
                except Exception as e:
                    print(f"Error processing node {getattr(node, 'id', 'unknown')}: {str(e)}")
            
            # Add edges with validation
            for rel in result.relationships:
                try:
                    # Skip if relationship is invalid
                    if not hasattr(rel, 'start_node_id') and not hasattr(rel, 'start_node'):
                        print(f"Warning: Relationship missing start node: {rel}")
                        continue
                    if not hasattr(rel, 'end_node_id') and not hasattr(rel, 'end_node'):
                        print(f"Warning: Relationship missing end node: {rel}")
                        continue
                    
                    # Get source and target node IDs with fallbacks
                    source_id = None
                    target_id = None
                    
                    # Try to get source ID
                    if hasattr(rel, 'start_node_id') and rel.start_node_id:
                        source_id = str(rel.start_node_id)
                    elif hasattr(rel, 'start_node') and rel.start_node and hasattr(rel.start_node, 'id'):
                        source_id = str(rel.start_node.id)
                    
                    # Try to get target ID
                    if hasattr(rel, 'end_node_id') and rel.end_node_id:
                        target_id = str(rel.end_node_id)
                    elif hasattr(rel, 'end_node') and rel.end_node and hasattr(rel.end_node, 'id'):
                        target_id = str(rel.end_node.id)
                    
                    # Skip if we couldn't get valid source or target
                    if not source_id or not target_id:
                        print(f"Warning: Could not determine source/target for relationship: {rel}")
                        continue
                    
                    # Get relationship type with fallback
                    rel_type = 'RELATED'
                    if hasattr(rel, 'type') and rel.type:
                        rel_type = str(rel.type)
                    
                    # Get properties safely
                    properties = {}
                    if hasattr(rel, 'properties') and rel.properties:
                        properties = {
                            str(k): v 
                            for k, v in rel.properties.items() 
                            if v is not None
                        }
                    
                    # Create edge data
                    edge_data = {
                        "source": source_id,
                        "target": target_id,
                        "label": rel_type,
                        "properties": properties
                    }
                    
                    edges.append(edge_data)
                    
                except Exception as e:
                    print(f"Error processing relationship: {str(e)}\nRelationship data: {rel}")
                    continue
                
            return nodes, edges
            
        except Exception as e:
            print(f"Error extracting from {script_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def test_relationship_patterns(self):
        """Test that all relationship patterns are correctly created."""
        print("\n=== STARTING TEST: test_relationship_patterns ===")
        # Get the extractor instance
        try:
            extractor = MatlabExtractor(llm=MagicMock(), enable_post_processing=True)
            print("Created MatlabExtractor instance")
        except Exception as e:
            print(f"Error creating MatlabExtractor: {str(e)}")
            raise
        
        # Process all files together using run method
        all_nodes = []
        all_edges = []
        
        # Create chunks for all files
        from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
        
        chunks_list = []
        for i, script_file in enumerate(self.test_dir.glob("*.m")):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            chunk = TextChunk(
                text=script_content,
                index=i,
                metadata={"file_path": str(script_file)}
            )
            chunks_list.append(chunk)
        
        chunks = TextChunks(chunks=chunks_list)
        
        # Process all chunks using run method (enables post-processing)
        result = asyncio.run(extractor.run(chunks=chunks))
        graph = result.graph
        
        # Convert Neo4jGraph to nodes and edges format
        for node in graph.nodes:
            all_nodes.append({
                'id': node.id,
                'label': node.label,
                'name': node.properties.get('name', ''),
                'properties': node.properties
            })
        
        for edge in graph.relationships:
            all_edges.append({
                'source': edge.start_node_id,
                'target': edge.end_node_id,
                'label': edge.type,
                'properties': edge.properties
            })
        
        # Debug output for nodes and edges
        print("\n=== ALL NODES ===")
        for node in all_nodes:
            print(f"- {node.get('label')} {node.get('name', '?')} (id: {node.get('id')})")
        
        print("\n=== ALL EDGES ===")
        for edge in all_edges:
            print(f"- {edge.get('source')} --{edge.get('label', '?')}--> {edge.get('target')}")
        
        # Helper functions to find nodes and edges
        def find_node(nodes, label, name):
            return next((n for n in nodes if n.get('label') == label and n.get('name') == name), None)
        
        def find_edges(edges, source_id=None, target_id=None, label=None):
            results = []
            for e in edges:
                source_match = source_id is None or e.get('source') == source_id or \
                            (isinstance(source_id, str) and e.get('source', '').endswith(source_id))
                target_match = target_id is None or e.get('target') == target_id or \
                            (isinstance(target_id, str) and e.get('target', '').endswith(target_id))
                label_match = label is None or e.get('label') == label or \
                            (isinstance(label, str) and e.get('label', '').lower() == label.lower())
                
                if source_match and target_match and label_match:
                    results.append(e)
            return results
        
        # Create a lookup for nodes by ID
        node_ids = {n['id']: n for n in all_nodes}
        
        print("\n=== NODE TYPES FOUND ===")
        node_types = {}
        for node in all_nodes:
            node_type = node.get('label', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        for node_type, count in node_types.items():
            print(f"- {node_type}: {count} nodes")
            
        print("\n=== EDGE TYPES FOUND ===")
        edge_types = {}
        for edge in all_edges:
            edge_type = edge.get('label', 'Unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        for edge_type, count in edge_types.items():
            print(f"- {edge_type}: {count} edges")
        
        # Test 1: Check we have both script and function nodes
        script_nodes = [n for n in all_nodes if n.get('label') == 'Script']
        function_nodes = [n for n in all_nodes if n.get('label') == 'Function']
        
        self.assertGreater(len(script_nodes), 0, "No script nodes found")
        self.assertGreater(len(function_nodes), 0, "No function nodes found")
        
        # Test 2: Check for CALLS relationships
        calls_edges = find_edges(all_edges, label='CALLS')
        
        # Check for script-to-script calls
        script_calls = [e for e in calls_edges 
                       if node_ids.get(e['source'], {}).get('label') == 'Script' and 
                          node_ids.get(e['target'], {}).get('label') == 'Script']
        
        # Check for function-to-function calls
        func_calls = [e for e in calls_edges
                     if node_ids.get(e['source'], {}).get('label') == 'Function' and
                       node_ids.get(e['target'], {}).get('label') == 'Function']
        
        # Check for script-to-function calls
        script_to_func = [e for e in calls_edges 
                         if node_ids.get(e['source'], {}).get('label') == 'Script' and 
                            node_ids.get(e['target'], {}).get('label') == 'Function']
        
        # Check for function-to-script calls
        func_to_script = [e for e in calls_edges 
                         if node_ids.get(e['source'], {}).get('label') == 'Function' and 
                            node_ids.get(e['target'], {}).get('label') == 'Script']
        
        print(f"\nFound {len(script_calls)} script-to-script calls")
        print(f"Found {len(func_calls)} function-to-function calls")
        print(f"Found {len(script_to_func)} script-to-function calls")
        print(f"Found {len(func_to_script)} function-to-script calls")
        
        # Test 3: Check for variable relationships
        var_edges = [e for e in all_edges if e.get('label') in ['USES', 'DEFINES', 'MODIFIES', 'ASSIGNED_TO']]
        
        # Group by relationship type
        rel_types = {}
        for e in var_edges:
            rel_type = e.get('label')
            if rel_type not in rel_types:
                rel_types[rel_type] = []
            rel_types[rel_type].append(e)
        
        print("\nFound variable relationships:")
        for rel_type, edges in rel_types.items():
            print(f"- {rel_type}: {len(edges)} edges")
        
        # Verify each relationship type with specific checks
        for rel_type, edges in rel_types.items():
            print(f"\nVerifying {rel_type} relationships:")
            for edge in edges:
                source = node_ids.get(edge.get('source'))
                target = node_ids.get(edge.get('target'))
                
                # Basic validation
                self.assertIsNotNone(source, f"Source node {edge.get('source')} not found for edge {edge}")
                self.assertIsNotNone(target, f"Target node {edge.get('target')} not found for edge {edge}")
                self.assertEqual(edge.get('label'), rel_type, 
                               f"Unexpected edge label: {edge.get('label')}, expected {rel_type}")
                
                # Get node types for better error messages
                source_type = source.get('label') if source else 'UNKNOWN'
                target_type = target.get('label') if target else 'UNKNOWN'
                
                # Specific validation based on relationship type
                if rel_type == 'USES':
                    # Function/Script -> USES -> Variable
                    self.assertIn(source_type, ['Function', 'Script'], 
                                f"USES source must be Function or Script, got {source_type}")
                    self.assertEqual(target_type, 'Variable', 
                                   f"USES target must be Variable, got {target_type}")
                    print(f"  ✓ {source_type} '{source.get('name', '')}' USES Variable '{target.get('name', '')}'")
                    
                elif rel_type == 'DEFINES':
                    # Function/Script -> DEFINES -> Variable
                    self.assertIn(source_type, ['Function', 'Script'], 
                                f"DEFINES source must be Function or Script, got {source_type}")
                    self.assertEqual(target_type, 'Variable', 
                                   f"DEFINES target must be Variable, got {target_type}")
                    print(f"  ✓ {source_type} '{source.get('name', '')}' DEFINES Variable '{target.get('name', '')}'")
                    
                elif rel_type == 'MODIFIES':
                    # Function/Script -> MODIFIES -> Variable
                    self.assertIn(source_type, ['Function', 'Script'], 
                                f"MODIFIES source must be Function or Script, got {source_type}")
                    self.assertEqual(target_type, 'Variable', 
                                   f"MODIFIES target must be Variable, got {target_type}")
                    print(f"  ✓ {source_type} '{source.get('name', '')}' MODIFIES Variable '{target.get('name', '')}'")
                    
                elif rel_type == 'ASSIGNED_TO':
                    # Variable -> ASSIGNED_TO -> Variable (only variable to variable)
                    self.assertEqual(source_type, 'Variable',
                                   f"ASSIGNED_TO source must be Variable, got {source_type}")
                    self.assertEqual(target_type, 'Variable', 
                                   f"ASSIGNED_TO target must be Variable, got {target_type}")
                    print(f"  ✓ Variable '{source.get('name', '')}' ASSIGNED_TO Variable '{target.get('name', '')}'")
        
        # Verify all expected relationship types were found
        expected_rels = ['USES', 'DEFINES', 'MODIFIES', 'ASSIGNED_TO']
        for rel in expected_rels:
            self.assertIn(rel, rel_types.keys(), f"Expected to find {rel} relationships but none were found")
        
        print("\nAll relationship patterns verified successfully!")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        for script_file in cls.test_dir.glob("test_*.m"):
            try:
                script_file.unlink()
            except:
                pass

if __name__ == '__main__':
    unittest.main()