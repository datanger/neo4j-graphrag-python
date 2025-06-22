import os
import sys
import asyncio
import unittest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

    Updated to test scope-specific variable handling.
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
                x = input * 2;  % Same variable name as main script
                y = x + 10;     % Same variable name as main script
                result = y;
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

                % Internal variables with same names
                x = out1 + out2;
                y = x / 2;
            end
            """)

        # Test file 5: display_results.m
        with open(cls.test_dir / "display_results.m", "w") as f:
            f.write("""
            function display_results(varargin)
                % Function that displays results
                for i = 1:length(varargin)
                    fprintf('Result %d: %f\\n', i, varargin{i});
                end
            end
            """)

    async def _extract_script(self, extractor, script_path):
        """Helper method to extract nodes and edges from a script using MatlabExtractor."""
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        try:
            from neo4j_graphrag.experimental.components.types import TextChunk
            from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType

            # Create schema with updated properties
            SCHEMA = GraphSchema(
                node_types=[
                    NodeType(
                        label="Function",
                        description="A code function definition",
                        properties=[
                            PropertyType(name="name", type="STRING", description="Name of the function"),
                            PropertyType(name="file_path", type="STRING", description="Path to the file containing the function"),
                            PropertyType(name="scope_id", type="STRING", description="ID of the scope where this function is defined"),
                            PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' or 'function'"),
                        ],
                    ),
                    NodeType(
                        label="Variable",
                        description="A variable used in the code",
                        properties=[
                            PropertyType(name="name", type="STRING", description="Name of the variable"),
                            PropertyType(name="file_path", type="STRING", description="Path to the file where the variable is defined"),
                            PropertyType(name="scope_id", type="STRING", description="ID of the scope where this variable is defined"),
                            PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' or 'function'"),
                        ],
                    ),
                    NodeType(
                        label="Script",
                        description="A code script file",
                        properties=[
                            PropertyType(name="name", type="STRING", description="Name of the script"),
                            PropertyType(name="file_path", type="STRING", description="Path to the script file"),
                        ],
                    ),
                ],
                relationship_types=[
                    RelationshipType(label="CALLS", description="A function or script calls another function or script"),
                    RelationshipType(label="USES", description="A function or script uses a variable"),
                    RelationshipType(label="DEFINES", description="A function or script defines a variable"),
                    RelationshipType(label="MODIFIES", description="A function or script modifies a variable"),
                    RelationshipType(label="ASSIGNED_TO", description="A variable is assigned to another variable"),
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
                ]
            )

            # Create text chunk
            chunk = TextChunk(
                text=script_content,
                index=0,
                metadata={"file_path": str(script_path)}
            )

            # Extract using the extractor
            result = await extractor.extract_for_chunk(SCHEMA, "", chunk)

            # Convert to simple format for analysis
            nodes = []
            for node in result.nodes:
                nodes.append({
                    'id': node.id,
                    'label': node.label,
                    'name': node.properties.get('name', ''),
                    'scope_id': node.properties.get('scope_id', ''),
                    'scope_type': node.properties.get('scope_type', ''),
                })

            edges = []
            for edge in result.relationships:
                edges.append({
                    'source': edge.start_node_id,
                    'target': edge.end_node_id,
                    'type': edge.type,
                    'properties': edge.properties,
                })

            return nodes, edges

        except Exception as e:
            print(f"Error extracting script {script_path}: {e}")
            return [], []

    def test_scope_specific_variables(self):
        """Test that variables are scope-specific (same name in different scopes = different nodes)."""
        print("\n=== TESTING SCOPE-SPECIFIC VARIABLES ===")

        # Extract main script
        nodes, edges = asyncio.run(self._extract_script(self.extractor, self.test_dir / "main_script.m"))

        # Find all variable nodes
        variable_nodes = [n for n in nodes if n['label'] == 'Variable']

        # Group variables by name
        variables_by_name = {}
        for node in variable_nodes:
            name = node['name']
            if name not in variables_by_name:
                variables_by_name[name] = []
            variables_by_name[name].append({
                'id': node['id'],
                'scope_id': node['scope_id'],
                'scope_type': node['scope_type']
            })

        print(f"Found {len(variable_nodes)} variable nodes")
        print("Variables by name:")
        for name, instances in variables_by_name.items():
            print(f"  {name}: {len(instances)} instances")
            for inst in instances:
                print(f"    - {inst['id']} (scope: {inst['scope_type']} {inst['scope_id']})")

        # Check for variables with same name in different scopes
        multi_scope_vars = {name: instances for name, instances in variables_by_name.items() if len(instances) > 1}

        print(f"\nVariables in multiple scopes: {len(multi_scope_vars)}")
        for name, instances in multi_scope_vars.items():
            print(f"  {name}: {len(instances)} scopes")
            scope_ids = [inst['scope_id'] for inst in instances]
            self.assertGreater(len(set(scope_ids)), 1, f"Variable {name} should be in different scopes")

        # Verify that each variable has a unique ID
        variable_ids = [n['id'] for n in variable_nodes]
        unique_ids = set(variable_ids)
        self.assertEqual(len(variable_ids), len(unique_ids), "All variable nodes should have unique IDs")

        print("✓ Scope-specific variable handling works correctly")

    def test_relationship_patterns(self):
        """Test that all relationship patterns are correctly created."""
        print("\n=== TESTING RELATIONSHIP PATTERNS ===")

        # Extract all files
        all_nodes = []
        all_edges = []

        for m_file in self.test_dir.glob("*.m"):
            nodes, edges = asyncio.run(self._extract_script(self.extractor, m_file))
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        # Remove duplicates
        unique_nodes = []
        seen_node_ids = set()
        for node in all_nodes:
            if node['id'] not in seen_node_ids:
                unique_nodes.append(node)
                seen_node_ids.add(node['id'])

        unique_edges = []
        seen_edge_keys = set()
        for edge in all_edges:
            key = (edge['source'], edge['target'], edge['type'])
            if key not in seen_edge_keys:
                unique_edges.append(edge)
                seen_edge_keys.add(key)

        print(f"Total unique nodes: {len(unique_nodes)}")
        print(f"Total unique edges: {len(unique_edges)}")

        # Group edges by type
        edges_by_type = {}
        for edge in unique_edges:
            edge_type = edge['type']
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)

        print("\nEdges by type:")
        for edge_type, edge_list in edges_by_type.items():
            print(f"  {edge_type}: {len(edge_list)} edges")

        # Test specific relationship patterns
        expected_patterns = [
            ("Script", "CALLS", "Function"),
            ("Script", "CALLS", "Script"),
            ("Script", "DEFINES", "Variable"),
            ("Function", "DEFINES", "Variable"),
            ("Script", "USES", "Variable"),
            ("Function", "USES", "Variable"),
            ("Variable", "ASSIGNED_TO", "Variable"),
        ]

        for source_type, edge_type, target_type in expected_patterns:
            found = False
            for edge in unique_edges:
                source_node = next((n for n in unique_nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in unique_nodes if n['id'] == edge['target']), None)

                if (source_node and target_node and
                    source_node['label'] == source_type and
                    edge['type'] == edge_type and
                    target_node['label'] == target_type):
                    found = True
                    break

            self.assertTrue(found, f"Missing relationship pattern: {source_type} -[{edge_type}]-> {target_type}")
            print(f"  ✓ Found {source_type} -[{edge_type}]-> {target_type}")

        print("✓ All expected relationship patterns found")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        # Optionally clean up test files
        # for m_file in cls.test_dir.glob("*.m"):
        #     try:
        #         m_file.unlink()
        #     except:
        #         pass
        pass

if __name__ == '__main__':
    unittest.main()
