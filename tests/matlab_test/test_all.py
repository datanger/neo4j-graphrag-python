import os
import sys
import asyncio
import unittest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor
from neo4j_graphrag.experimental.components.code_extractor.matlab.schema import SCHEMA

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

    Updated to test scope-specific variable handling and requirements compliance.
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
            max_concurrency=5
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

            chunk = TextChunk(
                text=script_content,
                index=0,
                metadata={"file_path": str(script_path)}
            )

            # Use predefined SCHEMA
            result = await extractor.extract_for_chunk(SCHEMA, "", chunk)
            
            if result and hasattr(result, 'nodes'):
                nodes = result.nodes
            else:
                nodes = []
                
            if result and hasattr(result, 'relationships'):
                relationships = result.relationships
            else:
                relationships = []
                
            return nodes, relationships
        except Exception as e:
            print(f"Error extracting from {script_path}: {e}")
            return [], []

    def test_scope_specific_variables(self):
        """Test that variables are scope-specific with proper ID format."""
        print("\nTesting scope-specific variables...")
        
        async def run_test():
            # Extract from main script
            main_nodes, _ = await self._extract_script(self.extractor, self.test_dir / "main_script.m")
            
            # Extract from helper function
            func_nodes, _ = await self._extract_script(self.extractor, self.test_dir / "helper_function.m")
            
            # Find variable nodes
            main_vars = [node for node in main_nodes if node.get('type') == 'Variable']
            func_vars = [node for node in func_nodes if node.get('type') == 'Variable']
            
            # Check main script variables
            for var in main_vars:
                var_id = var.get('id', '')
                properties = var.get('properties', {})
                name = properties.get('name', '')
                scope_id = properties.get('scope_id', '')
                scope_type = properties.get('scope_type', '')
                
                # Check ID format
                expected_id = f"var_{name}_{scope_id}"
                assert var_id == expected_id, f"Variable ID format mismatch: {var_id} != {expected_id}"
                
                # Check scope properties
                assert scope_type == 'script', f"Main script variable should have scope_type 'script': {scope_type}"
                assert 'main_script' in scope_id, f"Main script variable should have main_script in scope_id: {scope_id}"
            
            # Check function variables
            for var in func_vars:
                var_id = var.get('id', '')
                properties = var.get('properties', {})
                name = properties.get('name', '')
                scope_id = properties.get('scope_id', '')
                scope_type = properties.get('scope_type', '')
                
                # Check ID format
                expected_id = f"var_{name}_{scope_id}"
                assert var_id == expected_id, f"Variable ID format mismatch: {var_id} != {expected_id}"
                
                # Check scope properties
                assert scope_type == 'function', f"Function variable should have scope_type 'function': {scope_type}"
                assert 'helper_function' in scope_id, f"Function variable should have helper_function in scope_id: {scope_id}"
            
            # Check that variables with same names have different IDs
            main_var_names = [var.get('properties', {}).get('name', '') for var in main_vars]
            func_var_names = [var.get('properties', {}).get('name', '') for var in func_vars]
            
            common_names = set(main_var_names) & set(func_var_names)
            for name in common_names:
                main_var = next(v for v in main_vars if v.get('properties', {}).get('name') == name)
                func_var = next(v for v in func_vars if v.get('properties', {}).get('name') == name)
                
                assert main_var.get('id') != func_var.get('id'), f"Variables with same name should have different IDs: {name}"
            
            print(f"✓ Scope-specific variables working correctly ({len(main_vars)} main script vars, {len(func_vars)} function vars)")
            return True
        
        return asyncio.run(run_test())

    def test_relationship_patterns(self):
        """Test that all required relationship patterns are created."""
        print("\nTesting relationship patterns...")
        
        async def run_test():
            # Extract from all files
            all_nodes = []
            all_relationships = []
            
            test_files = ["main_script.m", "helper_function.m", "modify_variables.m", "helper_script.m", "display_results.m"]
            
            for file_name in test_files:
                file_path = self.test_dir / file_name
                if file_path.exists():
                    nodes, relationships = await self._extract_script(self.extractor, file_path)
                    all_nodes.extend(nodes)
                    all_relationships.extend(relationships)
            
            # Check relationship types
            rel_types = set(rel.get('type') for rel in all_relationships)
            expected_types = {"CALLS", "USES", "DEFINES", "MODIFIES", "ASSIGNED_TO"}
            
            for rel_type in expected_types:
                assert rel_type in rel_types, f"Missing relationship type: {rel_type}"
            
            # Check specific patterns
            patterns_found = set()
            for rel in all_relationships:
                source_type = rel.get('source_type', '')
                rel_type = rel.get('type', '')
                target_type = rel.get('target_type', '')
                pattern = (source_type, rel_type, target_type)
                patterns_found.add(pattern)
            
            # Verify all required patterns are present
            required_patterns = [
                ("Script", "CALLS", "Function"),
                ("Script", "CALLS", "Script"),
                ("Function", "CALLS", "Function"),
                ("Script", "USES", "Variable"),
                ("Function", "USES", "Variable"),
                ("Script", "DEFINES", "Variable"),
                ("Function", "DEFINES", "Variable"),
                ("Script", "DEFINES", "Function"),
                ("Script", "MODIFIES", "Variable"),
                ("Function", "MODIFIES", "Variable"),
                ("Variable", "ASSIGNED_TO", "Variable"),
            ]
            
            for pattern in required_patterns:
                assert pattern in patterns_found, f"Missing required pattern: {pattern}"
            
            print(f"✓ All {len(required_patterns)} required relationship patterns found")
            return True
        
        return asyncio.run(run_test())

    def test_schema_compliance(self):
        """Test that the schema matches the requirements."""
        print("\nTesting schema compliance...")
        
        # Test that SCHEMA is properly defined
        assert SCHEMA is not None, "SCHEMA should be defined"
        
        # Test node types
        node_labels = [node.label for node in SCHEMA.node_types]
        expected_labels = ["Function", "Variable", "Script"]
        assert all(label in node_labels for label in expected_labels), f"Missing node types: {expected_labels}"
        
        # Test relationship types
        rel_labels = [rel.label for rel in SCHEMA.relationship_types]
        expected_rel_labels = ["CALLS", "USES", "DEFINES", "ASSIGNED_TO", "MODIFIES"]
        assert all(label in rel_labels for label in expected_rel_labels), f"Missing relationship types: {expected_rel_labels}"
        
        # Test patterns
        assert len(SCHEMA.patterns) >= 12, f"Should have at least 12 patterns, found {len(SCHEMA.patterns)}"
        
        print("✓ Schema compliance verified")
        return True

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

if __name__ == "__main__":
    unittest.main()
