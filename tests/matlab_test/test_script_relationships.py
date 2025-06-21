"""Tests for MATLAB script and function call relationships."""
import os
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor_fixed import MatlabExtractor

class TestMatlabScriptRelationships(unittest.TestCase):
    """Test MATLAB script and function call relationships."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path(__file__).parent
        cls.test_script = cls.test_dir / 'test_script_and_variable_relationships.m'
        cls.external_script = cls.test_dir / 'external_script.m'
        
    def setUp(self):
        """Set up before each test."""
        self.extractor = MatlabExtractor(debug=True)
        
    def test_script_to_script_calls(self):
        """Test that script-to-script calls are correctly detected."""
        # Process the external script first
        with open(self.external_script, 'r') as f:
            external_content = f.read()
        nodes, edges = self.extractor.extract(str(self.external_script), external_content)
        
        # Process the test script
        with open(self.test_script, 'r') as f:
            test_content = f.read()
        nodes, edges = self.extractor.extract(str(self.test_script), test_content)
        
        # Verify nodes were created for both scripts and the function
        script_names = {n['name'] for n in nodes if n['label'] == 'Script'}
        self.assertIn('test_script_and_variable_relationships', script_names)
        self.assertIn('external_script', script_names)
        
        # Verify function node was created
        function_names = {n['name'] for n in nodes if n['label'] == 'Function'}
        self.assertIn('call_external_script', function_names)
        
        # Verify CALLS relationships
        calls_relationships = [
            (e['source'], e['target'], e['properties'].get('call_type'))
            for e in edges if e['label'] == 'CALLS'
        ]
        
        # Get node IDs for verification
        script_id = next(n['id'] for n in nodes 
                        if n['name'] == 'test_script_and_variable_relationships' 
                        and n['label'] == 'Script')
        external_script_id = next(n['id'] for n in nodes 
                                if n['name'] == 'external_script' 
                                and n['label'] == 'Script')
        func_id = next(n['id'] for n in nodes 
                      if n['name'] == 'call_external_script' 
                      and n['label'] == 'Function')
        
        # Verify script-to-script call
        self.assertIn((script_id, external_script_id, 'script_call'), calls_relationships)
        
        # Verify function-to-script call
        self.assertIn((func_id, external_script_id, 'script_call'), calls_relationships)
    
    def test_variable_assignments(self):
        """Test that variable assignments are correctly tracked."""
        # Process both scripts
        with open(self.external_script, 'r') as f:
            external_content = f.read()
        self.extractor.extract(str(self.external_script), external_content)
        
        with open(self.test_script, 'r') as f:
            test_content = f.read()
        nodes, edges = self.extractor.extract(str(self.test_script), test_content)
        
        # Get node IDs
        script_id = next(n['id'] for n in nodes 
                        if n['name'] == 'test_script_and_variable_relationships' 
                        and n['label'] == 'Script')
        
        # Verify variables are defined
        var_names = {n['name'] for n in nodes if n['label'] == 'Variable'}
        expected_vars = {'x', 'y', 'z', 'a', 'b', 'script_var1', 'script_var2', 'result'}
        self.assertTrue(expected_vars.issubset(var_names))
        
        # Verify DEFINES relationships
        defines_relationships = [
            (e['source'], e['target'])
            for e in edges if e['label'] == 'DEFINES'
        ]
        
        # Verify each variable is defined in the correct scope
        for var in ['x', 'y', 'z']:
            var_id = next(n['id'] for n in nodes if n['name'] == var)
            self.assertIn((script_id, var_id), defines_relationships)

if __name__ == '__main__':
    unittest.main()
