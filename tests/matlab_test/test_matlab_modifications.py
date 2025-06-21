import os
import sys
import asyncio
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, AsyncMock

# Mock LLM class for testing
class MockLLM:
    def __init__(self, *args, **kwargs):
        self.generate = AsyncMock(return_value="Mock description")
    
    async def __call__(self, *args, **kwargs):
        return await self.generate(*args, **kwargs)

from neo4j_graphrag.experimental.components.code_extractor import MatlabExtractor

class TestMatlabModifications(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all tests."""
        # Initialize a mock LLM
        cls.llm = MockLLM()
        
    def setUp(self):
        """Set up before each test."""
        self.extractor = MatlabExtractor(
            llm=self.llm,
            debug=True  # Enable debug output
        )
    
    async def run_extractor(self, test_file: str) -> Dict[str, Any]:
        """Helper method to run the extractor and return the result."""
        # Read the test file
        with open(test_file, 'r') as f:
            code = f.read()
        
        # Run the extractor with the code directly
        self.extractor.text = code
        self.extractor.path = test_file
        self.extractor._parse_matlab_code()
        return {}
    
    def test_modification_relationships(self):
        """Test that variable modification relationships are correctly detected."""
        # Test file with various modification patterns
        test_file = "/tmp/matlab_test/test_modifications.m"
        
        # Run the extractor in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.run_extractor(test_file))
        finally:
            loop.close()
        
        # Get all nodes and edges
        nodes = self.extractor.nodes
        edges = self.extractor.edges
        
        # Debug output
        print("\n=== DEBUG OUTPUT ===")
        print(f"Total nodes: {len(nodes)}")
        print(f"Total edges: {len(edges)}")
        
        # Print all nodes
        print("\nNODES:")
        for node in nodes:
            print(f"- {node.get('label')} {node.get('name', '?')} (id: {node.get('id')})")
        
        # Print all edges
        print("\nEDGES:")
        for edge in edges:
            source = next((n for n in nodes if n['id'] == edge['source']), {'name': '?'})
            target = next((n for n in nodes if n['id'] == edge['target']), {'name': '?'})
            print(f"- {source.get('name', '?')} --{edge.get('label', '?')}--> {target.get('name', '?')}")
        
        # Get all MODIFIES relationships
        modifies_rels = [
            e for e in edges 
            if e.get('label') == 'MODIFIES'
        ]
        
        # Debug output for MODIFIES relationships
        print("\nMODIFIES relationships:")
        for rel in modifies_rels:
            source = next((n for n in nodes if n['id'] == rel['source']), {})
            target = next((n for n in nodes if n['id'] == rel['target']), {})
            print(f"  {source.get('name', '?')} --MODIFIES--> {target.get('name', '?')} (line: {rel.get('properties', {}).get('line', '?')})")
        
        # Verify we found some MODIFIES relationships
        self.assertGreater(len(modifies_rels), 0, "No MODIFIES relationships found")
        
        # Check for specific expected modifications in the new test file
        expected_modifications = [
            ('inner_scope', 'out2'),
            ('swap_values', 'out1'),
            ('swap_values', 'out2'),
            ('modify_handle_object', 'result'),
            ('compute_results', 'sum_val'),
            ('compute_results', 'diff_val'),
            ('outer_scope_test', 'out1'),
            ('outer_scope_test', 'out2'),
            ('modify_struct_field', 's'),
            ('GlobalCounter', 'obj'),
            ('get', 'c'),
            ('DataHandle', 'obj')
        ]
        
        # Convert to sets for easier comparison
        found_modifications = {
            (next((n.get('name', '?') for n in nodes if n.get('id') == rel['source']), '?'),
             next((n.get('name', '?') for n in nodes if n.get('id') == rel['target']), '?'))
            for rel in modifies_rels
        }
        
        if self.extractor.debug:
            print("\nFound modifications:", sorted(found_modifications))
            print("Expected modifications:", expected_modifications)
        
        # Verify all expected modifications are present
        for mod in expected_modifications:
            found = any(src == mod[0] and tgt == mod[1] for src, tgt in found_modifications)
            self.assertTrue(found, f"Expected modification not found: {mod}")
            
        # Verify we have at least the expected number of MODIFIES relationships
        self.assertGreaterEqual(len(modifies_rels), len(expected_modifications),
                             f"Expected at least {len(expected_modifications)} MODIFIES relationships, found {len(modifies_rels)}")
        
        # Verify the scope information in MODIFIES relationships
        for rel in modifies_rels:
            if 'properties' in rel and 'scope_type' in rel['properties']:
                self.assertIn(rel['properties']['scope_type'], ['Function', 'Script'],
                            f"Unexpected scope type: {rel['properties']['scope_type']}")
                self.assertIn('scope_id', rel['properties'], "Missing scope_id in MODIFIES relationship")

if __name__ == '__main__':
    unittest.main()
