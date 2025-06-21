import os
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from neo4j_graphrag.experimental.components.code_extractor.matlab.matlab_extractor import MatlabExtractor

class TestDuplicateRemoval(unittest.TestCase):
    """Test that duplicate nodes and edges are properly removed."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        test_dir = Path(__file__).parent / "examples"
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a simple test file with potential duplicates
        with open(test_dir / "test_duplicates.m", "w") as f:
            f.write("""
            % Test file with potential duplicates
            
            % Define variables multiple times
            x = 10;
            y = x + 5;
            x = 20;  % Redefine x
            
            % Use variables multiple times
            z = x + y;
            result = x + y + z;
            
            % Call function multiple times
            helper_function(x);
            helper_function(y);
            """)
        
        # Create helper function
        with open(test_dir / "helper_function.m", "w") as f:
            f.write("""
            function result = helper_function(input)
                result = input * 2;
            end
            """)
        
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
    
    def test_no_duplicate_nodes(self):
        """Test that no duplicate nodes are created."""
        print("\n=== TESTING DUPLICATE NODE REMOVAL ===")
        
        # Process the test file
        from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
        
        chunks_list = []
        for i, script_file in enumerate(Path(__file__).parent.glob("examples/*.m")):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            chunk = TextChunk(
                text=script_content,
                index=i,
                metadata={"file_path": str(script_file)}
            )
            chunks_list.append(chunk)
        
        chunks = TextChunks(chunks=chunks_list)
        
        # Process chunks
        result = asyncio.run(self.extractor.run(chunks=chunks))
        graph = result.graph
        
        # Convert to simple format for analysis
        nodes = []
        for node in graph.nodes:
            nodes.append({
                'id': node.id,
                'label': node.label,
                'name': node.properties.get('name', ''),
            })
        
        # Check for duplicate node IDs
        node_ids = [n['id'] for n in nodes]
        unique_node_ids = set(node_ids)
        
        print(f"Total nodes: {len(nodes)}")
        print(f"Unique node IDs: {len(unique_node_ids)}")
        print("Node IDs:", node_ids)
        
        self.assertEqual(len(nodes), len(unique_node_ids), 
                        f"Found duplicate nodes: {len(nodes)} nodes but only {len(unique_node_ids)} unique IDs")
        
        # Check for duplicate node names within same label
        node_groups = {}
        for node in nodes:
            label = node['label']
            name = node['name']
            if label not in node_groups:
                node_groups[label] = []
            node_groups[label].append(name)
        
        for label, names in node_groups.items():
            unique_names = set(names)
            if len(names) != len(unique_names):
                print(f"Warning: Duplicate names in {label}: {names}")
        
        print("✓ No duplicate nodes found")
    
    def test_no_duplicate_edges(self):
        """Test that no duplicate edges are created."""
        print("\n=== TESTING DUPLICATE EDGE REMOVAL ===")
        
        # Process the test file
        from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
        
        chunks_list = []
        for i, script_file in enumerate(Path(__file__).parent.glob("examples/*.m")):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            chunk = TextChunk(
                text=script_content,
                index=i,
                metadata={"file_path": str(script_file)}
            )
            chunks_list.append(chunk)
        
        chunks = TextChunks(chunks=chunks_list)
        
        # Process chunks
        result = asyncio.run(self.extractor.run(chunks=chunks))
        graph = result.graph
        
        # Convert to simple format for analysis
        edges = []
        for edge in graph.relationships:
            edges.append({
                'source': edge.start_node_id,
                'target': edge.end_node_id,
                'type': edge.type,
                'line_number': edge.properties.get('line_number'),
            })
        
        # Filter to only MATLAB-specific edge types
        matlab_edge_types = {'CALLS', 'USES', 'DEFINES', 'MODIFIES', 'ASSIGNED_TO'}
        matlab_edges = [e for e in edges if e['type'] in matlab_edge_types]
        
        # Check for duplicate edges
        edge_keys = [(e['source'], e['target'], e['type'], e.get('line_number')) for e in matlab_edges]
        unique_edge_keys = set(edge_keys)
        
        print(f"Total edges: {len(edges)}")
        print(f"MATLAB-specific edges: {len(matlab_edges)}")
        print(f"Unique MATLAB edge keys: {len(unique_edge_keys)}")
        
        # Group edges by type for better analysis
        edge_types = {}
        for edge in matlab_edges:
            edge_type = edge['type']
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append(f"{edge['source']} -> {edge['target']}")
        
        print("MATLAB edges by type:")
        for edge_type, edge_list in edge_types.items():
            print(f"  {edge_type}: {len(edge_list)} edges")
            if len(edge_list) > 10:  # Only show first 10 if too many
                print(f"    Examples: {edge_list[:10]}")
            else:
                print(f"    All: {edge_list}")
        
        self.assertEqual(len(matlab_edges), len(unique_edge_keys), 
                        f"Found duplicate MATLAB edges: {len(matlab_edges)} edges but only {len(unique_edge_keys)} unique keys")
        
        print("✓ No duplicate MATLAB edges found")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        test_dir = Path(__file__).parent / "examples"
        for script_file in test_dir.glob("*.m"):
            try:
                script_file.unlink()
            except:
                pass

if __name__ == '__main__':
    unittest.main() 