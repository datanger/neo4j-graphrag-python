"""Module for indexing codebases into Neo4j for retrieval."""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import neo4j
from tqdm import tqdm

from neo4j_graphrag.embeddings.base import Embedder


class CodeIndexer:
    """Indexes a codebase into Neo4j for retrieval."""
    
    def __init__(
        self,
        driver: neo4j.Driver,
        embedder: Optional[Embedder] = None,
        database: Optional[str] = None
    ):
        """Initialize the code indexer.
        
        Args:
            driver: Neo4j driver instance
            embedder: Embedder for generating vector embeddings
            database: Neo4j database name (optional)
        """
        self.driver = driver
        self.embedder = embedder
        self.database = database
        
    def _process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file and extract code elements.
        
        Handles both Python (.py) and MATLAB (.m) files.
        """
        file_ext = file_path.suffix.lower()
        
        # Handle MATLAB files
        if file_ext == '.m':
            # try:
            from src.neo4j_graphrag.matlab_analyzer.ast_parser import MATLABASTParser
            
            # Parse MATLAB file
            parser = MATLABASTParser(path=str(file_path))
            parser.parse()
            output = parser.format_output()
            
            # Convert MATLAB parser output to standard format
            elements = []
            relationships = []
            
            for line in output.split(';'):
                if not line.strip():
                    continue
                    
                parts = [p.strip('"') for p in line.split('|')]
                if len(parts) < 4:
                    continue
                    
                if parts[0] == 'entity':
                    # Handle node
                    node_id = parts[1]
                    node_type = parts[2]
                    try:
                        properties = json.loads(parts[3])
                        
                        element = {
                            'id': node_id,
                            'type': node_type,
                            'name': properties.get('name', node_id.split('_')[-1].lower()),
                            'file_path': str(file_path),
                            'properties': properties
                        }
                        elements.append(element)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing node properties: {e}")
                        continue
                        
                elif parts[0] == 'relationship':
                    # Handle relationship
                    if len(parts) >= 6:
                        try:
                            rel = {
                                'source': parts[1],
                                'target': parts[2],
                                'type': parts[3],
                                'properties': json.loads(parts[4]) if len(parts) > 4 and parts[4] else {}
                            }
                            relationships.append(rel)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing relationship properties: {e}")
                            continue
            
            # Process relationships and add them to the appropriate nodes
            node_map = {node['id']: node for node in elements}
            
            for rel in relationships:
                source_id = rel['source']
                target_id = rel['target']
                
                # Add relationship to source node
                if source_id in node_map:
                    if 'relationships' not in node_map[source_id]:
                        node_map[source_id]['relationships'] = []
                    node_map[source_id]['relationships'].append({
                        'type': rel['type'],
                        'target': target_id,
                        'properties': rel.get('properties', {})
                    })
                
                # For bidirectional relationships, you can uncomment this section
                """
                # Add reverse relationship to target node
                if target_id in node_map:
                    if 'relationships' not in node_map[target_id]:
                        node_map[target_id]['relationships'] = []
                    node_map[target_id]['relationships'].append({
                        'type': f"{rel['type']}_BY",  # e.g., CALLS -> CALLED_BY
                        'target': source_id,
                        'properties': rel.get('properties', {})
                    })
                """
                    
            return elements
                
            # except Exception as e:
            #     print(f"Error processing MATLAB file {file_path}: {str(e)}")
            #     return []
        
        # Handle Python files (original implementation)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            elements = []
            
            # Process functions and classes
            for node in ast.walk(tree):
                element = None
                
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    element = {
                        'type': 'function',
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'code': ast.unparse(node),
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'file_path': str(file_path)
                    }
                elif isinstance(node, ast.ClassDef):
                    element = {
                        'type': 'class',
                        'name': node.name,
                        'docstring': ast.get_docstring(node) or '',
                        'code': ast.unparse(node),
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'file_path': str(file_path)
                    }
                
                if element and self.embedder:
                    # Create embedding for semantic search
                    text_to_embed = f"{element['type']} {element['name']}\n{element['docstring']}"
                    element['embedding'] = self.embedder.embed_query(text_to_embed)
                
                if element:
                    elements.append(element)
            
            return elements
            
        except SyntaxError:
            # Skip files with syntax errors
            return []
    
    def _create_nodes_and_relationships(self, elements: List[Dict], tx) -> None:
        """Create nodes and relationships in Neo4j.
        
        Args:
            elements: List of code elements to process
            tx: Neo4j transaction
        """
        node_id_map = {}  # Maps element ID to Neo4j node ID
        
        # First pass: create all nodes
        for element in elements:
            if 'type' not in element or 'name' not in element:
                continue
                
            # Prepare node properties
            node_props = {
                'name': element['name'],
                'file_path': element.get('file_path', ''),
                **element.get('properties', {})
            }
            
            # Use the provided ID or generate one
            node_id = element.get('id', f"{element['type'].lower()}_{element['name']}")
            
            # Remove None values
            node_props = {k: v for k, v in node_props.items() if v is not None}
            
            try:
                # Create or update node with a unique constraint on id
                query = (
                    f"MERGE (n:{element['type']} {{id: $id}})\n"
                    "SET n += $props\n"
                    "RETURN id(n) as node_id"
                )
                result = tx.run(query, id=node_id, props=node_props)
                record = result.single()
                
                if record:
                    neo4j_node_id = record['node_id']
                    node_id_map[node_id] = neo4j_node_id
                    # Store the Neo4j node ID for relationship creation
                    element['_neo4j_id'] = neo4j_node_id
                    
                    # Handle embeddings if available
                    if self.embedder and 'embedding' in element:
                        embedding = element['embedding']
                        if isinstance(embedding, (list, tuple)) and len(embedding) > 0:
                            tx.run(
                                """
                                MATCH (n) WHERE id(n) = $node_id
                                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                                RETURN count(*)
                                """,
                                node_id=neo4j_node_id,
                                embedding=embedding
                            )
            except Exception as e:
                print(f"Error creating node {node_id}: {str(e)}")
                continue
        
        # Second pass: create relationships
        for element in elements:
            if '_neo4j_id' not in element:
                continue
                
            source_id = element['_neo4j_id']
            
            # Process relationships
            for rel in element.get('relationships', []):
                target_id = rel.get('target')
                if not target_id:
                    continue
                    
                # Look up target node ID
                neo4j_target_id = None
                
                # First try to find by node_id in our map
                if target_id in node_id_map:
                    neo4j_target_id = node_id_map[target_id]
                else:
                    # If not found, try to find by ID property
                    query = "MATCH (n {id: $target_id}) RETURN id(n) as node_id"
                    result = tx.run(query, target_id=target_id)
                    record = result.single()
                    if record:
                        neo4j_target_id = record['node_id']
                
                if neo4j_target_id is not None and neo4j_target_id != source_id:
                    rel_type = rel.get('type', 'RELATES_TO')
                    rel_props = rel.get('properties', {})
                    
                    try:
                        # Create relationship with properties
                        rel_query = (
                            "MATCH (a), (b) WHERE id(a) = $source_id AND id(b) = $target_id\n"
                            f"MERGE (a)-[r:{rel_type}]->(b)\n"
                            "SET r += $props"
                        )
                        tx.run(rel_query, 
                              source_id=source_id, 
                              target_id=neo4j_target_id,
                              props=rel_props)
                    except Exception as e:
                        print(f"Error creating relationship {rel_type} from {source_id} to {target_id}: {str(e)}")
    
    def _ensure_schema(self):
        """Ensure the Neo4j schema is set up correctly."""
        with self.driver.session(database=self.database) as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT file_path IF NOT EXISTS
                FOR (f:File) REQUIRE f.path IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT function_name IF NOT EXISTS
                FOR (f:Function) REQUIRE (f.name, f.file_path, f.start_line) IS NODE KEY
            """)
            
            session.run("""
                CREATE CONSTRAINT class_name IF NOT EXISTS
                FOR (c:Class) REQUIRE (c.name, c.file_path, c.start_line) IS NODE KEY
            """)
            
            # Create vector index if embedder is available
            if self.embedder:
                session.run("""
                    CREATE VECTOR INDEX code_embeddings IF NOT EXISTS
                    FOR (n:CodeElement) ON (n.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimension,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """, {"dimension": len(self.embedder.embed_query("test"))})
    
    def index_codebase(self, code_path: Union[str, Path]) -> None:
        """Index a codebase.
        
        Args:
            code_path: Path to the root of the codebase to index
        """
        code_path = Path(code_path)
        if not code_path.exists():
            raise ValueError(f"Code path {code_path} does not exist")
        
        self._ensure_schema()
        
        # Find all source files
        file_patterns = [
            "*.py", "*.m", "*.java", "*.c", "*.cpp", "*.ts",
            "*.md", "*.txt"
        ]
        source_files = []
        for pattern in file_patterns:
            source_files.extend(code_path.rglob(pattern))
        
        with self.driver.session(database=self.database) as session:
            for file_path in tqdm(source_files, desc="Indexing files"):
                # try:
                # Create file node
                session.run(
                    """
                    MERGE (f:File {path: $path})
                    SET f.name = $name, f.size = $size, f.last_modified = datetime()
                    """,
                    {
                        "path": str(file_path.relative_to(code_path)),
                        "name": file_path.name,
                        "size": file_path.stat().st_size
                    }
                )
                
                # Process file and add code elements
                elements = self._process_file(file_path)
                
                if not elements:
                    continue
                
                # Create nodes and relationships in a transaction
                with session.begin_transaction() as tx:
                    self._create_nodes_and_relationships(elements, tx)
                    
                    # Create file relationships for the first element (usually script/class)
                    if elements and '_neo4j_id' in elements[0]:
                        tx.run(
                            """
                            MATCH (f:File {path: $path}), (e) 
                            WHERE id(e) = $element_id
                            MERGE (f)-[:CONTAINS]->(e)
                            """,
                            {
                                "path": str(file_path.relative_to(code_path)),
                                "element_id": elements[0]['_neo4j_id']
                            }
                        )
                # except Exception as e:
                #     print(f"Error processing file {file_path}: {str(e)}")
                #     continue
        
        # Create full-text index for hybrid search if it doesn't exist
        try:
            session.run("""
                CREATE FULLTEXT INDEX code_search IF NOT EXISTS
                FOR (n:CodeElement) ON EACH [n.docstring, n.code, n.name]
            """)
        except Exception as e:
            print(f"Error creating full-text index: {str(e)}")
