"""Module containing different retriever implementations for code search."""

from typing import Any, Dict, List, Optional, Union

import neo4j
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    HybridRetriever,
    Text2CypherRetriever as BaseText2CypherRetriever,
)
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.llm.base import LLMInterface


class VectorCodeRetriever(VectorRetriever):
    """Vector-based retriever for code search.
    
    This retriever performs semantic search over code elements using vector embeddings.
    It's best for finding code that is semantically similar to the query.
    """
    
    def __init__(
        self,
        driver: neo4j.Driver,
        embedder: Embedder,
        database: Optional[str] = None,
        index_name: str = "code_embeddings",
    ):
        """Initialize the vector code retriever.
        
        Args:
            driver: Neo4j driver instance
            embedder: Embedder for generating query embeddings
            database: Neo4j database name (optional)
            index_name: Name of the vector index to use
        """
        super().__init__(
            driver=driver,
            index_name=index_name,
            embedder=embedder,
            return_properties=["name", "docstring", "code", "file_path", "start_line"],
            neo4j_database=database,
        )
    
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for code elements similar to the query text.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            **kwargs: Additional arguments to pass to the parent search method
            
        Returns:
            List of matching code elements with metadata
        """
        print(f"\n[DEBUG] Searching with query: {query_text}")
        print(f"[DEBUG] Using index: {self.index_name}")
        
        # Debug: Check vector index exists
        with self.driver.session(database=self.neo4j_database) as session:
            index_info = session.run("""
                SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
                WHERE type = 'VECTOR' AND name = $index_name
                RETURN name, labelsOrTypes, properties, options
            """, index_name=self.index_name).single()
            
            if not index_info:
                print(f"[ERROR] Vector index '{self.index_name}' not found!")
                # List all vector indexes for debugging
                all_indexes = session.run("""
                    SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
                    WHERE type = 'VECTOR'
                    RETURN name, labelsOrTypes, properties
                """).data()
                print(f"[DEBUG] Available vector indexes: {all_indexes}")
            else:
                print(f"[DEBUG] Using vector index: {dict(index_info)}")
        
        # Get embedding for the query
        print("[DEBUG] Generating embedding for query...")
        try:
            query_embedding = self.embedder.embed_query(query_text)
            print(f"[DEBUG] Generated embedding of length: {len(query_embedding) if query_embedding else 0}")
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            return []
        
        # Execute the search
        print("[DEBUG] Executing vector search...")
        try:
            result = super().search(
                query_text=query_text,
                top_k=top_k,
                **kwargs
            )
            # Convert RetrieverResult to list of dicts
            if hasattr(result, 'nodes') and hasattr(result, 'scores'):
                formatted_results = []
                for node, score in zip(result.nodes, result.scores):
                    formatted_results.append({
                        'node': node,
                        'score': score,
                        'metadata': {'score': score}
                    })
                print(f"[DEBUG] Search returned {len(formatted_results)} results")
                return formatted_results
            else:
                print("[WARNING] Unexpected result format from super().search()")
                return []
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Format results
        formatted_results = []
        for item in result:
            if hasattr(item, 'metadata') and hasattr(item, 'node'):
                # Handle case where result is a structured object with metadata and node
                formatted_results.append({
                    'score': item.metadata.get('score', 0.0),
                    'type': list(item.node.labels)[0] if hasattr(item.node, 'labels') and item.node.labels else 'CodeElement',
                    'name': item.node.get('name', ''),
                    'docstring': item.node.get('docstring', ''),
                    'file_path': item.node.get('file_path', ''),
                    'start_line': item.node.get('start_line', 0),
                    'code': item.node.get('code', ''),
                })
            elif isinstance(item, dict):
                # Handle case where result is already a dictionary
                formatted_results.append({
                    'score': item.get('score', 0.0),
                    'type': item.get('type', 'CodeElement'),
                    'name': item.get('name', ''),
                    'docstring': item.get('docstring', ''),
                    'file_path': item.get('file_path', ''),
                    'start_line': item.get('start_line', 0),
                    'code': item.get('code', ''),
                })
        
        return formatted_results


class HybridCodeRetriever(HybridRetriever):
    """Hybrid retriever that combines vector and keyword search for code.
    
    This retriever combines semantic search with keyword matching to provide
    more relevant results by leveraging both meaning and exact term matching.
    """
    
    def __init__(
        self,
        driver: neo4j.Driver,
        embedder: Embedder,
        database: Optional[str] = None,
        vector_index_name: str = "code_embeddings",
        fulltext_index_name: str = "code_search",
    ):
        """Initialize the hybrid code retriever.
        
        Args:
            driver: Neo4j driver instance
            embedder: Embedder for generating query embeddings
            database: Neo4j database name (optional)
            vector_index_name: Name of the vector index to use
            fulltext_index_name: Name of the fulltext index to use
        """
        super().__init__(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=embedder,
            return_properties=["name", "docstring", "code", "file_path", "start_line"],
            neo4j_database=database,
        )
    
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for code elements using hybrid search.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            **kwargs: Additional arguments to pass to the parent search method
            
        Returns:
            List of matching code elements with metadata
        """
        print(f"\n[DEBUG] Hybrid searching with query: {query_text}")
        print(f"[DEBUG] Using vector index: {self.vector_index_name}")
        print(f"[DEBUG] Using fulltext index: {self.fulltext_index_name}")
        
        try:
            result = super().search(
                query_text=query_text,
                top_k=top_k,
                **kwargs
            )
            
            # Convert RetrieverResult to list of dicts
            if hasattr(result, 'nodes') and hasattr(result, 'scores'):
                formatted_results = []
                for node, score in zip(result.nodes, result.scores):
                    formatted_results.append({
                        'node': node,
                        'score': score,
                        'metadata': {'score': score}
                    })
                print(f"[DEBUG] Hybrid search returned {len(formatted_results)} results")
                return formatted_results
            else:
                print("[WARNING] Unexpected result format from super().search()")
                return []
                
        except Exception as e:
            print(f"[ERROR] Hybrid search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
            
        # Format results
        formatted_results = []
        for item in result:
            if hasattr(item, 'metadata') and hasattr(item, 'node'):
                # Handle case where result is a structured object with metadata and node
                formatted_results.append({
                    'score': item.metadata.get('score', 0.0),
                    'type': list(item.node.labels)[0] if hasattr(item.node, 'labels') and item.node.labels else 'CodeElement',
                    'name': item.node.get('name', ''),
                    'docstring': item.node.get('docstring', ''),
                    'file_path': item.node.get('file_path', ''),
                    'start_line': item.node.get('start_line', 0),
                    'code': item.node.get('code', ''),
                })
            elif isinstance(item, dict):
                # Handle case where result is already a dictionary
                formatted_results.append({
                    'score': item.get('score', 0.0),
                    'type': item.get('type', 'CodeElement'),
                    'name': item.get('name', ''),
                    'docstring': item.get('docstring', ''),
                    'file_path': item.get('file_path', ''),
                    'start_line': item.get('start_line', 0),
                    'code': item.get('code', ''),
                })
        
        return formatted_results


class Text2CypherCodeRetriever(BaseText2CypherRetriever):
    """Retriever that uses natural language to generate Cypher queries for code search.
    
    This retriever is useful for complex queries that can be expressed in natural language
    and translated to Cypher for precise graph traversal.
    """
    
    def __init__(
        self,
        driver: neo4j.Driver,
        llm: LLMInterface,
        database: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ):
        """Initialize the text-to-Cypher code retriever.
        
        Args:
            driver: Neo4j driver instance
            llm: LLM for generating Cypher queries
            database: Neo4j database name (optional)
            examples: List of example queries (optional)
        """
        if examples is None:
            examples = [
                "Find all functions that process user authentication",
                "Show me classes that inherit from BaseModel",
                "Find functions that call the database and handle errors"
            ]
            
        super().__init__(
            driver=driver,
            llm=llm,
            neo4j_database=database,
            examples=examples,
        )
    
    def search(
        self,
        query_text: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for code elements using natural language.
        
        Args:
            query_text: Natural language query
            **kwargs: Additional arguments to pass to the parent search method
            
        Returns:
            List of matching code elements with metadata
        """
        result = super().search(query_text=query_text, **kwargs)
        
        # Format results
        formatted_results = []
        for record in result.records:
            element = record.get('element')
            if element is not None:
                formatted_results.append({
                    'score': 1.0,  # Text2Cypher doesn't provide scores
                    'type': list(element.labels)[0] if hasattr(element, 'labels') and element.labels else 'CodeElement',
                    'name': element.get('name', ''),
                    'docstring': element.get('docstring', ''),
                    'file_path': element.get('file_path', ''),
                    'start_line': element.get('start_line', 0),
                    'code': element.get('code', ''),
                })
        
        return formatted_results
