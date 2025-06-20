"""Example usage of the code indexer and retrievers."""

import os
from pathlib import Path
from typing import Union, List, Dict, Any
from pathlib import Path
import os

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.llm.ollama_llm import OllamaLLM

from coder import CodeIndexer, VectorCodeRetriever, HybridCodeRetriever, Text2CypherCodeRetriever

# Configuration
# Try with bolt protocol first, which is more commonly used with Neo4j Desktop
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize Neo4j driver with connection verification
try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=5  # 5 second timeout
    )
    # Verify the connection works
    driver.verify_connectivity()
    print("Successfully connected to Neo4j!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    print(f"URI: {NEO4J_URI}")
    print(f"User: {NEO4J_USER}")
    print("Please check your Neo4j Desktop connection details and try again.")
    exit(1)

def check_database_contents(driver: Driver, database: str = None) -> Dict[str, int]:
    """Check the contents of the Neo4j database to see what's been indexed."""
    try:
        with driver.session(database=database) as session:
            # First try with APOC if available
            try:
                result = session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(*) as count', {})
                    YIELD value
                    RETURN label, value.count as count
                    ORDER BY count DESC
                """)
                return {record["label"]: record["count"] for record in result}
            except Exception:
                # Fallback to non-APOC query
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(*) as count
                    ORDER BY count DESC
                """)
                counts = {}
                for record in result:
                    for label in record["labels"]:
                        counts[label] = counts.get(label, 0) + record["count"]
                return counts
    except Exception as e:
        print(f"Error checking database contents: {e}")
        return {}

def index_codebase(code_path: Union[str, Path]):
    """Example of indexing a codebase."""
    # Initialize embedder
    embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Create indexer
    indexer = CodeIndexer(
        driver=driver,
        embedder=embedder,
        database=NEO4J_DATABASE
    )
    
    # Index the codebase
    print(f"Indexing codebase at: {code_path}")
    indexer.index_codebase(code_path)
    print("Indexing complete!")

def search_with_vector_retriever():
    """Example of using the vector retriever."""
    # Initialize embedder
    embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Create retriever
    retriever = VectorCodeRetriever(
        driver=driver,
        embedder=embedder,
        database=NEO4J_DATABASE
    )
    
    # Search for code - using specific queries based on the database contents
    queries = [
        "function definition",
        "class definition",
        "import statement",
        "function call",
        "class method"
    ]
    
    try:
        results = []
        for query in queries:
            print(f"\nSearching for: {query}")
            try:
                query_results = retriever.search(query, top_k=2)
                print(f"Found {len(query_results)} results")
                results.extend(query_results)
                if len(results) >= 5:  # Stop if we have enough results
                    results = results[:5]
                    break
            except Exception as e:
                print(f"Error with query '{query}': {e}")
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
        results = []
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.3f}):")
        print(f"Type: {result['type']}")
        print(f"Name: {result['name']}")
        print(f"File: {result['file_path']}:{result['start_line']}")
        print(f"Docstring: {result['docstring'][:100]}..." if result['docstring'] else "No docstring")

def search_with_hybrid_retriever():
    """Example of using the hybrid retriever."""
    # First check database contents
    print("\nChecking database contents...")
    counts = check_database_contents(driver, NEO4J_DATABASE)
    if counts:
        print("\nNodes in database by label:")
        for label, count in counts.items():
            print(f"- {label}: {count}")
    else:
        print("No nodes found in the database. Have you indexed any code?")
        return
    
    # Initialize embedder
    print("\nInitializing Ollama embedder...")
    embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Create retriever
    print("Creating HybridCodeRetriever...")
    retriever = HybridCodeRetriever(
        driver=driver,
        embedder=embedder,
        database=NEO4J_DATABASE,
        vector_index_name="code_embeddings",
        fulltext_index_name="code_search"
    )
    
    # Search for code - using specific queries based on the database contents
    queries = [
        "class definition",
        "function definition",
        "import statement",
        "function call",
        "class method"
    ]
    
    try:
        results = []
        for query in queries:
            print(f"\nSearching for: {query}")
            try:
                query_results = retriever.search(query, top_k=2)
                print(f"Found {len(query_results)} results")
                results.extend(query_results)
                if len(results) >= 5:  # Stop if we have enough results
                    results = results[:5]
                    break
            except Exception as e:
                print(f"Error with query '{query}': {e}")
    except Exception as e:
        print(f"Error during search: {e}")
        results = []
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.3f}):")
        print(f"Type: {result['type']}")
        print(f"Name: {result['name']}")
        print(f"File: {result['file_path']}:{result['start_line']}")
        print(f"Docstring: {result['docstring'][:100]}..." if result['docstring'] else "No docstring")

def search_with_text2cypher_retriever():
    """Example of using the text-to-Cypher retriever."""
    # Initialize LLM
    llm = OllamaLLM(model_name="deepseek-coder-v2:16b", model_params={"temperature": 0})
    
    # Create retriever
    retriever = Text2CypherCodeRetriever(
        driver=driver,
        llm=llm,
        database=NEO4J_DATABASE
    )
    
    # Search for code using natural language
    query = "Find all functions that process HTTP requests and handle errors"
    print(f"\nSearching for: {query}")
    results = retriever.search(query)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Type: {result['type']}")
        print(f"Name: {result['name']}")
        print(f"File: {result['file_path']}:{result['start_line']}")
        print(f"Docstring: {result['docstring'][:100]}..." if result['docstring'] else "No docstring")
        print(f"Code:\n{result['code'][:200]}..." if result['code'] else "No code")

if __name__ == "__main__":
    try:
        # First check if we should index the codebase
        should_index = input("Do you want to index the codebase? (y/n): ").strip().lower() == 'y'
        
        if should_index:
            code_path = input(f"Enter path to index (default: {os.getcwd()}): ").strip()
            if not code_path:
                code_path = os.getcwd()
            print(f"Indexing codebase at: {code_path}")
            index_codebase(code_path)
        
        # Check database contents first
        print("\n=== Database Contents ===")
        counts = check_database_contents(driver, NEO4J_DATABASE)
        if counts:
            print("\nNodes in database by label:")
            for label, count in counts.items():
                print(f"- {label}: {count}")
            
            # Only run searches if we have data
            run_searches = input("\nRun example searches? (y/n): ").strip().lower() == 'y'
            if run_searches:
                print("\n=== Vector Search ===")
                search_with_vector_retriever()
                
                print("\n=== Hybrid Search ===")
                search_with_hybrid_retriever()
                
        else:
            print("No data found in the database. Please index a codebase first.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the driver when done
        print("\nClosing database connection...")
        driver.close()
