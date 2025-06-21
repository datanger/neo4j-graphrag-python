from neo4j import GraphDatabase

def clear_neo4j():
    # Neo4j connection details - update these if your setup is different
    URI = "bolt://localhost:7687"
    AUTH = ("neo4j", "password")  # Default credentials, update if different
    
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                # Delete all nodes and relationships
                result = session.run("""
                MATCH (n)
                DETACH DELETE n
                RETURN count(*) as deleted
                """)
                count = result.single()["deleted"]
                print(f"Successfully deleted {count} nodes and their relationships")
    except Exception as e:
        print(f"Error clearing Neo4j database: {e}")

if __name__ == "__main__":
    clear_neo4j()
