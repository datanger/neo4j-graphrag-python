import neo4j
import sys

# --- Neo4j Connection Settings ---
# You can modify these values if your setup is different
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DB = "neo4j"
# ---------------------------------

def clear_database():
    """Connects to Neo4j and deletes all nodes and relationships."""
    driver = None
    try:
        driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DB) as session:
            print(f"Connecting to database '{NEO4J_DB}'...")
            print("Executing Cypher: MATCH (n) DETACH DELETE n")
            session.run("MATCH (n) DETACH DELETE n")
            print(f"Successfully cleared all data from the '{NEO4J_DB}' database.")
    except neo4j.exceptions.AuthError as e:
        sys.exit(f"Authentication failed. Please check your credentials in the script. Error: {e}")
    except neo4j.exceptions.ServiceUnavailable as e:
        sys.exit(f"Could not connect to Neo4j at {NEO4J_URI}. Is the database running? Error: {e}")
    except Exception as e:
        sys.exit(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    clear_database()
