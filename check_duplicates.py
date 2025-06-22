import neo4j
import sys
from collections import defaultdict

# --- Neo4j Connection Settings ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DB = "neo4j"
# ---------------------------------

def check_for_duplicates():
    """
    Connects to Neo4j, fetches all nodes and relationships,
    and checks for duplicates based on key properties.
    """
    driver = None
    try:
        driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DB) as session:
            print(f"Connecting to database '{NEO4J_DB}' to check for duplicates...")

            # 1. Fetch all nodes and check for duplicates
            nodes_result = session.run("MATCH (n) RETURN id(n) AS neo4j_id, labels(n) AS labels, properties(n) AS props")

            node_counts = defaultdict(int)
            node_properties = {}
            total_nodes = 0

            for record in nodes_result:
                total_nodes += 1
                labels = tuple(sorted(record["labels"]))
                props = record["props"]

                # Use the 'id' property from the graphrag script as the unique key
                # If 'id' doesn't exist, use the full property set as a fallback.
                unique_key = props.get("id", str(sorted(props.items())))

                node_identifier = (labels, unique_key)
                node_counts[node_identifier] += 1
                node_properties[node_identifier] = props

            print("\n--- Node Duplication Report ---")
            print(f"Total nodes found in database: {total_nodes}")
            duplicate_nodes_found = False
            for identifier, count in node_counts.items():
                if count > 1:
                    duplicate_nodes_found = True
                    labels, key = identifier
                    print(f"  - DUPLICATE FOUND: {count} nodes with Labels {labels} and Key '{key}'")

            if not duplicate_nodes_found:
                print("  + No duplicate nodes found based on their 'id' property.")

            # 2. Fetch all relationships and check for duplicates
            rels_result = session.run("""
                MATCH (start)-[r]->(end)
                RETURN properties(start).id AS start_id,
                       properties(end).id AS end_id,
                       type(r) AS rel_type
            """)

            rel_counts = defaultdict(int)
            total_rels = 0

            for record in rels_result:
                total_rels += 1
                rel_identifier = (
                    record["start_id"],
                    record["rel_type"],
                    record["end_id"]
                )
                rel_counts[rel_identifier] += 1

            print("\n--- Relationship Duplication Report ---")
            print(f"Total relationships found in database: {total_rels}")
            duplicate_rels_found = False
            for identifier, count in rel_counts.items():
                if count > 1:
                    duplicate_rels_found = True
                    start, rel_type, end = identifier
                    print(f"  - DUPLICATE FOUND: {count} relationships of type '{rel_type}' from node '{start}' to '{end}'")

            if not duplicate_rels_found:
                print("  + No duplicate relationships found.")

            print("\n--- Summary ---")
            if not duplicate_nodes_found and not duplicate_rels_found:
                print("✅ Success: No duplicate data was found in the database.")
            else:
                print("❌ Issue: Duplicate data exists. Please clear the database and re-run the import script with merging enabled.")


    except neo4j.exceptions.ServiceUnavailable as e:
        sys.exit(f"Connection Error: Could not connect to Neo4j at {NEO4J_URI}. Is it running? Details: {e}")
    except Exception as e:
        sys.exit(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    check_for_duplicates()
