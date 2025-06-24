"""Validation script: ensure no single Variable node has DEFINES relationships from more than one scope.

Usage: run after KG is loaded into Neo4j. Exits with code 0 if check passes, non-zero otherwise.
Env vars:
  NEO4J_URI (default bolt://localhost:7687)
  NEO4J_USERNAME (default neo4j)
  NEO4J_PASSWORD (default password)
"""
from neo4j import GraphDatabase
import os
import sys
import pprint

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

cypher = (
    "MATCH (src)-[:DEFINES]->(v:Variable)\n"
    "WITH v, collect(distinct src.id) AS definers\n"
    "WHERE size(definers) > 1\n"
    "RETURN v.id AS variable_id, definers"
)

def main() -> None:
    print(f"Connecting to Neo4j at {URI} ...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        rows = session.run(cypher).data()
    driver.close()

    if not rows:
        print("PASS: No variable node is defined by more than one scope.")
        sys.exit(0)

    print("FAIL: The following variable nodes have multiple DEFINES relationships:")
    pprint.pprint(rows, width=120)
    sys.exit(1)


if __name__ == "__main__":
    main()
