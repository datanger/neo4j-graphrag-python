"""Interactive Q&A over Neo4j knowledge-graph built by
`examples/customize/build_graph/pipeline/kg_builder_from_code.py`.

A simple loop:
1. Reads a natural-language question from the user.
2. Uses an Ollama-hosted LLM (via `OllamaLLM`) to translate the question into
   a Cypher query with `Text2CypherRetriever`.
3. Executes the generated Cypher via the Neo4j driver and prints the results
   together with the query that was run.

Environment variables (all optional):
    NEO4J_URI          default: bolt://localhost:7687
    NEO4J_USERNAME     default: neo4j
    NEO4J_PASSWORD     default: password
    NEO4J_DATABASE     default: neo4j
    OLLAMA_MODEL       default: deepseek-r1:14b

Example:
    $ python query_graph.py
    Q> Which class calls function myHelper?
    (prints Cypher and results)
"""
from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

from neo4j import GraphDatabase

from neo4j_graphrag.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever, RawSearchResult


# ---------------------------------------------------------------------------
# Configuration (env vars with sensible defaults)
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")


def format_records(records: List[Dict[str, Any]]) -> str:
    """Pretty-format a list of Neo4j records (as dicts)."""
    if not records:
        return "<no results>"

    # Collect all unique keys to make a header row
    keys: List[str] = sorted({key for rec in records for key in rec.keys()})
    header = " | ".join(keys)
    sep = "-+-".join("-" * len(k) for k in keys)

    lines = [header, sep]
    for rec in records:
        line = " | ".join(str(rec.get(k, "")) for k in keys)
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    print("Connecting to Neo4j…", end=" ")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    # quick connectivity test
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1").consume()
    except Exception as exc:  # pragma: no cover
        print("failed")
        sys.exit(f"❌  Cannot connect to Neo4j: {exc}")
    print("ok")

    # LLM wrapper (temperature 0 for deterministic Cypher)
    llm = OllamaLLM(model_name=OLLAMA_MODEL, model_params={"temperature": 0})

        # Try to create Text→Cypher retriever. Neo4j sandboxing might block APOC
    # procedures that the helper uses to auto-fetch schema. If that fails,
    # fall back to a minimal hand-crafted schema string that covers the MATLAB
    # KG built by `kg_builder_from_code.py`.
    try:
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_database=NEO4J_DATABASE,
        )
    except Exception as exc:
        print("⚠️  Auto-fetching Neo4j schema failed:", exc)
        print("   Falling back to builtin schema description…")
        SCHEMA_FALLBACK = """
NODE TYPES
----------
Class(name, description, file_path)
Function(name, description, file_path, parameters, returns)
Variable(name, file_path)
Script(name, file_path)

RELATIONSHIPS
-------------
(:Function)-[:CALLS]->(:Function|:Script)
(:Script)-[:CALLS]->(:Function|:Script)
(:Function)-[:USES]->(:Variable)
(:Function)-[:DEFINES]->(:Variable)
(:Variable)-[:ASSIGNED_TO]->(:Variable)
(:Class)-[:HAS_METHOD]->(:Function)
(:Class)-[:HAS_PROPERTY]->(:Variable)
(:Class)-[:INHERITS_FROM]->(:Class)
"""
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=SCHEMA_FALLBACK,
            neo4j_database=NEO4J_DATABASE,
        )

    print("\nAsk me something about the graph!  (Ctrl-D / Ctrl-C to exit)\n")

    try:
        for line in sys.stdin:
            query = line.strip()
            if not query:
                continue
            try:
                print(query)
                result: RawSearchResult = retriever.get_search_results(query)
            except Exception as exc:
                print(f"⚠️  Retrieval error: {exc}\n")
                continue

            cypher = result.metadata.get("cypher", "<unknown>")
            print("\nGenerated Cypher:\n", cypher, sep="")
            print("\nResults:")
            print(format_records([dict(r) for r in result.records]))
            print("\n---")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
