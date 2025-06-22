#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import logging
import numpy as np
from abc import abstractmethod
from typing import Any, Generator, Literal, Optional, Union

import neo4j
from pydantic import validate_call

from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.neo4j_queries import (
    UPSERT_NODE_QUERY,
    UPSERT_NODE_QUERY_VARIABLE_SCOPE_CLAUSE,
    UPSERT_RELATIONSHIP_QUERY,
    UPSERT_RELATIONSHIP_QUERY_VARIABLE_SCOPE_CLAUSE,
)
from neo4j_graphrag.utils.version_utils import (
    get_version,
    is_version_5_23_or_above,
)
from neo4j_graphrag.utils import driver_config

logger = logging.getLogger(__name__)


def batched(rows: list[Any], batch_size: int) -> Generator[list[Any], None, None]:
    index = 0
    for i in range(0, len(rows), batch_size):
        start = i
        end = min(start + batch_size, len(rows))
        batch = rows[start:end]
        yield batch
        index += 1


class KGWriterModel(DataModel):
    """Data model for the output of the Knowledge Graph writer.

    Attributes:
        status (Literal["SUCCESS", "FAILURE"]): Whether the write operation was successful.
    """

    status: Literal["SUCCESS", "FAILURE"]
    metadata: Optional[dict[str, Any]] = None


class KGWriter(Component):
    """Abstract class used to write a knowledge graph to a data store."""

    @abstractmethod
    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """
        Writes the graph to a data store.

        Args:
            graph (Neo4jGraph): The knowledge graph to write to the data store.
            lexical_graph_config (LexicalGraphConfig): Node labels and relationship types in the lexical graph.
        """
        pass


class Neo4jWriter(KGWriter):
    """Writes a knowledge graph to a Neo4j database.

    Args:
        driver (neo4j.driver): The Neo4j driver to connect to the database.
        neo4j_database (Optional[str]): The name of the Neo4j database. If not provided, this defaults to the server's default database ("neo4j" by default) (`see reference to documentation <https://neo4j.com/docs/operations-manual/current/database-administration/#manage-databases-default>`_).
        batch_size (int): The number of nodes or relationships to write to the database in a batch. Defaults to 1000.

    Example:

    .. code-block:: python

        from neo4j import GraphDatabase
        from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
        from neo4j_graphrag.experimental.pipeline import Pipeline

        URI = "neo4j://localhost:7687"
        AUTH = ("neo4j", "password")
        DATABASE = "neo4j"

        driver = GraphDatabase.driver(URI, auth=AUTH)
        writer = Neo4jWriter(driver=driver, neo4j_database=DATABASE)

        pipeline = Pipeline()
        pipeline.add_component(writer, "writer")

    """

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: Optional[str] = None,
        batch_size: int = 1000,
    ):
        self.driver = driver_config.override_user_agent(driver)
        self.neo4j_database = neo4j_database
        self.batch_size = batch_size
        version_tuple, _, _ = get_version(self.driver, self.neo4j_database)
        self.is_version_5_23_or_above = is_version_5_23_or_above(version_tuple)

    def _db_setup(self) -> None:
        # create index on __KGBuilder__.id
        # used when creating the relationships
        self.driver.execute_query(
            "CREATE INDEX __entity__id IF NOT EXISTS  FOR (n:__KGBuilder__) ON (n.id)",
            database_=self.neo4j_database,
        )

    def _ensure_neo4j_compatible(self, value: Any) -> Any:
        """Ensure value is Neo4j-compatible type"""
        if value is None:
            return None

        # First, try to convert to a basic Python type
        try:
            # Handle numpy types
            if hasattr(value, 'item') and hasattr(value, 'dtype'):
                value = value.item()

            # Handle numeric types - be very aggressive with conversion
            if isinstance(value, (int, np.integer, float, np.floating, bool, np.bool_)):
                # For any numeric type, convert to int or float
                try:
                    # First try to convert to float
                    float_val = float(value)
                    # If it's a whole number, convert to int, otherwise keep as float
                    if float_val.is_integer():
                        return int(float_val)
                    return float_val
                except (ValueError, TypeError):
                    # If conversion fails, convert to string
                    return str(value)

            # Handle sequences
            if isinstance(value, (list, tuple)):
                return [self._ensure_neo4j_compatible(x) for x in value]

            # Handle dictionaries
            if isinstance(value, dict):
                return {str(k): self._ensure_neo4j_compatible(v) for k, v in value.items()}

            # Handle strings
            if isinstance(value, str):
                return value

            # For any other type, convert to string
            return str(value)

        except Exception as e:
            # If any error occurs during conversion, log and convert to string
            logger.warning(f"Could not convert value {value} of type {type(value)}: {str(e)}")
            return str(value)

    def _convert_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Convert all property values to Neo4j-compatible types"""
        if not properties:
            return {}

        converted = {}
        for k, v in properties.items():
            try:
                # First convert the value using ensure_neo4j_compatible
                converted_val = self._ensure_neo4j_compatible(v)

                # Force convert numeric values to strings to avoid Long type issues
                if isinstance(converted_val, (int, float, np.integer, np.floating)):
                    converted[k] = str(converted_val)
                # Handle sequences
                elif isinstance(converted_val, (list, tuple)):
                    safe_list = []
                    for item in converted_val:
                        # Convert all numeric items to strings
                        if isinstance(item, (int, float, np.integer, np.floating)):
                            safe_list.append(str(item))
                        elif isinstance(item, (str, bool)) or item is None:
                            safe_list.append(item)
                        else:
                            safe_list.append(str(item))
                    converted[k] = safe_list
                # Handle nested dictionaries
                elif isinstance(converted_val, dict):
                    converted[k] = self._convert_properties(converted_val)
                # Handle other types
                else:
                    converted[k] = str(converted_val)

                # Log the conversion
                logger.debug(f"Converted property '{k}': {type(v)} -> {type(converted[k])} ({converted[k]})")

            except Exception as e:
                logger.warning(f"Error converting property '{k}': {str(e)}")
                converted[k] = str(v)

        return converted

    def _validate_row(self, row: dict) -> None:
        """Validate a row before insertion into Neo4j"""
        if not isinstance(row, dict):
            raise ValueError(f"Row must be a dictionary, got {type(row)}")

        # Check for required fields
        if 'id' not in row or not row['id']:
            raise ValueError("Row is missing required field 'id'")

        # Ensure properties is a dictionary
        if 'properties' not in row or not isinstance(row['properties'], dict):
            row['properties'] = {}

        # Ensure labels is a list
        if 'labels' not in row or not isinstance(row['labels'], list):
            row['labels'] = []

        # Log the row structure for debugging
        logger.debug(f"Validated row with id: {row.get('id')}")
        logger.debug(f"  Labels: {row.get('labels')}")
        logger.debug(f"  Properties: {row.get('properties', {})}")

    def _nodes_to_rows(
        self, nodes: list[Neo4jNode], lexical_graph_config: LexicalGraphConfig
    ) -> list[dict[str, Any]]:
        rows = []
        for node in nodes:
            try:
                # Safely get node attributes with defaults
                node_id = str(getattr(node, 'id', 'unknown'))
                node_label = str(getattr(node, 'label', ''))

                # Initialize labels list with only specific labels (Function, Variable, Script, Chunk)
                labels = []
                if node_label in ['Function', 'Variable', 'Script', 'Chunk']:
                    labels = [node_label]

                # Safely get and convert properties
                properties = {}
                if hasattr(node, 'properties') and node.properties:
                    try:
                        properties = self._convert_properties(dict(node.properties))
                    except Exception as e:
                        logger.warning(f"Error converting properties for node {node_id}: {str(e)}")

                # Create row with all required fields
                row = {
                    "id": node_id,
                    "properties": properties,
                    "labels": labels
                }

                # Handle embedding properties if present
                if hasattr(node, 'embedding_properties') and node.embedding_properties:
                    try:
                        row["embedding_properties"] = self._convert_properties(dict(node.embedding_properties))
                    except Exception as e:
                        logger.warning(f"Error converting embedding properties for node {node_id}: {str(e)}")

                # Validate the row before adding
                self._validate_row(row)
                rows.append(row)

            except Exception as e:
                logger.error(f"Error processing node {getattr(node, 'id', 'unknown')}: {str(e)}")
                logger.error(f"Node data: {str(node.__dict__)[:500]}..." if hasattr(node, '__dict__') else "No node data available")
                continue

        return rows

    def _upsert_nodes(
        self, nodes: list[Neo4jNode], lexical_graph_config: LexicalGraphConfig
    ) -> None:
        """Upserts a single node into the Neo4j database."""
        try:
            # Convert nodes to rows with detailed logging
            rows = self._nodes_to_rows(nodes, lexical_graph_config)

            # Log the first few rows for debugging
            for i, row in enumerate(rows[:3]):  # Log first 3 rows
                logger.debug(f"Processing node row {i+1}/{len(rows)}: {row.get('id')}")
                logger.debug(f"  Labels: {row.get('labels')}")
                logger.debug(f"  Properties: {row.get('properties', {})}")

            parameters = {"rows": rows}

            if self.is_version_5_23_or_above:
                self.driver.execute_query(
                    UPSERT_NODE_QUERY_VARIABLE_SCOPE_CLAUSE,
                    parameters_=parameters,
                    database_=self.neo4j_database,
                )
            else:
                self.driver.execute_query(
                    UPSERT_NODE_QUERY,
                    parameters_=parameters,
                    database_=self.neo4j_database,
                )

        except Exception as e:
            logger.error(f"Error in _upsert_nodes: {str(e)}")
            logger.error(f"Node data that caused the error: {str(rows) if 'rows' in locals() else 'N/A'}")
            raise

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        """Upserts a single relationship into the Neo4j database.

        Args:
            rels (list[Neo4jRelationship]): The relationships batch to upsert into the database.
        """
        rows = []
        for rel in rels:
            try:
                row = {
                    "start_node_id": str(rel.start_node_id),
                    "end_node_id": str(rel.end_node_id),
                    "type": str(rel.type),
                    "properties": self._convert_properties(rel.properties or {})
                }

                # Handle embedding properties if present
                if hasattr(rel, 'embedding_properties') and rel.embedding_properties:
                    row["embedding_properties"] = self._convert_properties(rel.embedding_properties)

                rows.append(row)
            except Exception as e:
                logger.warning(f"Error processing relationship {getattr(rel, 'type', 'unknown')}: {str(e)}")
                continue

        scope_clause = ""
        if not self.is_version_5_23_or_above:
            scope_clause = UPSERT_RELATIONSHIP_QUERY_VARIABLE_SCOPE_CLAUSE

        # Use MERGE instead of CREATE
        query = UPSERT_RELATIONSHIP_QUERY.replace("CREATE", "MERGE")

        for batch in batched(rows, self.batch_size):
            try:
                self.driver.execute_query(
                    query,
                    parameters_={"rows": batch},
                    database_=self.neo4j_database,
                )
            except Exception as e:
                logger.error(f"Error in _upsert_relationships: {str(e)}")
                logger.error(f"Relationship data that caused the error: {str(batch) if 'batch' in locals() else 'N/A'}")
                raise

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """Upserts a knowledge graph into a Neo4j database.

        Args:
            graph (Neo4jGraph): The knowledge graph to upsert into the database.
            lexical_graph_config (LexicalGraphConfig): Node labels and relationship types for the lexical graph.
        """
        try:
            self._db_setup()

            for batch in batched(graph.nodes, self.batch_size):
                self._upsert_nodes(batch, lexical_graph_config)

            for batch in batched(graph.relationships, self.batch_size):
                self._upsert_relationships(batch)

            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "node_count": len(graph.nodes),
                    "relationship_count": len(graph.relationships),
                },
            )
        except neo4j.exceptions.ClientError as e:
            logger.exception(e)
            return KGWriterModel(status="FAILURE", metadata={"error": str(e)})
