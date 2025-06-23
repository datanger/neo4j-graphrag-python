#!/usr/bin/env python3
from neo4j import GraphDatabase, exceptions
from typing import List, Dict, Any, Optional
import argparse
import sys
import os

def get_default_neo4j_uri():
    """Get the default Neo4j URI based on common configurations."""
    # Try to get from environment variable
    uri = os.environ.get('NEO4J_URI')
    if uri:
        return uri
    
    # Common Neo4j Desktop URI
    return 'bolt://localhost:7687'

def get_default_username():
    """Get the default Neo4j username."""
    return os.environ.get('NEO4J_USER', 'neo4j')

def get_default_password():
    """Get the default Neo4j password from environment variable if set."""
    return os.environ.get('NEO4J_PASSWORD')

class Neo4jConnectionError(Exception):
    """Custom exception for Neo4j connection errors."""
    pass

class Neo4jQueries:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def test_connection(self) -> bool:
        """Test the Neo4j connection."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
                return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def run_query(self, query: str, **parameters) -> List[Dict[str, Any]]:
        """Run a query and return the results as a list of dictionaries."""
        try:
            with self.driver.session() as session:
                result = session.run(query, **parameters)
                return [dict(record) for record in result]
        except exceptions.Neo4jError as e:
            print(f"Neo4j error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

def main():
    # Get default values
    default_uri = get_default_neo4j_uri()
    default_user = get_default_username()
    default_password = get_default_password()
    
    parser = argparse.ArgumentParser(description='Run Neo4j queries from command line')
    parser.add_argument('--uri', default=default_uri, 
                        help=f'Neo4j URI (default: {default_uri})')
    parser.add_argument('--user', default="neo4j", 
                        help=f'Neo4j username (default: {default_user})')
    parser.add_argument('--password', default="password", 
                        help='Neo4j password (default: from NEO4J_PASSWORD env var if set)')
    parser.add_argument('--query', type=int, choices=range(1, 6), 
                        help='Query number to run (1-5)')
    
    args = parser.parse_args()
    
    # Validate connection parameters
    if not args.password:
        print("Error: Password is required. Set NEO4J_PASSWORD environment variable or use --password")
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Create Neo4j connection
    try:
        print(f"Connecting to Neo4j at {args.uri} as user '{args.user}'...")
        neo4j = Neo4jQueries(args.uri, args.user, args.password)
        
        # Test connection
        if not neo4j.test_connection():
            print("Failed to connect to Neo4j. Please check your credentials and that Neo4j is running.")
            print("Common issues:")
            print("1. Neo4j Desktop: Make sure the database is started")
            print("2. Check your username/password")
            print("3. Verify the bolt port (default: 7687)")
            print("4. Check if Neo4j is configured to accept remote connections if needed")
            sys.exit(1)
            
        print("Successfully connected to Neo4j!")
        
    except exceptions.ServiceUnavailable as e:
        print(f"Neo4j service unavailable: {e}")
        print("Is Neo4j running? Try starting it with 'neo4j start'")
        sys.exit(1)
    except exceptions.AuthError as e:
        print(f"Authentication failed: {e}")
        print("Please check your username and password.")
        sys.exit(1)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)
    
    queries = {
        1: """
        MATCH (src)-[r1:DEFINES]->(v:Variable)<-[r2:MODIFIES]-(src)
        RETURN src.name AS source, v.name AS variable, r1.line AS def_line, r2.line AS mod_line
        """,
        
        2: """
        MATCH (src)-[r:DEFINES|MODIFIES]->(v:Variable)
        RETURN 
          src.name AS source, 
          type(r) AS relationship_type, 
          v.name AS variable,
          r.line AS line_number,
          r.scope_type AS scope_type
        ORDER BY source, relationship_type, variable
        """,
        
        3: """
        MATCH (definer)-[:DEFINES]->(v:Variable)<-[:MODIFIES]-(modifier)
        WHERE definer <> modifier
        RETURN 
          definer.name AS defined_in,
          modifier.name AS modified_in,
          v.name AS variable,
          modifier.scope_type AS modifier_scope
        """,
        
        4: """
        MATCH (definer)-[:DEFINES]->(v:Variable)
        WITH v, collect(definer) AS definers
        MATCH (user)-[r:USES]->(v)
        WHERE NOT user IN definers AND NOT (user)-[:MODIFIES]->(v)
        RETURN 
          user.name AS user,
          v.name AS variable,
          [d IN definers | d.name] AS defined_in
        """,
        
        5: """
        MATCH (n)-[r]->(m)
        WHERE ANY(label IN labels(n) WHERE label IN ['Function', 'Script', 'Variable'])
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        LIMIT 100
        """
    }
    
    try:
        if args.query:
            # Run a specific query
            query_num = args.query
            print(f"\n=== Running Query {query_num} ===\n")
            print(f"Query: {queries[query_num].strip()}\n")
            
            try:
                result = neo4j.run_query(queries[query_num])
                if result:
                    # Print column headers
                    if result:
                        headers = list(result[0].keys())
                        print(" | ".join(headers))
                        print("-" * 50)
                        
                    # Print rows
                    for row in result:
                        print(" | ".join(str(row.get(k, "")) for k in headers))
                    
                    print(f"\nFound {len(result)} result(s)")
                else:
                    print("No results found.")
                    
            except Exception as e:
                print(f"Error running query: {e}")
                
        else:
            # Run all queries
            for query_num, query in queries.items():
                print(f"\n=== Query {query_num} ===\n")
                print(f"{query.strip()}\n")
                
                try:
                    result = neo4j.run_query(query)
                    if result:
                        # Print column headers
                        headers = list(result[0].keys())
                        print(" | ".join(headers))
                        print("-" * 50)
                        
                        # Print first 5 rows
                        for i, row in enumerate(result[:5]):
                            print(" | ".join(str(row.get(k, "")) for k in headers))
                            
                        if len(result) > 5:
                            print(f"... and {len(result) - 5} more rows")
                            
                        print(f"\nFound {len(result)} result(s)")
                    else:
                        print("No results found.")
                        
                except Exception as e:
                    print(f"Error running query: {e}")
                    
    finally:
        neo4j.close()
        print("\nConnection closed.")

def query_cross_scope_relationships():
    """查询Neo4j数据库中的跨作用域变量调用关系"""
    
    # 连接Neo4j数据库
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    try:
        with driver.session(database="neo4j") as session:
            
            print("=== 查询跨作用域变量调用关系 ===\n")
            
            # 1. 查询所有跨作用域的USES关系
            print("1. 跨作用域USES关系:")
            result = session.run("""
                MATCH (s)-[r:USES]->(v:Variable) 
                WHERE s.scope_id <> v.scope_id 
                RETURN s.name as source_name, s.scope_id as source_scope, 
                       v.name as var_name, v.scope_id as var_scope, 
                       r.usage_type as usage_type, r.post_processed as post_processed
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['source_name']} ({record['source_scope']}) -> {record['var_name']} ({record['var_scope']})")
                print(f"    使用类型: {record['usage_type']}, 后处理: {record['post_processed']}")
                print()
            
            # 2. 查询基于执行顺序的跨作用域关系
            print("\n2. 基于执行顺序的跨作用域关系:")
            result = session.run("""
                MATCH (s)-[r:USES]->(v:Variable) 
                WHERE r.usage_type = 'cross_scope_execution_order'
                RETURN s.name as source_name, s.scope_id as source_scope, 
                       v.name as var_name, v.scope_id as var_scope,
                       r.source_script as source_script, r.target_script as target_script,
                       r.execution_order as exec_order, r.source_execution_order as source_exec_order
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['source_name']} -> {record['var_name']}")
                print(f"    源脚本: {record['source_script']}, 目标脚本: {record['target_script']}")
                print(f"    执行顺序: {record['source_execution_order']} -> {record['exec_order']}")
                print()
            
            # 3. 查询脚本间的CALLS关系
            print("\n3. 脚本间CALLS关系:")
            result = session.run("""
                MATCH (s1:Script)-[r:CALLS]->(s2:Script)
                RETURN s1.name as caller, s2.name as callee, 
                       r.call_type as call_type, r.post_processed as post_processed
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['caller']} -> {record['callee']}")
                print(f"    调用类型: {record['call_type']}, 后处理: {record['post_processed']}")
                print()
            
            # 4. 查询函数间的CALLS关系
            print("\n4. 函数间CALLS关系:")
            result = session.run("""
                MATCH (f1:Function)-[r:CALLS]->(f2:Function)
                RETURN f1.name as caller, f2.name as callee, 
                       r.call_type as call_type, r.post_processed as post_processed
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['caller']} -> {record['callee']}")
                print(f"    调用类型: {record['call_type']}, 后处理: {record['post_processed']}")
                print()
            
            # 5. 查询脚本调用函数的关系
            print("\n5. 脚本调用函数的关系:")
            result = session.run("""
                MATCH (s:Script)-[r:CALLS]->(f:Function)
                RETURN s.name as script, f.name as function, 
                       r.call_type as call_type, r.post_processed as post_processed
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['script']} -> {record['function']}")
                print(f"    调用类型: {record['call_type']}, 后处理: {record['post_processed']}")
                print()
            
            # 6. 统计各种关系类型的数量
            print("\n6. 关系类型统计:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            
            for record in result:
                print(f"  {record['rel_type']}: {record['count']}")
            
            # 7. 查询变量定义和使用统计
            print("\n7. 变量定义和使用统计:")
            result = session.run("""
                MATCH (v:Variable)
                RETURN v.scope_type as scope_type, count(v) as var_count
                ORDER BY var_count DESC
            """)
            
            for record in result:
                print(f"  {record['scope_type']} 变量: {record['var_count']}")
            
    finally:
        driver.close()

if __name__ == "__main__":
    query_cross_scope_relationships()
