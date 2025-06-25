from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
INDEX_NAME = "kg_embeddings"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
    # 1. 检查数据库中的节点
    with driver.session() as session:
        # 检查所有节点
        result = session.run("MATCH (n) RETURN count(n) as count")
        total_count = result.single()["count"]
        print(f"Total nodes in database: {total_count}")
        
        # 检查 Function 节点
        result = session.run("MATCH (n:Function) RETURN count(n) as count")
        func_count = result.single()["count"]
        print(f"Function nodes: {func_count}")
        
        # 检查有 embedding 的 Function 节点
        result = session.run("MATCH (n:Function) WHERE n.embedding IS NOT NULL RETURN count(n) as count")
        embedding_count = result.single()["count"]
        print(f"Function nodes with embeddings: {embedding_count}")
        
        # 检查向量索引
        result = session.run("SHOW INDEXES")
        indexes = list(result)
        print(f"\nAvailable indexes:")
        for idx in indexes:
            print(f"  - {idx['name']}: {idx['type']} on {idx['labelsOrTypes']}")
        
        # 检查一些 Function 节点的属性
        result = session.run("MATCH (n:Function) WHERE n.embedding IS NOT NULL RETURN n.name, n.file_path LIMIT 5")
        functions = list(result)
        print(f"\nSample Function nodes with embeddings:")
        for func in functions:
            print(f"  - name: {func['n.name']}, file_path: {func['n.file_path']}")
        
        # 检查是否有包含 "model" 的 Function 节点
        result = session.run("MATCH (n:Function) WHERE n.name CONTAINS 'model' OR n.name = 'model' RETURN n.name, n.file_path LIMIT 10")
        model_functions = list(result)
        print(f"\nFunction nodes containing 'model':")
        for func in model_functions:
            print(f"  - name: {func['n.name']}, file_path: {func['n.file_path']}")

    # 2. 测试 embedding 生成
    print("\nTesting embedding generation...")
    embedder = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-8B:Q5_K_M")
    
    test_text = "function varargout = model(x,parameters,nvp)"
    embedding = embedder.embed_query(test_text)
    print(f"Generated embedding dimension: {len(embedding)}")
    print(f"Embedding sample: {embedding[:5]}...")

    # 3. 测试向量检索
    print("\nTesting vector retrieval...")
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)
    
    # 直接测试检索
    query_text = "function varargout = model(x,parameters,nvp)"
    print(f"Query text: {query_text}")
    
    try:
        documents = retriever.retrieve(query_text, top_k=5)
        print(f"Retrieved {len(documents)} documents")
        
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(f"  Content: {doc.content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
            print(f"  Score: {getattr(doc, 'score', 'N/A')}")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

    # 4. 测试不同的查询文本
    print("\nTesting different query texts...")
    test_queries = [
        "model function",
        "function model",
        "FinBERT model",
        "bert model",
        "model(x,parameters,nvp)"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            documents = retriever.retrieve(query, top_k=3)
            print(f"  Retrieved {len(documents)} documents")
            if documents:
                for i, doc in enumerate(documents, 1):
                    print(f"    Doc {i}: {doc.metadata.get('file_path', 'No file_path')} - {doc.content[:50]}...")
            else:
                print("    No documents retrieved")
        except Exception as e:
            print(f"    Error: {e}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.close()
    print("\nNeo4j driver closed.") 