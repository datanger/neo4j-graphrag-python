from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.retrievers import VectorRetriever

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
INDEX_NAME = "kg_embeddings"  # match your pipeline index name

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
    # 首先测试数据库连接
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"Connected to Neo4j. Total nodes in database: {count}")
        
        # 检查 CodeElement 节点（统一索引）
        result = session.run("MATCH (n:CodeElement) RETURN count(n) as count")
        code_element_count = result.single()["count"]
        print(f"CodeElement nodes in database: {code_element_count}")
        
        # 检查是否有带 embedding 的 CodeElement 节点
        result = session.run("MATCH (n:CodeElement) WHERE n.embedding IS NOT NULL RETURN count(n) as count")
        embedding_count = result.single()["count"]
        print(f"CodeElement nodes with embeddings: {embedding_count}")
        
        # 检查向量索引
        result = session.run("""
            SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
            WHERE type = 'VECTOR' AND name = $index_name
            RETURN name, labelsOrTypes, properties, options
        """, index_name=INDEX_NAME)
        index_info = result.single()
        if index_info:
            print(f"✓ Vector index '{INDEX_NAME}' found:")
            print(f"  - Labels: {index_info['labelsOrTypes']}")
            print(f"  - Properties: {index_info['properties']}")
            print(f"  - Dimensions: {index_info['options'].get('indexConfig', {}).get('vector.dimensions', 'N/A')}")
        else:
            print(f"✗ Vector index '{INDEX_NAME}' not found!")

    # Use Ollama for embeddings and LLM
    print("\nInitializing embedding model...")
    embedder = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-8B:Q5_K_M")
    
    print("Initializing LLM...")
    llm = OllamaLLM(model_name="deepseek-r1:14b", model_params={"temperature": 0})

    # Initialize the retriever with better configuration
    print("Initializing vector retriever...")
    retriever = VectorRetriever(
        driver, 
        INDEX_NAME, 
        embedder,
        return_properties=["name", "file_path", "code_snippet", "start_line", "end_line"],
        neo4j_database="neo4j"
    )

    # Instantiate the RAG pipeline
    print("Initializing GraphRAG...")
    rag = GraphRAG(retriever=retriever, llm=llm)

    # Query the graph: change to a MATLAB KG-relevant question
    query_text = """
    %   Z = model(X,parameters,'PARAM1', VAL1, 'PARAM2', VAL2, ...) specifies
    %   the optional parame..."
    """
    
    print(f"\nExecuting query: {query_text.strip()}")
    print("=" * 50)
    
    # 先测试向量检索，看看能检索到什么
    print("\n=== Testing Vector Retrieval ===")
    try:
        # 直接测试检索器
        query_embedding = embedder.embed_query("function varargout = model")
        print(f"Generated query embedding of length: {len(query_embedding)}")
        
        # 执行向量搜索
        raw_result = retriever.get_search_results(
            query_text="function varargout = model",
            top_k=10,
            effective_search_ratio=2
        )
        
        print(f"Raw search returned {len(raw_result.records)} records")
        
        # 显示检索到的记录
        for i, record in enumerate(raw_result.records[:5], 1):
            node = record.get("node")
            score = record.get("score")
            print(f"\nRecord {i} (score: {score:.4f}):")
            if node:
                print(f"  Labels: {list(node.labels)}")
                print(f"  Name: {node.get('name', 'N/A')}")
                print(f"  File path: {node.get('file_path', 'N/A')}")
                print(f"  Start line: {node.get('start_line', 'N/A')}")
                print(f"  Code preview: {str(node.get('code', ''))[:100]}...")
        
    except Exception as e:
        print(f"Error in vector retrieval test: {e}")
        import traceback
        traceback.print_exc()
    
    # 执行 RAG 搜索
    print("\n=== Executing RAG Search ===")
    response = rag.search(
        query_text=query_text, 
        retriever_config={"top_k": 10, "effective_search_ratio": 2},
        return_context=True  # 返回检索上下文以便调试
    )
    
    print("\nQuery Response:")
    print("=" * 50)
    print(response.answer)
    
    # 显示检索到的文档详情
    if hasattr(response, 'retriever_result') and response.retriever_result:
        print(f"\n=== Retrieved Documents ({len(response.retriever_result.items)} items) ===")
        for i, item in enumerate(response.retriever_result.items, 1):
            print(f"\nDocument {i}:")
            print(f"  Content: {item.content[:200]}...")
            if hasattr(item, 'metadata') and item.metadata:
                print(f"  Metadata: {item.metadata}")
                if 'file_path' in item.metadata:
                    print(f"  File path: {item.metadata['file_path']}")
    else:
        print("\nNo documents retrieved!")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Close the driver when done
    driver.close()
    print("\nNeo4j driver closed.")

