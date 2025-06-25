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

# Use Ollama for embeddings and LLM
embedder = OllamaEmbeddings(model="nomic-embed-text:latest")
llm = OllamaLLM(model_name="deepseek-r1:14b", model_params={"temperature": 0})

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Instantiate the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph: change to a MATLAB KG-relevant question
query_text = "哪些脚本与tests/matlab_test/test_data/PredictMaskedTokensUsingBERT.m有关系？"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)
driver.close()