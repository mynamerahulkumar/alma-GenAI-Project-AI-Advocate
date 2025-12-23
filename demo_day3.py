"""
Demo Script - Day 3: RAG Chain & Web Search
============================================
This script demonstrates the complete RAG pipeline and web search integration.

Run with: python demo_day3.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.documents import Document
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from tools.tavily_search import TavilySearchTool


def main():
    print("\n" + "=" * 70)
    print("🎓 DAY 3 DEMO: RAG Chain & Web Search")
    print("=" * 70)
    
    # Step 1: Create sample knowledge base
    print("\n📚 Step 1: Creating knowledge base...")
    sample_docs = [
        Document(
            page_content="Python is a high-level programming language created by Guido van Rossum in 1991. "
                        "It emphasizes code readability and has a comprehensive standard library.",
            metadata={"source": "python_intro.txt"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn "
                        "and improve from experience without being explicitly programmed.",
            metadata={"source": "ml_basics.txt"}
        ),
        Document(
            page_content="Deep learning uses artificial neural networks with multiple layers to process data "
                        "and is particularly effective for image recognition and natural language processing.",
            metadata={"source": "deep_learning.txt"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
                        "It enables data connection, agent logic, and integration with various tools.",
            metadata={"source": "langchain_overview.txt"}
        ),
    ]
    
    print(f"✅ Created {len(sample_docs)} documents")
    
    # Step 2: Setup vector store
    print("\n📊 Step 2: Setting up vector store...")
    embedder = EmbeddingManager()
    vs_manager = VectorStoreManager(embedder)
    vs_manager.create_from_documents(sample_docs)
    print("✅ Vector store created and indexed")
    
    # Step 3: Create RAG chain
    print("\n⚙️  Step 3: Initializing RAG chain...")
    rag = RAGChain(vs_manager)
    print(f"✅ RAG chain ready")
    print(f"   LLM Model: {rag.model_name}")
    print(f"   Temperature: {rag.temperature}")
    
    # Step 4: Test document-only queries
    print("\n📄 Step 4: Testing document-only queries...")
    doc_queries = [
        "What is Python?",
        "Explain machine learning",
        "What is LangChain?"
    ]
    
    for i, query in enumerate(doc_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        result = rag.query(query)
        print(f"   Answer: {result['answer'][:150]}...")
        print(f"   Sources: {result['sources']}")
    
    # Step 5: Test web search tool
    print("\n🌐 Step 5: Testing web search tool...")
    search_tool = TavilySearchTool(max_results=3)
    print("✅ Tavily search tool initialized")
    print("   (Ready to search: 'NVIDIA stock price', 'Latest AI news', etc.)")
    
    # Step 6: Test streaming
    print("\n⚡ Step 6: Testing streaming response...")
    query = "Tell me about Python"
    print(f"\n   Query: '{query}'")
    print("\n   Streaming response:")
    print("   ", end="", flush=True)
    
    for chunk in rag.query_stream(query):
        print(chunk, end="", flush=True)
    
    print("\n\n   ✅ Streaming complete")
    
    # Statistics
    print("\n" + "=" * 70)
    print("📊 STATISTICS")
    print("=" * 70)
    print(f"Documents in knowledge base: {len(sample_docs)}")
    print(f"LLM Model: {rag.model_name}")
    print(f"Top-K retrieval: {rag.retrieve('test', k=3).__len__()}")
    print(f"Web search available: Yes")
    
    print("\n" + "=" * 70)
    print("🎉 DAY 3 DEMO COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ RAG retrieves relevant documents from your knowledge base")
    print("✓ LLM generates answers augmented with retrieved context")
    print("✓ Responses are streamed for better UX")
    print("✓ Web search can augment document search")
    print("✓ Sources are tracked for transparency")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
