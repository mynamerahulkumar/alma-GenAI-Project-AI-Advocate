# RAG Chatbot Workshop - 4 Day Course

A full-stack RAG (Retrieval-Augmented Generation) chatbot built with LangChain, Streamlit, FAISS, and Tavily search.

## 🎯 Course Overview

This project is designed for a **4-day workshop** (2 hours per day) teaching RAG application development.

### Day-by-Day Breakdown

| Day | Topic | Files | Key Concepts |
|-----|-------|-------|--------------|
| **Day 1** | Foundations & Document Processing | `config/`, `core/document_processor.py` | RAG intro, document loaders, text splitting |
| **Day 2** | Vector Store & Embeddings | `core/embeddings.py`, `core/vector_store.py` | Embeddings, FAISS, similarity search |
| **Day 3** | RAG Chain & Tavily Tool | `core/chain.py`, `tools/tavily_search.py` | LLM integration, chains, web search |
| **Day 4** | Streamlit UI & Integration | `ui/`, `app.py` | Chat interface, streaming, deployment |

## 🛠️ Tech Stack

- **LLM**: Groq (FREE - Llama 3.1)
- **Embeddings**: HuggingFace sentence-transformers (FREE - runs locally)
- **Vector Store**: FAISS (FREE - runs locally)
- **Web Search**: Tavily API
- **UI**: Streamlit
- **Framework**: LangChain

## 📁 Project Structure

```
rag-chatbot/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration & API keys
├── core/
│   ├── __init__.py
│   ├── document_processor.py # Document loading & splitting
│   ├── embeddings.py         # HuggingFace embeddings
│   ├── vector_store.py       # FAISS operations
│   └── chain.py              # RAG chain orchestration
├── tools/
│   ├── __init__.py
│   └── tavily_search.py      # Web search integration
├── ui/
│   ├── __init__.py
│   ├── components.py         # Reusable UI components
│   └── chat_interface.py     # Chat logic
├── data/
│   ├── documents/            # Uploaded documents
│   └── faiss_index/          # Persisted vector index
├── app.py                    # Main Streamlit app
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Keys (FREE!)

1. **Groq API Key** (FREE): https://console.groq.com/
2. **Tavily API Key** (FREE tier): https://tavily.com/

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

## 📖 SOLID Principles Applied

This project follows SOLID principles for maintainable code:

- **S**ingle Responsibility: Each module has one job
- **O**pen/Closed: Extensible without modifying existing code
- **L**iskov Substitution: Components can be swapped
- **I**nterface Segregation: Small, focused interfaces
- **D**ependency Inversion: Depend on abstractions

## 🎓 Teaching Notes

### Day 1: Foundations
- Explain RAG architecture (Retrieve → Augment → Generate)
- Walk through `config/settings.py` - environment variables
- Deep dive into `core/document_processor.py` - loaders & splitters
- **Hands-on**: Load and split a sample document

### Day 2: Vector Store
- Explain embeddings (text → vectors)
- Show `core/embeddings.py` - HuggingFace models
- Explain FAISS and similarity search
- Walk through `core/vector_store.py`
- **Hands-on**: Create embeddings and search

### Day 3: RAG Chain
- Explain LLM integration with Groq
- Walk through `core/chain.py` - prompt templates, chains
- Show `tools/tavily_search.py` - web search
- **Hands-on**: Build complete RAG pipeline

### Day 4: UI & Integration
- Explain Streamlit components
- Walk through `ui/components.py` and `ui/chat_interface.py`
- Show `app.py` - putting it all together
- **Hands-on**: Run the complete application

## 📝 License

MIT License - Feel free to use for teaching and learning!

# Sample question
 Personal Liberty: Procedure Established by Law