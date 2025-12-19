
# Alma GenAI Project - AI Advocate

A comprehensive RAG (Retrieval-Augmented Generation) system demonstrating document processing, embeddings, vector stores, and AI-powered question answering.

**Pipeline Flow**: Document Process → Embeddings → Vector Store → RAG Chain

---

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: Required for downloading models and dependencies

---

## 🚀 Setup Instructions

### Option 1: Setup with UV (Recommended - Fast!)

UV is a modern, fast Python package manager written in Rust. It's significantly faster than pip.

#### Windows Setup with UV

1. **Install UV**
   ```powershell
   # Using PowerShell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the Repository**
   ```powershell
   git clone <repository-url>
   cd alma-GenAI-Project-AI-Advocate
   ```

3. **Create Virtual Environment and Install Dependencies**
   ```powershell
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   .venv\Scripts\activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```powershell
   # Create .env file
   copy .env.example .env
   # Edit .env and add your API keys
   notepad .env
   ```

#### macOS/Linux Setup with UV

1. **Install UV**
   ```bash
   # Using curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or using Homebrew (macOS)
   brew install uv
   ```

2. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd alma-GenAI-Project-AI-Advocate
   ```

3. **Create Virtual Environment and Install Dependencies**
   ```bash
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env and add your API keys
   nano .env
   ```

---

### Option 2: Setup with Standard Python

#### Windows Setup with Python

1. **Install Python**
   - Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```powershell
     python --version
     pip --version
     ```

2. **Clone the Repository**
   ```powershell
   git clone <repository-url>
   cd alma-GenAI-Project-AI-Advocate
   ```

3. **Create Virtual Environment**
   ```powershell
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   .\venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**
   ```powershell
   # Create .env file
   copy .env.example .env
   # Edit .env and add your API keys
   notepad .env
   ```

#### macOS/Linux Setup with Python

1. **Install Python**
   ```bash
   # macOS (using Homebrew)
   brew install python@3.10
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3-pip
   
   # Verify installation
   python3 --version
   pip3 --version
   ```

2. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd alma-GenAI-Project-AI-Advocate
   ```

3. **Create Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env and add your API keys
   nano .env
   ```

---

## 🔑 Environment Variables

Create a `.env` file in the project root with the following:

```env
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Getting API Keys

- **GROQ API Key**: Sign up at [https://console.groq.com](https://console.groq.com)
- **Tavily API Key**: Sign up at [https://tavily.com](https://tavily.com)

---

## 🧪 Testing the APIs - Demo Scripts

The project includes two demo scripts to test different components:

### Demo Day 1: Document Processing

Tests document loading and chunking functionality.

**Windows:**
```powershell
python demo_day1.py
```

**macOS/Linux:**
```bash
python demo_day1.py
```

**Expected Output:**
- Creates a sample document
- Loads and processes the document
- Splits content into chunks
- Displays chunk details (content, metadata, token count)

---

### Demo Day 2: Embeddings & Vector Store

Tests embeddings creation and vector store operations.

**Windows:**
```powershell
python demo_day2.py
```

**macOS/Linux:**
```bash
python demo_day2.py
```

**Expected Output:**
- Creates sample documents
- Generates embeddings using HuggingFace models
- Builds FAISS vector store
- Performs similarity search
- Displays search results with scores

**Note**: The first run will download the embedding model (~80MB), which may take a minute.

---

### Demo Day 3: Complete RAG Pipeline (If Available)

**Windows:**
```powershell
python demo_day3.py
```

**macOS/Linux:**
```bash
python demo_day3.py
```

---

## 🏃‍♂️ Running the Main Application

**Windows:**
```powershell
python main.py
```

**macOS/Linux:**
```bash
python main.py
```

---

## 📦 Project Structure

```
alma-GenAI-Project-AI-Advocate/
├── config/              # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── core/                # Core functionality
│   ├── __init__.py
│   ├── document_processor.py  # Document loading & chunking
│   ├── embeddings.py          # Embedding generation
│   └── vector_store.py        # Vector store operations
├── data/                # Data storage
│   └── faiss_index/     # FAISS vector indices
├── utils/               # Utility functions
├── demo_day1.py         # Document processing demo
├── demo_day2.py         # Embeddings & vector store demo
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
└── README.md            # This file
```

---

## 🔍 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` for packages
- **Solution**: Ensure virtual environment is activated and dependencies are installed
  ```bash
  pip install -r requirements.txt
  ```

**Issue**: FAISS import error on Windows
- **Solution**: Install Visual C++ redistributables or use `faiss-cpu` instead

**Issue**: Slow embedding generation
- **Solution**: First run downloads models; subsequent runs will be faster

**Issue**: API key errors
- **Solution**: Verify `.env` file exists with correct API keys

---

## 📚 Key Dependencies

- **LangChain**: Framework for LLM applications
- **GROQ**: Fast LLM inference API (Free tier available)
- **HuggingFace Sentence Transformers**: Free embedding models
- **FAISS**: Efficient similarity search and vector storage
- **Tavily**: Web search API for RAG
- **Streamlit**: Web UI framework

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## 📄 License

This project is part of the Alma GenAI program.

---

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review demo script outputs for errors
3. Verify all dependencies are installed correctly
4. Ensure API keys are properly configured

---

**Happy Coding! 🚀**
