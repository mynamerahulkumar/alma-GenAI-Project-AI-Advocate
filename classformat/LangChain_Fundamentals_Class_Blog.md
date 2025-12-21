# 🔗 LangChain Fundamentals: Building AI Applications with RAG

> A comprehensive 2-hour hands-on session on LangChain and Retrieval-Augmented Generation (RAG)

---

## 📋 Session Flow

### Learning Objective
1.1. Theme  
1.2. Primary Goals

### LangChain Fundamentals
2.1. Getting Started with LangChain & Groq API  
2.2. Understanding Messages in LangChain  
2.3. Prompt Templates for Reusable Prompts  
2.4. LCEL Chains - Composing with the Pipe Operator  
2.5. Output Parsers for Structured Responses  
2.6. Streaming & Batching for Efficiency  
2.7. Tools & Function Calling  
2.8. **Activity 1:** Try it Yourself Quiz

### RAG (Retrieval-Augmented Generation) Fundamentals
3.1. Introduction to RAG Architecture  
3.2. Document Loaders - Loading Your Data  
3.3. Text Splitters - Chunking for Context  
3.4. Embeddings - Converting Text to Vectors  
3.5. Vector Stores with FAISS  
3.6. Retrievers - Finding Relevant Documents  
3.7. Building Complete RAG Chains  
3.8. Source Attribution & Streaming RAG  
3.9. **Activity 2:** True or False Quiz

### Summary & Next Steps
4.1. What Did We Learn?  
4.2. Shortcomings & Challenges  
4.3. Explore Further  
4.4. Enhance Your Knowledge  
4.5. Try it Yourself Tasks

---

## 🎯 Learning Objective

In this session, you will learn the **fundamentals of LangChain**, a powerful framework for building applications with Large Language Models (LLMs). We'll start with core concepts like messages, prompt templates, and chains, then progress to building a complete **Retrieval-Augmented Generation (RAG)** system that can answer questions from your own documents.

**Focus:**
- LangChain core components (Messages, Prompts, Chains)
- LangChain Expression Language (LCEL)
- Output parsing and structured responses
- Document processing and embeddings
- Vector stores and similarity search
- Building end-to-end RAG pipelines

**Duration:** 2 Hours

**Pre-requisites:**
- Basic Python programming knowledge
- Understanding of APIs and HTTP requests
- Familiarity with basic ML/AI concepts (helpful but not required)

---

## 1.1 🌟 Theme

### Real-World AI Applications with LangChain

Companies like **OpenAI, Google, Microsoft, and Amazon** are leveraging LLM-powered applications to transform how we interact with information. From customer support chatbots to intelligent document search, these applications rely on frameworks like **LangChain** to orchestrate complex AI workflows.

**Real-World Examples:**

| Company | Application | LangChain Use Case |
|---------|-------------|-------------------|
| Legal Tech Firms | Document Analysis | RAG for case law search |
| Healthcare | Medical Q&A | RAG with medical literature |
| E-commerce | Customer Support | Chatbots with product knowledge |
| Finance | Research Assistants | RAG with financial reports |

In this session, we'll build a **Legal Q&A System** using landmark Supreme Court judgments - demonstrating how RAG enables AI to answer questions from domain-specific documents accurately.

**Why LangChain?**
- 🔗 **Composability**: Chain components together easily
- 🚀 **Speed**: Get from prototype to production quickly
- 🛠️ **Flexibility**: Works with any LLM provider
- 📚 **RAG Support**: Built-in document processing and retrieval

---

## 1.2 🎯 Primary Goals

By the end of this session, you will be able to:

- ✅ **Initialize** and configure LangChain with the Groq API
- ✅ **Construct** conversations using SystemMessage, HumanMessage, and AIMessage
- ✅ **Create** reusable prompt templates with dynamic variables
- ✅ **Build** chains using LCEL (LangChain Expression Language) pipe operator
- ✅ **Parse** LLM outputs into structured data using Pydantic models
- ✅ **Load** and split documents for processing
- ✅ **Generate** embeddings and store them in vector databases
- ✅ **Implement** a complete RAG pipeline for document Q&A
- ✅ **Add** source attribution to RAG responses

---

## 2. 🔗 LangChain Fundamentals

### 2.1 Getting Started with LangChain & Groq API

LangChain is a framework for developing applications powered by language models. We'll use **Groq** - a free, fast LLM provider - for this session.

**Step 1: Get Your Free API Key**
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up and create a new API key
3. Add it to your `.env` file:

```
GROQ_API_KEY=your-api-key-here
```

**Step 2: Install Dependencies**

```python
# Install required packages
%pip install langchain langchain-groq python-dotenv -q
```

**Step 3: Initialize the LLM**

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize ChatGroq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Free, fast, powerful
    temperature=0,                     # 0=deterministic, 1=creative
    max_tokens=1024,
)

# Test the connection
response = llm.invoke("What is LangChain in one sentence?")
print(response.content)
```

**Available Groq Models:**

| Model | Speed | Best For |
|-------|-------|----------|
| `llama-3.3-70b-versatile` | 280 tok/s | General purpose, high quality |
| `llama-3.1-8b-instant` | 560 tok/s | Fast responses, simpler tasks |
| `mixtral-8x7b-32768` | 450 tok/s | Complex reasoning |

---

### 2.2 Understanding Messages in LangChain

LangChain uses **message types** to structure conversations with LLMs:

| Message Type | Purpose | Example |
|--------------|---------|---------|
| `SystemMessage` | Sets AI behavior/role | "You are a helpful Python tutor" |
| `HumanMessage` | User's input | "Explain list comprehension" |
| `AIMessage` | AI's response (for history) | Previous assistant responses |

**Example: Multi-turn Conversation**

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Create a conversation with context
conversation = [
    SystemMessage(content="You are a friendly AI assistant."),
    HumanMessage(content="My name is Alex."),
    AIMessage(content="Nice to meet you, Alex! How can I help?"),
    HumanMessage(content="What's my name?"),  # Testing memory
]

response = llm.invoke(conversation)
print(response.content)  # Output: "Your name is Alex!"
```

**Key Insight:** The LLM doesn't have memory by default. We maintain context by passing the full conversation history with each request.

---

### 2.3 Prompt Templates for Reusable Prompts

**Prompt Templates** allow you to create reusable prompts with variables - essential for building scalable LLM applications.

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a reusable template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}. Be concise."),
    ("human", "{question}"),
])

# Use with different inputs
formatted = prompt.invoke({
    "topic": "Machine Learning",
    "question": "What is gradient descent?"
})

response = llm.invoke(formatted)
print(response.content)
```

**Benefits:**
- 🔄 Reusable across different inputs
- 📝 Separates prompt logic from business logic
- ✅ Easy to test and iterate

---

### 2.4 LCEL Chains - Composing with the Pipe Operator

**LCEL (LangChain Expression Language)** lets you compose components using the **pipe operator `|`** - just like Unix pipes!

```
┌────────┐    ┌─────┐    ┌────────────────┐    ┌────────┐
│ Input  │───▶│Prompt│───▶│      LLM      │───▶│ Output │
└────────┘    └─────┘    └────────────────┘    └────────┘
```

**Building a Chain:**

```python
from langchain_core.output_parsers import StrOutputParser

# Create components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    ("human", "Explain {concept} in simple terms."),
])

# Build chain with LCEL
chain = prompt | llm | StrOutputParser()

# Use the chain
result = chain.invoke({"concept": "neural networks"})
print(result)
```

**LCEL Benefits:**
- 🔗 Composable: Chain any compatible components
- 🔄 Reusable: Call with different inputs
- ⚡ Supports streaming and batching automatically

---

### 2.5 Output Parsers for Structured Responses

Transform unstructured LLM text into **structured data** using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List

# Define the structure you want
class MovieRecommendation(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    reason: str = Field(description="Why recommended")

# Get structured output
structured_llm = llm.with_structured_output(MovieRecommendation)
response = structured_llm.invoke("Recommend a sci-fi movie from the 2010s")

# Access as object properties
print(f"Title: {response.title}")
print(f"Year: {response.year}")
print(f"Genre: {response.genre}")
```

**Why Structured Output?**
- ✅ Type-safe responses
- ✅ Easy to integrate with downstream systems
- ✅ Consistent data format

---

### 2.6 Streaming & Batching for Efficiency

**Streaming** - Get tokens as they're generated (better UX):

```python
# Stream response token by token
for chunk in llm.stream("Write a haiku about coding"):
    print(chunk.content, end="", flush=True)
```

**Batching** - Process multiple inputs in parallel:

```python
# Process multiple prompts efficiently
prompts = ["What is Python?", "What is JavaScript?", "What is Rust?"]
responses = llm.batch(prompts)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}\nA: {response.content[:100]}...\n")
```

---

### 2.7 Tools & Function Calling

**Tools** allow LLMs to call external functions - the foundation for building **AI agents**:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "☀️ Sunny, 72°F",
        "london": "🌧️ Rainy, 55°F",
        "tokyo": "⛅ Cloudy, 68°F",
    }
    return weather_data.get(city.lower(), f"No data for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return f"Result: {eval(expression)}"

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, calculate])

# LLM decides when to call tools
response = llm_with_tools.invoke("What's the weather in Tokyo?")
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = get_weather.invoke(tool_call['args'])
    print(result)  # ⛅ Cloudy, 68°F
```

---

### 2.8 📝 Activity 1: Try it Yourself Quiz

Fill in the blanks with the correct option:

1. ________________ (SystemMessage / HumanMessage) is used to set the behavior and role of the AI assistant.

2. The ________________ (`|` / `&`) operator is used in LCEL to chain components together.

3. ________________ (temperature=0 / temperature=1) makes the LLM responses more deterministic and consistent.

4. ________________ (StrOutputParser / Pydantic models) are used with `with_structured_output()` to get type-safe responses.

5. The ________________ (`.stream()` / `.batch()`) method processes multiple inputs in parallel for efficiency.

6. The ________________ (`@tool` / `@chain`) decorator is used to create functions that LLMs can call.

7. ________________ (ChatPromptTemplate / StrOutputParser) allows creating reusable prompts with variables.

<details>
<summary><strong>📋 Click to reveal answers</strong></summary>

1. **SystemMessage** - Sets the AI's behavior and role
2. **`|`** (pipe operator) - Chains components in LCEL
3. **temperature=0** - Makes responses deterministic
4. **Pydantic models** - Used for structured output
5. **`.batch()`** - Processes multiple inputs in parallel
6. **`@tool`** - Decorator for creating callable functions
7. **ChatPromptTemplate** - Creates reusable prompts with variables

</details>

---

## 3. 🔍 RAG (Retrieval-Augmented Generation) Fundamentals

### 3.1 Introduction to RAG Architecture

**RAG** combines **retrieval** (finding relevant documents) with **generation** (LLM response). Instead of relying only on the LLM's training data, we provide context from our own documents.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  PDF/Docs   │───▶│   Chunks    │───▶│  Embeddings │              │
│  │  (Source)   │    │  (Split)    │    │  (Vectors)  │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│                                               │                     │
│                                               ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Answer    │◀───│     LLM     │◀───│Vector Store │              │
│  │ (Response)  │    │ (Generate)  │    │   (FAISS)   │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│         ▲                 ▲                   │                     │
│         │                 │                   ▼                     │
│         │           ┌─────────────┐    ┌─────────────┐              │
│         └───────────│   Prompt    │◀───│  Retriever  │◀── Query     │
│                     │  (Context)  │    │  (Search)   │              │
│                     └─────────────┘    └─────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why RAG?**
- 📚 Access to custom/private data
- 🎯 More accurate, grounded responses
- 🔄 Easy to update knowledge (just add documents)
- 💰 More cost-effective than fine-tuning

---

### 3.2 Document Loaders - Loading Your Data

Document loaders bring data from various sources into LangChain's `Document` format:

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF document
pdf_path = "data/supreme_court_judgments.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"📄 Loaded {len(documents)} pages")
print(f"First page preview: {documents[0].page_content[:200]}...")
print(f"Metadata: {documents[0].metadata}")
```

**Each Document Contains:**
- `page_content`: The text content
- `metadata`: Source info (page number, file name, etc.)

**Supported Loaders:**
| Loader | Source |
|--------|--------|
| `PyPDFLoader` | PDF files |
| `TextLoader` | Text files |
| `WebBaseLoader` | Web pages |
| `CSVLoader` | CSV files |

---

### 3.3 Text Splitters - Chunking for Context

LLMs have **context length limits**. We split documents into smaller chunks that fit within the model's context window while maintaining semantic coherence.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Max characters per chunk
    chunk_overlap=200,     # Overlap for context continuity
    separators=["\n\n", "\n", ". ", " ", ""]  # Split priorities
)

chunks = text_splitter.split_documents(documents)
print(f"📦 Split {len(documents)} pages into {len(chunks)} chunks")
```

**Key Parameters:**

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `chunk_size` | Maximum chunk length | 500-2000 chars |
| `chunk_overlap` | Shared content between chunks | 10-20% of chunk_size |
| `separators` | Priority order for splitting | Paragraphs → Sentences → Words |

---

### 3.4 Embeddings - Converting Text to Vectors

**Embeddings** convert text into numerical vectors that capture semantic meaning. Similar texts have similar vectors.

```
"The court ruled in favor"  → [0.23, -0.45, 0.67, ...]
"The judgment was positive" → [0.21, -0.43, 0.65, ...]  ← Similar!
```

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
)

# Test embedding
sample_text = "The Supreme Court of India"
vector = embeddings.embed_query(sample_text)
print(f"📐 Embedding dimension: {len(vector)}")  # 384
```

**Popular Embedding Models:**

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better |
| `text-embedding-3-small` (OpenAI) | 1536 | Fast | Best |

---

### 3.5 Vector Stores with FAISS

**Vector stores** are databases optimized for storing and searching embeddings using similarity search.

```python
from langchain_community.vectorstores import FAISS

# Create vector store from documents
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Similarity search
query = "What is the Kesavananda Bharati case?"
similar_docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(similar_docs, 1):
    print(f"Doc {i} (Page {doc.metadata.get('page', 'N/A')}):")
    print(f"  {doc.page_content[:150]}...")
```

**Popular Vector Stores:**

| Store | Type | Best For |
|-------|------|----------|
| FAISS | In-memory | Development, small datasets |
| Chroma | Persistent | Medium datasets, easy setup |
| Pinecone | Cloud | Production, large scale |

---

### 3.6 Retrievers - Finding Relevant Documents

A **retriever** is an interface that returns relevant documents given a query:

```python
# Convert vector store to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 4}     # Return top 4 documents
)

# Use the retriever
docs = retriever.invoke("What are fundamental rights?")
print(f"Retrieved {len(docs)} documents")
```

**Search Types:**
- **Similarity**: Returns top-k most similar documents
- **MMR (Maximum Marginal Relevance)**: Balances relevance with diversity

---

### 3.7 Building Complete RAG Chains

Now let's combine everything into a complete RAG pipeline:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful legal assistant.
Answer based ONLY on the following context:

Context:
{context}

If the answer is not in the context, say "I cannot find this information."
"""),
    ("human", "{question}")
])

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain with LCEL
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve → Format
        "question": RunnablePassthrough()    # Pass question through
    }
    | rag_prompt     # Create prompt with context
    | llm            # Generate response
    | StrOutputParser()  # Parse to string
)

# Use the RAG chain
answer = rag_chain.invoke("What is the basic structure doctrine?")
print(answer)
```

---

### 3.8 Source Attribution & Streaming RAG

**Adding Source Attribution:**

```python
from langchain_core.runnables import RunnableParallel

# RAG chain that returns sources
rag_with_sources = RunnableParallel(
    {
        "answer": rag_chain,
        "sources": retriever
    }
)

result = rag_with_sources.invoke("Explain judicial review")
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for i, doc in enumerate(result['sources'], 1):
    print(f"  {i}. Page {doc.metadata.get('page', 'N/A')}")
```

**Streaming RAG Responses:**

```python
# Stream the answer for better UX
for chunk in rag_chain.stream("What is Article 21?"):
    print(chunk, end="", flush=True)
```

**Saving & Loading Vector Store:**

```python
# Save for later use
vectorstore.save_local("faiss_index")

# Load in future sessions
from langchain_community.vectorstores import FAISS
loaded_store = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

### 3.9 📝 Activity 2: True or False Quiz

State whether the following statements are True or False:

1. RAG stands for "Retrieval-Augmented Generation" and combines document retrieval with LLM generation.

2. The `chunk_overlap` parameter should always be set to 0 for best results.

3. FAISS is a vector store that performs similarity search using embeddings.

4. Embeddings convert text into numerical vectors where similar texts have similar vector representations.

5. In a RAG pipeline, the retriever is called after the LLM generates a response.

6. The `RecursiveCharacterTextSplitter` splits documents at paragraph and sentence boundaries.

7. HuggingFace embeddings require an API key to use.

8. MMR (Maximum Marginal Relevance) search balances relevance with diversity in retrieved documents.

<details>
<summary><strong>📋 Click to reveal answers</strong></summary>

1. **True** - RAG = Retrieval-Augmented Generation
2. **False** - Overlap helps maintain context between chunks (typically 10-20%)
3. **True** - FAISS is Facebook's vector similarity search library
4. **True** - This is the core concept of embeddings
5. **False** - Retriever is called BEFORE the LLM to provide context
6. **True** - It tries paragraph first, then sentence, then character splits
7. **False** - HuggingFace embeddings run locally, no API key needed
8. **True** - MMR reduces redundancy by prioritizing diverse results

</details>

---

## 4. 📊 Explore Further

Uncover and unlock new insights as you dive into the captivating content found in the provided link.

📖 **LangChain RAG Tutorial**: https://python.langchain.com/docs/tutorials/rag/

Tackle these questions head-on after reading the tutorial:

1. True or False: The `create_retrieval_chain` helper can automatically handle conversation history.

2. True or False: LangChain supports only FAISS as a vector store.

3. True or False: You can use `create_stuff_documents_chain` to format retrieved documents into the prompt.

4. True or False: RAG always requires a local vector store and cannot use cloud-based options.

<details>
<summary><strong>📋 Click to reveal answers</strong></summary>

1. **True** - LangChain provides chains that handle chat history
2. **False** - LangChain supports many vector stores (Chroma, Pinecone, Weaviate, etc.)
3. **True** - This is a common pattern in LangChain RAG implementations
4. **False** - Cloud vector stores like Pinecone and Weaviate are fully supported

</details>

---

## 📝 Summary

### What Did We Learn?

**LangChain Fundamentals:**
- ✅ Initialized ChatGroq with the free Groq API for fast LLM inference
- ✅ Used SystemMessage, HumanMessage, and AIMessage to structure conversations
- ✅ Created reusable ChatPromptTemplate with dynamic variables
- ✅ Built chains using LCEL's pipe operator (`|`) for composability
- ✅ Generated structured outputs using Pydantic models
- ✅ Implemented streaming for real-time responses and batching for efficiency
- ✅ Created tools that LLMs can call for external function access

**RAG Fundamentals:**
- ✅ Loaded documents using PyPDFLoader
- ✅ Split documents into chunks with RecursiveCharacterTextSplitter
- ✅ Generated embeddings using HuggingFace's sentence-transformers
- ✅ Stored and searched vectors with FAISS
- ✅ Built complete RAG chains for document Q&A
- ✅ Added source attribution to provide transparency
- ✅ Persisted vector stores for reuse

### Shortcomings & Challenges

| Challenge | Mitigation |
|-----------|------------|
| Chunk size affects retrieval quality | Experiment with different sizes (500-2000) |
| Embedding model choice impacts accuracy | Test multiple models for your domain |
| Large documents slow down indexing | Use batch processing and caching |
| Context window limits | Use summarization or hierarchical retrieval |
| Hallucinations in responses | Strengthen system prompts, use source attribution |

---

## 🚀 Enhance Your Knowledge

Explore these official documentation resources:

- 📘 **LangChain Documentation**: https://python.langchain.com/docs/
- 🔗 **LangChain Expression Language Guide**: https://python.langchain.com/docs/concepts/lcel/
- 📚 **LangChain RAG Tutorial**: https://python.langchain.com/docs/tutorials/rag/
- ⚡ **Groq API Documentation**: https://console.groq.com/docs/quickstart
- 🔍 **FAISS Documentation**: https://faiss.ai/
- 🤗 **HuggingFace Embeddings**: https://huggingface.co/sentence-transformers
- 📄 **Pydantic Documentation**: https://docs.pydantic.dev/

---

## 🛠️ Try it Yourself

### Task 1: LangChain Fundamentals

Answer the following questions by implementing in code:

1. Create a ChatPromptTemplate that takes `language` and `topic` as variables and asks the LLM to explain a programming concept.

2. Build an LCEL chain that uses a Pydantic model to return a structured response with `concept_name`, `definition`, and `example_code` fields.

3. Create a custom tool using the `@tool` decorator that converts temperatures between Celsius and Fahrenheit.

4. Implement a batching example that processes 5 different coding questions simultaneously.

5. Modify the streaming example to count the total number of tokens in the response.

### Task 2: RAG Implementation

Build your own RAG system with the following requirements:

1. Load a PDF document of your choice using PyPDFLoader.

2. Experiment with different `chunk_size` and `chunk_overlap` values. Document how they affect retrieval quality.

3. Create a RAG chain with source attribution that shows which pages were used.

4. Save your vector store to disk and load it in a new session.

5. Compare results between `similarity` and `mmr` search types.

---

### 🔵 Share Your Work!

Share your insights about implementing RAG systems and the challenges you faced on the **AlmaBetter Community Platform**.

Questions to discuss:
- What chunk_size worked best for your documents?
- How did you handle documents with tables or images?
- What strategies did you use to improve retrieval accuracy?

---

## 📦 Requirements

```
langchain>=0.3.0
langchain-core>=0.3.0
langchain-groq>=0.2.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
langchain-text-splitters>=0.3.0
python-dotenv>=1.0.0
pypdf>=4.0.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
```

---

## 🔑 Environment Setup

Create a `.env` file in your project root:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get your free API key at: https://console.groq.com/keys

---

<div align="center">

**Happy Learning! 🚀**

*Built with ❤️ for AlmaBetter Students*

</div>
