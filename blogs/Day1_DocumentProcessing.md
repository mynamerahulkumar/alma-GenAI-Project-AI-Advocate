# 🗓️ Day 1: Document Processing - Building the Foundation of RAG

**Duration:** 2 hours | **Type:** Foundational | **Difficulty:** Beginner

---

## 1️⃣ SESSION FLOW

### What We'll Cover Today (Step-by-Step)

1. **Introduction to RAG** (10 min)
   - What is Retrieval-Augmented Generation?
   - Why RAG matters in modern AI
   - Real-world applications

2. **Document Loading Fundamentals** (20 min)
   - Supported file formats (PDF, TXT, DOCX)
   - LangChain document loaders
   - The Document object structure

3. **Text Chunking Strategies** (20 min)
   - Why splitting matters
   - Chunking algorithms (fixed-size, recursive, semantic)
   - Overlap and context preservation

4. **Hands-On: Load & Chunk** (30 min)
   - Live coding with actual documents
   - Experimenting with chunk sizes
   - Understanding metadata

5. **Architecture Overview** (15 min)
   - How Day 1 fits into the RAG pipeline
   - Preview of vector stores (Day 2)
   - Q&A

6. **Exercise: Your First Document** (25 min)
   - Students load their own documents
   - Experiment with different chunk sizes
   - Troubleshooting

---

## 2️⃣ LEARNING OBJECTIVES

By the end of Day 1, students will be able to:

✅ **Understand RAG Fundamentals**
- Explain what RAG is and why it's needed
- Describe the differences between RAG and fine-tuning
- Identify real-world use cases for document-based AI

✅ **Master Document Loading**
- Load documents from PDF and TXT files using LangChain
- Access document metadata (source, page number)
- Handle different file formats correctly

✅ **Grasp Text Chunking**
- Explain why splitting documents into chunks is necessary
- Choose appropriate chunk sizes for different use cases
- Understand the trade-offs between chunk size and overlap
- Preserve document context through overlap strategy

✅ **Apply SOLID Principles**
- Understand the Single Responsibility Principle in DocumentProcessor
- See how separation of concerns improves code maintainability

**Prerequisites:** Basic Python knowledge, familiarity with file I/O

**Key Concepts:** Document loaders, text splitting, chunking strategy, metadata, overlap

---

## 3️⃣ THEME: THE KNOWLEDGE CURATOR

### Real-World Context

**Scenario:** Imagine you're building an AI assistant for a law firm. They have 500+ legal documents (PDFs, contracts, precedents). Simply throwing all this text at an LLM won't work—the model has token limits and needs *relevant* information.

**The Challenge:**
- 📄 Documents are too large to process as a whole
- 🔍 We need to find relevant parts quickly
- 💾 We need to preserve context and metadata
- 🎯 We need to feed the right information to the LLM

**The Solution:** Document processing is the first step that enables retrieval. Today, you become a **Knowledge Curator**—preparing documents so an AI can understand and retrieve the right information.

**Why This Day?**
- Day 1 prepares raw documents
- Day 2 converts them to searchable vectors
- Day 3 uses vectors to answer questions
- Day 4 builds the interactive interface

---

## 4️⃣ PRIMARY GOAL

### What You'll Build

A **Document Processing Pipeline** that:

1. ✅ Loads documents from multiple sources
2. ✅ Splits them intelligently into chunks
3. ✅ Preserves metadata (source, page, position)
4. ✅ Handles different file formats
5. ✅ Outputs clean, structured document chunks

### Architecture: Day 1 in the RAG Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE OVERVIEW                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Day 1: DOCUMENT PROCESSING ← YOU ARE HERE               │
│  ├─ Load documents (PDF, TXT)                            │
│  ├─ Split into chunks                                   │
│  └─ Preserve metadata                                   │
│           ↓                                              │
│  Day 2: EMBEDDING & INDEXING                            │
│  ├─ Convert chunks to vectors                           │
│  ├─ Store in FAISS                                      │
│  └─ Enable similarity search                            │
│           ↓                                              │
│  Day 3: RETRIEVAL & GENERATION                          │
│  ├─ Retrieve relevant chunks                            │
│  ├─ Pass to LLM                                         │
│  └─ Generate answers                                    │
│           ↓                                              │
│  Day 4: USER INTERFACE                                  │
│  ├─ Chat interface                                      │
│  ├─ File upload                                         │
│  └─ Interactive Q&A                                     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 5️⃣ MAIN CONTENT (PART 1): Document Loading Fundamentals

### What is RAG?

**RAG (Retrieval-Augmented Generation)** combines two powerful capabilities:

1. **Retrieval:** Finding relevant information from your documents
2. **Augmentation:** Enhancing LLM responses with that information
3. **Generation:** Creating answers based on retrieved context

**Why RAG?**

Without RAG, LLMs have limitations:
- 🚫 Can't access documents published after training
- 🚫 Can't access your company's proprietary information
- 🚫 Can't cite sources or update answers
- 🚫 Expensive to fine-tune for every new document

With RAG:
- ✅ Instantly add new documents
- ✅ Access private/proprietary information
- ✅ Answer questions grounded in facts
- ✅ Cite sources and maintain accuracy
- ✅ No expensive retraining needed

### Document Loaders in LangChain

LangChain provides **document loaders** for different file types. A loader converts files into `Document` objects.

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load a PDF
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()  # Returns: [Document, Document, ...]

# Load a text file
text_loader = TextLoader("notes.txt")
text_docs = text_loader.load()  # Returns: [Document]

# Each Document has:
# - page_content: str (the actual text)
# - metadata: dict (source, page number, etc.)
```

**Supported Formats:**
- 📄 PDF (via PyPDFLoader)
- 📝 TXT (via TextLoader)
- 📊 CSV (via CSVLoader)
- 🌐 Web (via WebBaseLoader)
- 📧 Email, Slack, GitHub, and many more...

### Understanding the Document Object

Every document in LangChain is a `Document` with two components:

```python
Document(
    page_content="This is the actual text from the document...",
    metadata={
        "source": "path/to/document.pdf",
        "page": 0,
        "row": 5  # For CSV
    }
)
```

**Why Metadata Matters:**
- 🔗 Track document source (cite sources in answers)
- 📄 Know which page information came from
- 🗂️ Filter by document type or date
- 📊 Organize retrieved results

---

## 6️⃣ QUIZ: Test Your Understanding (Part 1)

### Question Set 1: Fill in the Blanks

**Question 1:** RAG stands for ________ - ________ - ________

<details>
<summary>Answer</summary>
Retrieval - Augmented - Generation
</details>

---

**Question 2:** A LangChain `Document` object has two main components: ________ and ________

<details>
<summary>Answer</summary>
page_content and metadata
</details>

---

**Question 3:** To load a PDF file in LangChain, you use the ________ class.

<details>
<summary>Answer</summary>
PyPDFLoader
</details>

---

**Question 4:** The ________ component of metadata tells us which page a piece of text came from.

<details>
<summary>Answer</summary>
page (or page number)
</details>

---

**Question 5:** RAG is advantageous because it allows LLMs to access ________ that wasn't in their training data.

<details>
<summary>Answer</summary>
documents / new information / proprietary information / real-time data
</details>

---

## 7️⃣ MAIN CONTENT (PART 2): Text Chunking Strategies

### Why Split Documents into Chunks?

**The Problem:** Documents can be very large. Consider:
- A 100-page PDF = ~50,000 tokens
- An LLM has a context window (e.g., Claude: 200K tokens, GPT-4: 8K-128K tokens)
- We need to retrieve *relevant* pieces, not the whole document

**The Solution:** Split documents into smaller, manageable chunks.

### Chunking Strategy: Recursive Character Splitter

LangChain's `RecursiveCharacterTextSplitter` is like a smart cutting tool:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Target size in characters
    chunk_overlap=200     # Overlap to preserve context
)

chunks = splitter.split_documents(documents)
# Result: Original 50,000 tokens → 50+ chunks of ~1000 chars each
```

**How It Works:**

1. **First Cut:** Try splitting on `"\n\n"` (paragraph break)
2. **If Still Too Big:** Split on `"\n"` (line break)
3. **If Still Too Big:** Split on `" "` (space)
4. **If Still Too Big:** Split on `""` (character by character)

**Example:**

```
Original: "Chapter 1: Introduction\n\nParagraph 1. Long text...\n\nParagraph 2..."

After split:
├─ Chunk 1: "Chapter 1: Introduction\n\nParagraph 1. Long text..." [1000 chars]
├─ Chunk 2: "Paragraph 1. Long text...[overlap]...\n\nParagraph 2..." [1000 chars]
├─ Chunk 3: "Paragraph 2...[overlap]..." [800 chars]
```

### Understanding Chunk Size vs. Overlap

**Chunk Size** (1000 characters):
- ⬆️ **Larger chunks:** More context, fewer chunks, less expensive retrieval
- ⬇️ **Smaller chunks:** More specific, more chunks, more expensive retrieval

**Overlap** (200 characters):
- Ensures context isn't lost at chunk boundaries
- Helps retrieve related information together
- Typical: 10-20% of chunk size

**Real-World Tuning:**

| Use Case | Chunk Size | Overlap | Why |
|----------|-----------|---------|-----|
| Technical docs | 1500-2000 | 200-300 | Need full code examples |
| News articles | 500-800 | 100-150 | Standalone paragraphs |
| Legal docs | 2000-3000 | 300-500 | Complex clauses |
| Chat transcripts | 300-500 | 50-100 | Turn-based structure |

### Metadata Preservation

When you chunk documents, metadata travels with them:

```python
chunks = splitter.split_documents(documents)

# Each chunk preserves original metadata
for chunk in chunks:
    print(chunk.metadata)
    # Output: {"source": "document.pdf", "page": 0}
    
    # And includes chunk position
    print(chunk.page_content[:50])  # First 50 chars of chunk
```

---

## 8️⃣ ACTIVITY: True/False Assessment

**Directions:** Answer True or False for each statement.

**Statement 1:** RAG requires retraining the LLM every time you add new documents.
- [ ] True
- [x] **False** ← Correct! RAG doesn't require retraining; just add documents to the vector store.

---

**Statement 2:** A larger chunk size means fewer chunks overall.
- [x] **True** ← Correct! More text per chunk = fewer total chunks.
- [ ] False

---

**Statement 3:** Chunk overlap is always wasteful and should be minimized.
- [ ] True
- [x] **False** ← Correct! Overlap preserves context and helps retrieval.

---

**Statement 4:** Metadata in LangChain documents is optional and doesn't affect retrieval.
- [ ] True
- [x] **False** ← Correct! Metadata is crucial for source citation and filtering.

---

**Statement 5:** The RecursiveCharacterTextSplitter tries to cut at paragraph boundaries first before splitting at smaller units.
- [x] **True** ← Correct! It preserves text structure intelligently.
- [ ] False

---

## 9️⃣ EXPLORE FURTHER: Deep Dive Resources

### Advanced Questions for Your Research

1. **Semantic Chunking:** How would you chunk documents based on *meaning* rather than character count? What's the trade-off with recursive splitting?

2. **Multilingual Documents:** How would the chunking strategy differ for languages with different structures (e.g., Chinese, Japanese)?

3. **Performance Optimization:** If you have 10,000 documents to process, what strategies would speed up document loading?

4. **Token Counting:** Why does LangChain count tokens (not characters) for chunk sizes? What's the difference?

5. **Context Window:** How does your chunking strategy change if your LLM has a 4K token limit vs. 128K?

### Official Resources

- **LangChain Documentation - Text Splitters:** https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/
- **RecursiveCharacterTextSplitter Details:** https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_character
- **Document Loaders Guide:** https://python.langchain.com/docs/modules/data_connection/document_loaders/

### Recommended Reading

- "Chunking Strategies for RAG" - LangChain Blog
- "Embeddings: What Every Developer Should Know" - OpenAI Blog
- Research Paper: "Evaluating Chunking Strategies for Semantic Search"

---

## 🔟 SUMMARY: What We Learned Today

### Key Takeaways

**What is RAG?**
- Retrieval-Augmented Generation combines document retrieval with LLM generation
- Allows LLMs to answer questions grounded in your specific documents
- No retraining needed—just add new documents

**Document Loading**
- LangChain provides loaders for PDF, TXT, CSV, and many other formats
- Documents are objects with `page_content` and `metadata`
- Metadata is crucial for source tracking and filtering

**Text Chunking**
- Splitting documents into chunks makes them processable
- RecursiveCharacterTextSplitter preserves text structure
- Chunk size and overlap are tuning parameters based on your use case
- Metadata travels with chunks for tracking

**Why It Matters**
- Day 1 prepares documents for the vector store (Day 2)
- Clean, well-chunked documents = better retrieval later
- Proper chunking = better LLM answers

### Common Mistakes to Avoid

❌ **Mistake 1:** Chunks too large → Hard to retrieve specific info  
✅ **Fix:** Tune chunk size to your domain (test with 1000, 1500, 2000)

❌ **Mistake 2:** No overlap → Missing context  
✅ **Fix:** Always use 10-20% overlap

❌ **Mistake 3:** Ignoring metadata → Can't cite sources  
✅ **Fix:** Preserve metadata, use it in prompts

❌ **Mistake 4:** Different chunking per document → Inconsistent retrieval  
✅ **Fix:** Use same splitter for all documents

---

## 1️⃣1️⃣ ENHANCE YOUR KNOWLEDGE: Additional Learning Resources

### Official Documentation & Blogs

1. **LangChain Official Docs**
   - Document Loaders: https://python.langchain.com/docs/modules/data_connection/document_loaders/
   - Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/
   - RAG Introduction: https://python.langchain.com/docs/use_cases/question_answering/

2. **LangChain Blog**
   - "Building RAG Systems with LangChain": https://blog.langchain.dev/
   - Case studies and best practices

3. **DeepLearning.AI Short Courses**
   - "LangChain for LLM Application Development" (Free on Coursera)
   - "Building RAG Systems" (Free on DeepLearning.AI)

4. **Community Resources**
   - LangChain Discord: https://discord.gg/langchain
   - Stack Overflow: Tag `langchain`
   - GitHub Discussions: https://github.com/langchain-ai/langchain/discussions

### Videos to Watch

- "RAG Systems Explained" - LangChain YouTube
- "Document Processing with LangChain" - Various creators on YouTube

### Papers & Research

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)

---

## 1️⃣2️⃣ TRY IT YOURSELF: Tasks & Challenges

### Task 1: Load and Chunk Your Own Document

**Objective:** Apply what you learned with a real document.

**Steps:**
1. Prepare a document (PDF or TXT, 5-20 pages)
2. Load it using LangChain's document loader
3. Print the number of loaded documents and total characters
4. Experiment with different chunk sizes (500, 1000, 2000 characters)
5. For each configuration, print:
   - Number of chunks created
   - Average chunk size
   - Sample of first chunk
6. Which chunk size felt most balanced? Why?

**Expected Output:**
```
Loading document: my_doc.pdf
Loaded: 1 document(s), 45,230 characters total

Chunk Size: 500
├─ Chunks: 95
├─ Avg size: 478 chars
└─ Sample: "Introduction to Machine Learning..."

Chunk Size: 1000
├─ Chunks: 48
├─ Avg size: 942 chars
└─ Sample: "Introduction to Machine Learning..."
```

---

### Task 2: Understanding Metadata & Retrieval

**Objective:** See how metadata helps in retrieval scenarios.

**Steps:**
1. Load 3 different documents (PDF, TXT, mixed)
2. Chunk them all with consistent settings
3. Print metadata for the first 5 chunks
4. Identify what metadata you'd want to track
5. Write a function that filters chunks by source

**Expected Output:**
```
Chunk 1 Metadata:
├─ Source: budget_2024.pdf
├─ Page: 5
└─ Position: chunk_12

Chunk 2 Metadata:
├─ Source: meeting_notes.txt
├─ Page: 0
└─ Position: chunk_1
```

---

### Task 3: Chunking Strategy Deep Dive

**Objective:** Compare different chunking approaches.

**Challenge:**
1. Take a complex technical document
2. Split it 3 ways:
   - **Approach A:** Large chunks (2000 chars, 300 overlap)
   - **Approach B:** Medium chunks (1000 chars, 200 overlap)
   - **Approach C:** Small chunks (500 chars, 100 overlap)
3. Compare:
   - Number of chunks
   - Memory usage
   - Information redundancy (due to overlap)
4. Write down pros/cons of each approach

---

### Task 4: Build Your Document Processor

**Objective:** Create reusable code for document processing.

**Challenge:**
```python
# Create a DocumentProcessor class that:
# 1. Loads documents from a folder
# 2. Automatically detects file type (PDF, TXT)
# 3. Chunks with configurable parameters
# 4. Returns structured results with metadata

class DocumentProcessor:
    def __init__(self, chunk_size=1000, overlap=200):
        # Initialize
        pass
    
    def process_folder(self, folder_path):
        # Load all documents from folder
        pass
    
    def get_chunks(self):
        # Return all chunks with metadata
        pass

# Usage:
processor = DocumentProcessor(chunk_size=1500, overlap=250)
processor.process_folder("./documents/")
chunks = processor.get_chunks()
```

---

### Community Discussion

**Post your answers to these on the discussion forum:**

1. What's the best chunk size for your domain? Why?
2. Did you discover any edge cases with specific document types?
3. How would you handle corrupted or malformed PDFs?
4. What metadata would you add beyond what LangChain provides?

---

## 🏁 End of Day 1

### You Now Know:

✅ What RAG is and why it's revolutionary  
✅ How to load documents with LangChain  
✅ How to split documents intelligently  
✅ Why metadata matters  
✅ How Day 1 connects to Days 2-4  

### Tomorrow (Day 2):

🚀 We'll convert these chunks into **vectors**  
🚀 We'll build a searchable **vector store**  
🚀 We'll perform **semantic similarity search**  

**Action Items Before Day 2:**
- ✅ Complete all 4 tasks above
- ✅ Prepare a document to use for Days 2-4
- ✅ Run `demo_day1.py` to see the code in action
- ✅ Review SOLID principles (they'll matter in Day 3)

---

## 📚 Quick Reference

### Code Snippets You'll Use

```python
# Load documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Access chunk data
for chunk in chunks:
    print(f"Source: {chunk.metadata['source']}")
    print(f"Content: {chunk.page_content[:100]}...")
```

---

**Happy Learning! 🎓**

*Next: Day 2 - Vector Embeddings & Semantic Search*
