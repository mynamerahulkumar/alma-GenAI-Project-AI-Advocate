# 🧠 Day 3: RAG Pipeline & Web Search - Creating Intelligent Responses

**Duration:** 2 hours | **Type:** Advanced Integration | **Difficulty:** Intermediate-Advanced

---

## 1️⃣ SESSION FLOW

### What We'll Cover Today (Step-by-Step)

1. **RAG Architecture Review** (10 min)
   - How retrieval + generation works together
   - The flow: Query → Retrieve → Generate
   - Why this is more powerful than LLMs alone

2. **LLM Integration with Groq** (15 min)
   - Introduction to Groq API
   - Why Groq (speed, free tier, quality)
   - Setting up ChatGroq

3. **Building Prompt Templates** (15 min)
   - Context injection in prompts
   - Source attribution strategies
   - Testing prompts

4. **LangChain Chains & LCEL** (20 min)
   - What are chains?
   - LCEL (LangChain Expression Language)
   - Building your first RAG chain

5. **Implementing Web Search** (15 min)
   - Tavily API overview
   - Hybrid search (documents + web)
   - When to use each source

6. **Streaming Responses** (15 min)
   - Real-time response generation
   - Streaming in action
   - User experience benefits

7. **Integration & Testing** (10 min)
   - Full end-to-end RAG pipeline
   - Testing and debugging
   - Q&A

---

## 2️⃣ LEARNING OBJECTIVES

By the end of Day 3, students will be able to:

✅ **Understand RAG Orchestration**
- Explain the complete RAG workflow
- Understand when to retrieve vs. generate
- Design prompts for context injection
- Track and cite sources

✅ **Master LLM Integration**
- Use Groq API for fast, free LLM access
- Control temperature and max tokens
- Handle streaming responses
- Implement error handling

✅ **Build LangChain Chains**
- Write LCEL expressions
- Chain retrievers to LLMs
- Add custom processing steps
- Debug chain execution

✅ **Enhance with Web Search**
- Integrate Tavily web search
- Combine document + web results
- Implement hybrid search logic
- Format results for LLMs

✅ **Apply SOLID Principles**
- Each component has single responsibility
- Easy to replace components (plug-and-play)
- Chains are composable and testable

**Prerequisites:** Day 1-2 knowledge, understanding of prompts and LLMs

**Key Concepts:** RAG pipeline, chains, LCEL, streaming, web search, source tracking, prompt engineering

---

## 3️⃣ THEME: THE KNOWLEDGE SYNTHESIZER

### Real-World Context

**Scenario:** You're building customer support for a tech company.

**Before RAG (Traditional LLM):**
- Customer: "How do I set up OAuth in your API?"
- LLM: "OAuth is an authentication protocol. You would typically..."
- Problem: Generic answer, not your API's specifics ❌

**With RAG (Today's Solution):**
- Customer: "How do I set up OAuth in your API?"
- System: Retrieves your OAuth documentation
- System: Passes it to LLM with context
- LLM: "In your API, OAuth is configured by..."
- Result: Specific, accurate, sourced answer ✅

**Why This Day?**
- Days 1-2 prepared documents (loaded → embedded)
- Day 3 uses those documents to generate answers (RAG in action)
- Day 4 builds the user interface
- Today is where everything comes together

---

## 4️⃣ PRIMARY GOAL

### What You'll Build

A **Complete RAG System** that:

1. ✅ Retrieves relevant documents by semantic search
2. ✅ Passes context to an LLM
3. ✅ Generates accurate, sourced answers
4. ✅ Streams responses in real-time
5. ✅ Optionally searches the web for current information
6. ✅ Tracks and cites sources

### Architecture: Day 3 in the RAG Pipeline

```
┌──────────────────────────────────────────────────────────┐
│              COMPLETE RAG PIPELINE                        │
│                                                            │
│  USER QUERY                                               │
│    "Tell me about machine learning"                       │
│    ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ RETRIEVAL (Using Day 2 Vector Store)                │ │
│  │ ├─ Convert query to embedding                       │ │
│  │ ├─ Search FAISS for similar chunks                  │ │
│  │ └─ Retrieve top-3 relevant documents                │ │
│  └─────────────────────────────────────────────────────┘ │
│    ↓                                                       │
│  CONTEXT DOCUMENTS                                         │
│    "Machine learning is supervised and unsupervised...   │
│     Supervised learning uses labeled data..."             │
│    ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ AUGMENTED PROMPT                                     │ │
│  │ System: You are helpful. Use context to answer.     │ │
│  │ Context: [Retrieved documents]                      │ │
│  │ Question: Tell me about machine learning             │ │
│  └─────────────────────────────────────────────────────┘ │
│    ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ LLM (Groq - Fast, Free)                             │ │
│  │ ├─ Process prompt + context                         │ │
│  │ ├─ Generate answer                                  │ │
│  │ └─ Stream response tokens                           │ │
│  └─────────────────────────────────────────────────────┘ │
│    ↓                                                       │
│  GENERATED ANSWER                                          │
│    "Machine learning includes supervised and...           │
│     Source: ml_guide.pdf (page 2), ml_intro.txt"          │
│                                                            │
│  OPTIONAL: WEB SEARCH                                      │
│    If answer incomplete → Search the web                  │
│    Combine document + web results                         │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

## 5️⃣ MAIN CONTENT (PART 1): RAG Orchestration & LLM Integration

### Understanding the RAG Flow

**Step-by-step:**

```
1. QUERY ARRIVES
   ↓
2. RETRIEVE PHASE
   ├─ Find relevant documents (from vector store)
   ├─ Rank by similarity
   └─ Select top-K
   ↓
3. AUGMENTATION PHASE
   ├─ Format retrieved documents
   ├─ Inject into prompt
   └─ Create "augmented prompt"
   ↓
4. GENERATION PHASE
   ├─ Send augmented prompt to LLM
   ├─ LLM reads context
   ├─ LLM generates answer
   └─ Stream response
   ↓
5. POST-PROCESSING
   ├─ Attach source information
   ├─ Format for user
   └─ Return to user
```

### Introducing Groq: Fast, Free LLM API

**Why Groq?**

| Feature | Groq | OpenAI | Claude |
|---------|------|--------|--------|
| **Cost** | FREE (14k+ req/min) | $0.015/1K tokens | $0.80/1M tokens |
| **Speed** | ~500 tokens/sec | ~200 tokens/sec | ~150 tokens/sec |
| **Model** | Llama 3.1 8B-70B | GPT-4 | Claude 3 |
| **Setup** | 1 API key | 1 API key + billing | 1 API key + billing |
| **Perfect for** | Teaching, prototypes | Production | Enterprise |

**Getting Started:**

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,           # Creativity (0=deterministic, 1=creative)
    max_tokens=1000,           # Max response length
    groq_api_key="gsk_..."     # From console.groq.com
)

# Use it like any LangChain LLM
response = llm.invoke("Tell me about Python")
print(response.content)
```

**Model Options:**

```python
# Fast and lightweight (perfect for our use case)
"llama-3.1-8b-instant"

# More powerful but slower
"llama-3.1-70b-versatile"

# Better at reasoning
"mixtral-8x7b-32768"
```

### Prompt Templates for RAG

**Basic Structure:**

```python
from langchain.prompts import PromptTemplate

template = """You are a helpful assistant. Use the provided context to answer the question.
If you don't know the answer, say "I don't know" rather than making something up.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# Use it
formatted_prompt = prompt.format(
    context="Machine learning is...",
    question="What is ML?"
)
```

**Advanced: RAG Prompt with Source Tracking**

```python
from langchain.prompts import PromptTemplate

rag_template = """You are a helpful assistant that answers questions based on documents.

{context}

Question: {question}

Provide your answer and cite the source document(s) that support it."""

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_template
)
```

**How Context Gets Injected:**

```
Template: "Answer: {context}"
         
Documents retrieved:
  - "ML uses algorithms to learn patterns"
  - "Supervised learning needs labels"

Context becomes:
  Document 1: "ML uses algorithms to learn patterns"
  Document 2: "Supervised learning needs labels"

Final prompt:
  "Answer: 
   Document 1: ML uses algorithms to learn patterns
   Document 2: Supervised learning needs labels"
```

### Building Basic Chains

**LangChain Chains** connect components together. The simplest version:

```python
from langchain.chains import LLMChain

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Use it
response = chain.run(question="What is ML?", context="...")
print(response)
```

---

## 6️⃣ QUIZ: Test Your Understanding (Part 1)

### Question Set 1: Fill in the Blanks

**Question 1:** RAG orchestration follows three phases: ________, ________, and ________.

<details>
<summary>Answer</summary>
Retrieval, Augmentation, Generation (or: Retrieve, Augment, Generate)
</details>

---

**Question 2:** Groq is particularly useful for teaching because it offers ________ tier API access with ________ response times.

<details>
<summary>Answer</summary>
Free (or generous), fast (or instant)
</details>

---

**Question 3:** A ________ injects retrieved documents into a prompt template to provide context to the LLM.

<details>
<summary>Answer</summary>
PromptTemplate (or: prompt template)
</details>

---

**Question 4:** ________ determines how "creative" an LLM's responses are (0=deterministic, 1=highly creative).

<details>
<summary>Answer</summary>
Temperature
</details>

---

**Question 5:** The main advantage of RAG over a fine-tuned LLM is that you can update information without ________.

<details>
<summary>Answer</summary>
Retraining (or: fine-tuning again)
</details>

---

## 7️⃣ MAIN CONTENT (PART 2): LCEL Chains & Web Search

### Introduction to LCEL (LangChain Expression Language)

LCEL lets you compose chains with elegant, readable syntax:

```python
from langchain_core.runnables import RunnablePassthrough

# Traditional way (complex)
chain = LLMChain(llm=llm, prompt=prompt)

# LCEL way (readable)
chain = prompt | llm

# LCEL with retriever (RAG)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

**Breaking it down:**

```
┌────────────────┐
│ Input: Query   │
└────────┬───────┘
         │
    {"context": retriever, "question": RunnablePassthrough()}
         │
    ├─ Get context: retriever(query)
    └─ Pass query through: query
         │
    ┌────┴─────────────────┐
    │ prompt(context, q)   │
    └────┬────────────────┘
         │
    ┌────┴──────────────┐
    │ llm(prompt)       │
    └────┬──────────────┘
         │
    ┌────┴──────────────┐
    │ Output: Answer    │
    └───────────────────┘
```

### Complete RAG Chain Example

```python
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Set up retriever (from Day 2)
vector_store = FAISS.load_local("./faiss_index", embedder)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. Create LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# 3. Create prompt
template = """Answer based on context:
{context}
Question: {question}"""
prompt = PromptTemplate.from_template(template)

# 4. Build chain with LCEL
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Use it
answer = chain.invoke("What is machine learning?")
print(answer)
```

### Streaming Responses

**Without Streaming (traditional):**
```
chain.invoke(query)
← [waits 10 seconds]
← Returns complete answer
```

**With Streaming (modern UX):**
```
chain.stream(query)
← Returns "Machine"
← Returns "learning"
← Returns "is"
← ... [user sees response building in real-time]
```

**Implementation:**

```python
# Streaming with LCEL
for chunk in chain.stream({"question": "What is ML?"}):
    print(chunk, end="", flush=True)  # Print as it arrives
```

### Implementing Web Search with Tavily

**Why Web Search?**

- Documents might be outdated
- Users ask about current events
- Hybrid approach: Documents + Web = Best of both

**Setting up Tavily:**

```python
from tools.tavily_search import TavilySearchTool

search = TavilySearchTool(api_key="tvly_...")

# Search the web
results = search.search(
    query="latest machine learning trends 2024",
    max_results=3
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Link: {result['link']}")
    print(f"Content: {result['content']}\n")
```

**Combining Document + Web Search:**

```python
# Get document results
doc_results = retriever.invoke(query)

# Get web results
web_results = search.search(query, max_results=2)

# Combine and format for LLM
combined_context = f"""
Document sources:
{format_docs(doc_results)}

Recent web sources:
{format_web_results(web_results)}
"""

# Pass to RAG
answer = rag_chain.invoke({
    "context": combined_context,
    "question": query
})
```

### Source Tracking and Attribution

**Why Important?**
- Users trust sources
- Legal requirements in some domains
- Reproducible answers

**Implementation:**

```python
# Retrieve with sources
results_with_scores = vector_store.similarity_search_with_score(
    query="machine learning",
    k=3
)

# Track sources
sources = []
context_text = ""

for doc, score in results_with_scores:
    source = doc.metadata.get("source", "Unknown")
    sources.append(source)
    context_text += f"[{source}]: {doc.page_content}\n"

# Generate answer
answer = llm.invoke(prompt.format(context=context_text, question=query))

# Attach sources
output = {
    "answer": answer.content,
    "sources": list(set(sources)),  # Unique sources
    "confidence": score  # Optional relevance score
}
```

---

## 8️⃣ ACTIVITY: True/False Assessment

**Directions:** Answer True or False for each statement.

**Statement 1:** RAG requires the LLM to have access to internet during generation.
- [ ] True
- [x] **False** ← Correct! RAG provides documents upfront; no internet access needed.

---

**Statement 2:** Using Groq instead of OpenAI means lower quality answers.
- [ ] True
- [x] **False** ← Correct! Groq's Llama 3.1 is high quality and often faster.

---

**Statement 3:** Temperature should be 0 for all use cases.
- [ ] True
- [x] **False** ← Correct! Temperature depends on task (0 for facts, 0.8 for creativity).

---

**Statement 4:** Web search in RAG is always necessary.
- [ ] True
- [x] **False** ← Correct! Optional; use only for questions that need current info.

---

**Statement 5:** LCEL chains are purely syntactic sugar with no functional benefit.
- [ ] True
- [x] **False** ← Correct! LCEL enables streaming, parallelization, and better optimization.

---

## 9️⃣ EXPLORE FURTHER: Deep Dive Resources

### Advanced Questions for Your Research

1. **Alternative LLMs:** How would you integrate Claude, GPT-4, or Mixtral instead of Groq? What changes?

2. **Prompt Engineering:** How would you optimize prompts for different question types (factual, reasoning, creative)?

3. **Chain Debugging:** What tools can you use to debug LCEL chain execution? How would you trace errors?

4. **Hybrid Search:** Beyond document + web, what other sources could you integrate (APIs, databases)?

5. **Response Quality:** How would you measure RAG answer quality? What metrics matter?

6. **Caching:** How could you cache LLM responses to reduce API calls?

### Official Resources

- **LangChain Chains Documentation:** https://python.langchain.com/docs/modules/chains/
- **LCEL Documentation:** https://python.langchain.com/docs/expression_language/
- **Groq Documentation:** https://console.groq.com/docs
- **Tavily Search API:** https://tavily.com/
- **LangChain + Groq Integration:** https://python.langchain.com/docs/integrations/llms/groq

### Research Papers

- "In-Context Learning with Long-Context Models" (various authors)
- "Retrieval-Augmented Generation for Knowledge-Intensive Tasks" (Lewis et al., 2020)
- "LangChain: Building Production-Ready LLM Applications" (technical blog posts)

---

## 🔟 SUMMARY: What We Learned Today

### Key Takeaways

**RAG Pipeline:**
- Retrieve relevant documents
- Augment prompt with context
- Generate answers grounded in facts

**LLM Integration:**
- Groq provides fast, free API access
- Temperature controls creativity
- Streaming improves UX

**LCEL Chains:**
- Elegant, readable syntax
- Composable components
- Enables optimization and streaming

**Web Search:**
- Optional enhancement
- Combines documents + internet
- Improves answer currency

**Source Tracking:**
- Essential for credibility
- Enables verification
- Required in many domains

### Common Mistakes to Avoid

❌ **Mistake 1:** Retrieving too many/few documents  
✅ **Fix:** Start with k=3, tune based on results

❌ **Mistake 2:** Not controlling temperature  
✅ **Fix:** Use 0.3 for facts, 0.8 for creative tasks

❌ **Mistake 3:** Ignoring sources in prompt  
✅ **Fix:** Always include source attribution in template

❌ **Mistake 4:** Using web search for everything  
✅ **Fix:** Use hybrid only when freshness matters

---

## 1️⃣1️⃣ ENHANCE YOUR KNOWLEDGE: Additional Learning Resources

### Official Documentation & Blogs

1. **LangChain Expression Language (LCEL)**
   - Tutorial: https://python.langchain.com/docs/expression_language/
   - Advanced patterns: https://python.langchain.com/docs/expression_language/composition

2. **Groq API**
   - Console: https://console.groq.com/
   - Documentation: https://console.groq.com/docs/
   - Models: https://console.groq.com/docs/models

3. **LangChain Chains**
   - Overview: https://python.langchain.com/docs/modules/chains/
   - LCEL Cookbook: https://python.langchain.com/docs/expression_language/cookbook/

4. **Tavily Search Integration**
   - API Docs: https://tavily.com/
   - LangChain Integration: https://python.langchain.com/docs/integrations/tools/tavily

### Community Resources

- LangChain Discord: https://discord.gg/langchain
- GitHub Discussions: https://github.com/langchain-ai/langchain/discussions
- Reddit: r/LanguageModels, r/OpenAI

### Videos to Watch

- "Building RAG Systems with LangChain" - LangChain YouTube
- "LCEL Tutorial" - Official LangChain
- "Prompt Engineering for RAG" - DeepLearning.AI

---

## 1️⃣2️⃣ TRY IT YOURSELF: Tasks & Challenges

### Task 1: Build Your RAG Chain

**Objective:** Create a complete end-to-end RAG system.

**Steps:**
1. Load your Day 2 vector store
2. Initialize Groq LLM
3. Create a RAG prompt template
4. Build the LCEL chain
5. Test with 5 different queries
6. Save the results

**Expected Output:**
```
Query: "What is machine learning?"

Retrieved context:
├─ Document 1 (score 0.92): "ML is a subset of AI..."
├─ Document 2 (score 0.87): "Supervised learning uses..."
└─ Document 3 (score 0.81): "Neural networks..."

Generated Answer:
Machine learning is a subset of artificial intelligence
that enables systems to learn and improve... [sources: doc1.pdf, doc2.txt]
```

---

### Task 2: Temperature & Response Variation

**Objective:** Understand how temperature affects responses.

**Steps:**
1. Take one query
2. Generate answers with temperatures: 0.0, 0.3, 0.7, 1.0
3. Compare responses:
   - Are they consistent?
   - How creative do they get?
   - Which is best for your use case?
4. Document observations

**Expected Pattern:**
```
Temperature 0.0 (Deterministic):
├─ Response 1: "Machine learning is the field..."
├─ Response 2: "Machine learning is the field..."  ← Identical
└─ Response 3: "Machine learning is the field..."

Temperature 1.0 (Creative):
├─ Response 1: "ML lets computers groove to data..."
├─ Response 2: "Think of ML as teaching..."
└─ Response 3: "Algorithms absorb patterns..."  ← All different!
```

---

### Task 3: Hybrid Search (Documents + Web)

**Objective:** Implement hybrid search combining documents and web.

**Challenge:**
1. Create a query that would benefit from current information
   (e.g., "Latest AI trends 2024")
2. Implement hybrid search:
   - Get document results
   - Get web results
   - Combine into single context
3. Generate answer with combined context
4. Compare with document-only approach
5. Which is better? When?

**Expected Comparison:**
```
Document-Only:
- Sources: training_materials.pdf (2022)
- Accuracy: Medium
- Freshness: Low

Hybrid (Document + Web):
- Sources: training_materials.pdf + 5 recent articles
- Accuracy: High
- Freshness: High
```

---

### Task 4: Source Tracking System

**Objective:** Build a system that tracks and attributes sources.

**Challenge:**
```python
# Create a SourceTrackingRAG class that:
# 1. Retrieves documents with sources
# 2. Generates answer
# 3. Maps which source contributed to which answer part
# 4. Outputs answer + detailed attribution

class SourceTrackingRAG:
    def __init__(self, chain, retriever):
        # Initialize
        pass
    
    def query_with_sources(self, question):
        # Get documents with metadata
        pass
    
    def get_attribution(self):
        # Return: answer + source mapping
        pass

# Usage:
rag = SourceTrackingRAG(chain, retriever)
result = rag.query_with_sources("What is ML?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result['confidence']}")
```

---

### Task 5: Chain Optimization & Debugging

**Objective:** Optimize RAG chain performance.

**Challenge:**
1. Build your RAG chain
2. Time the execution:
   - Retrieval time
   - LLM generation time
   - Total time
3. Experiment with optimizations:
   - Different k values (1, 3, 5, 10)
   - Different model sizes
   - Caching strategies
4. Profile memory usage:
   - Loaded vector store size
   - LLM memory footprint
   - Total memory
5. Write optimization recommendations

**Expected Output:**
```
Baseline:
├─ Retrieval: 0.2s
├─ LLM: 5.3s
└─ Total: 5.5s

Optimized (k=3 instead of 10):
├─ Retrieval: 0.1s
├─ LLM: 5.2s
└─ Total: 5.3s  ← 3.6% faster
```

---

### Community Discussion

**Post your answers to these on the discussion forum:**

1. What temperature did you find optimal for your use case?
2. Did hybrid search improve your results? How?
3. What was the most challenging part of debugging your chain?
4. How would you handle very long context (100+ documents)?

---

## 🏁 End of Day 3

### You Now Know:

✅ The complete RAG pipeline (retrieve → augment → generate)  
✅ How to integrate Groq LLM  
✅ How to build LCEL chains  
✅ How to control response quality  
✅ How to track and cite sources  
✅ How to integrate web search  

### Tomorrow (Day 4):

🚀 We'll build a **Streamlit UI**  
🚀 We'll create **file upload** functionality  
🚀 We'll implement **chat history**  
🚀 We'll deploy the **complete application**  

**Action Items Before Day 4:**
- ✅ Complete all 5 tasks above
- ✅ Build and test your complete RAG chain
- ✅ Experiment with different prompts and temperatures
- ✅ Understand the full end-to-end flow

---

## 📚 Quick Reference

### Code Snippets You'll Use

```python
# Setup LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Build LCEL chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = chain.invoke("your question")

# Stream it
for chunk in chain.stream({"question": "your question"}):
    print(chunk, end="", flush=True)
```

---

**Happy Learning! 🎓**

*Next: Day 4 - Streamlit UI & Deployment*
