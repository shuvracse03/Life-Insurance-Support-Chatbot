# 🛡️ Life Insurance Support Assistant

An AI-powered life insurance support chatbot built with **LangGraph**, **Groq (Llama 3.3)**, **FastAPI**, and **Streamlit**.

The assistant answers questions about policy types, eligibility, benefits, riders, and the claims process — grounding every answer in a **configurable knowledge base** via RAG (Retrieval-Augmented Generation), with full multi-turn memory per session.

## ✨ Key Features

- 🧠 **LangGraph ReAct Agent**: Intelligent reasoning and tool selection
- 📚 **Configurable Knowledge Base**: Easy-to-update markdown-based content system
- 🔍 **RAG Implementation**: FAISS vector search with HuggingFace embeddings
- 💾 **Multi-session Memory**: SQLite-based conversation persistence
- 🚀 **Production Ready**: FastAPI backend with Streamlit frontend
- 🔧 **Hot-swappable Content**: Update knowledge without restarting the application

---

## 🏗️ Project Structure

```
.
├── backend/
│   ├── agent/              # LangGraph agent (graph, state, tools)
│   ├── app/                # FastAPI routes, schemas, dependencies
│   ├── knowledge_base/     # Markdown KB files + FAISS vector store
│   ├── memory/             # SQLite conversation checkpointer
│   ├── config.py           # LLM configuration (Groq)
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit chat UI
│   ├── components/         # Sidebar, chat components
│   └── requirements.txt
├── .env.example
├── run.sh
└── README.md
```

---

## ⚙️ Setup Guide

Follow these steps to get the Life Insurance Assistant running on your machine.

### Prerequisites

- **Python 3.12** or higher
- **Git** for cloning the repository
- **A valid Groq API key** (get one from [Groq Console](https://console.groq.com/))

### Step-by-Step Installation

#### 1. Clone and Navigate to the Repository

```bash
git clone <your-repo-url>
cd life-insurance-assistant
```

#### 2. Create and Activate Python Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# For Windows users:
# .venv\Scripts\activate
```

> **Important:** Always ensure your virtual environment is activated before running any Python commands.

#### 3. Install Project Dependencies

```bash
# Install backend dependencies (LangGraph, FastAPI, etc.)
pip install -r backend/requirements.txt

# Install frontend dependencies (Streamlit)
pip install -r frontend/requirements.txt
```

#### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Open .env in your favorite editor and add your Groq API key
# Example: GROQ_API_KEY=gsk_your_api_key_here
```

**Required Environment Variables:**

- `GROQ_API_KEY`: Your Groq API key (mandatory)
- `GROQ_MODEL`: Model to use (default: `llama-3.3-70b-versatile`)
- `FASTAPI_HOST`: Backend host (default: `localhost`)
- `FASTAPI_PORT`: Backend port (default: `8000`)

#### 5. Build the Knowledge Base Vector Store

```bash
# Navigate to backend directory
cd backend

# Build the FAISS vector store from knowledge base files
python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"

# Return to project root
cd ..
```

> **What happens here:** This step processes the markdown files in `backend/knowledge_base/data/` and creates a FAISS vector index for semantic search. This only needs to be done once (or when knowledge base content changes).

### Verification

To verify your setup is correct:

1. Check that all dependencies are installed:

   ```bash
   pip list | grep -E "(langchain|langgraph|fastapi|streamlit|groq)"
   ```

2. Verify the vector store was created:

   ```bash
   ls backend/knowledge_base/faiss_index/
   # Should show: index.faiss and index.pkl
   ```

3. Test environment variables:
   ```bash
   python -c "import os; print('GROQ_API_KEY set:', bool(os.getenv('GROQ_API_KEY')))"
   ```

---

## 🚀 Running the App

### Option A — One command (recommended)

```bash
chmod +x run.sh
./run.sh
```

### Option B — Manually in two terminals

**Terminal 1 — Backend:**

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**

```bash
source .venv/bin/activate
cd frontend
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🔑 Environment Variables

| Variable       | Description       | Default                   |
| -------------- | ----------------- | ------------------------- |
| `GROQ_API_KEY` | Your Groq API key | _(required)_              |
| `GROQ_MODEL`   | Groq model to use | `llama-3.3-70b-versatile` |
| `FASTAPI_HOST` | FastAPI host      | `localhost`               |
| `FASTAPI_PORT` | FastAPI port      | `8000`                    |

---

## 🧠 Architecture & LangGraph Flow

### System Overview

```
User (Streamlit UI)
       │
       │  POST /api/v1/chat
       ▼
FastAPI Backend
       │
       ▼
LangGraph Agent
  ├── llm_node (Groq Llama 3.3)
  │     └── bind_tools → search_knowledge_base, get_claims_process
  └── tools_node
        └── FAISS retriever (HuggingFace embeddings)
              └── Knowledge Base (Markdown files)
       │
       ▼
SQLite Checkpointer (multi-turn memory per session)
```

### LangGraph Agent Flow: Detailed Breakdown

The core of this application is a **LangGraph StateGraph** that implements a **ReAct (Reasoning + Acting) pattern** for intelligent question answering. Here's how it works:

#### 1. **State Management** (`AgentState`)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    session_id: str                          # Unique session identifier
```

**Design Reasoning:**

- `messages`: Maintains full conversation context using LangChain's `add_messages` annotation for automatic message aggregation
- `session_id`: Enables multi-user concurrent sessions with isolated conversation history
- Simple state design keeps the agent lightweight and maintainable

#### 2. **Node Architecture**

##### **LLM Node (`llm_node`)**

```python
def llm_node(state: AgentState) -> dict:
    llm = get_llm().bind_tools(TOOLS, tool_choice="auto")
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

**What it does:**

- Combines system prompt with conversation history
- Uses Groq's Llama 3.3 with tools bound for function calling
- `tool_choice="auto"` lets the LLM decide when to call tools vs. respond directly

**Why this approach:**

- **Automatic tool selection**: No hardcoded decision logic needed
- **Context preservation**: Full conversation history informs each response
- **Flexible reasoning**: LLM can choose to reason first, call tools, or answer directly

##### **Tool Executor Node (`tool_executor_node`)**

```python
def tool_executor_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool = TOOLS_BY_NAME[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}
```

**What it does:**

- Executes all tools requested by the LLM in parallel
- Returns structured `ToolMessage` objects with results
- Links each result to its corresponding tool call via `tool_call_id`

**Why this approach:**

- **Batch execution**: Multiple tool calls handled efficiently
- **Proper message formatting**: Tool results are properly structured for LLM consumption
- **Error isolation**: Each tool call is independent

#### 3. **Routing Logic (`should_continue`)**

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

**Flow Control:**

- If LLM requests tool calls → Route to `tool_executor_node`
- If LLM provides final answer → Route to `END`
- After tool execution → Always return to `llm_node` for synthesis

**Why this pattern:**

- **Simple decision logic**: Clear, readable routing based on message type
- **ReAct compliance**: Follows standard Reasoning → Acting → Reasoning cycle
- **Natural conversation flow**: Allows for multiple tool calls and follow-up questions

#### 4. **Graph Construction**

```python
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_executor_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
```

**Graph Topology:**

```
START → llm_node → [tools_node → llm_node]* → END
```

**Design Benefits:**

- **Deterministic flow**: Predictable execution path
- **Looping capability**: Can handle multi-step reasoning
- **Clean termination**: Natural end points when task is complete

### Tool Design Philosophy

#### **Tool 1: `search_knowledge_base`**

- **Purpose**: Semantic search over curated life insurance documentation
- **Technology**: FAISS vector store with HuggingFace embeddings
- **Reasoning**: RAG approach ensures accurate, grounded responses from authoritative sources

#### **Tool 2: `get_claims_process`**

- **Purpose**: Provides step-by-step claims filing guidance
- **Technology**: Static, well-structured procedure
- **Reasoning**: Claims process is standardized and doesn't require dynamic search

### Memory & Persistence

**SQLite Checkpointer Integration:**

- **Session isolation**: Each user gets a unique conversation thread
- **State persistence**: Conversation history survives server restarts
- **Stateful conversations**: Context maintained across multiple interactions

**Benefits:**

- **Scalability**: Handles multiple concurrent users
- **Reliability**: No conversation loss due to server issues
- **Context awareness**: Assistant remembers previous questions and answers

### Why This Architecture?

1. **Modularity**: Clean separation between reasoning, tool execution, and state management
2. **Extensibility**: Easy to add new tools or modify behavior
3. **Reliability**: Robust error handling and state persistence
4. **Performance**: Efficient tool execution and minimal state overhead
5. **Maintainability**: Clear code structure with single-responsibility components

This design provides a solid foundation for a production-ready conversational AI system while remaining simple enough for rapid iteration and debugging.

### 🎯 Configurable Knowledge Base & RAG Implementation

#### **✅ Fully Configurable Knowledge Base Structure**

The Life Insurance Assistant features a **completely configurable knowledge base** that can be easily customized and extended:

```
backend/knowledge_base/data/
├── benefits.md        # Policy benefits and coverage details
├── claims.md         # Claims process and requirements
├── eligibility.md    # Age, health, and occupation criteria
└── policy_types.md   # Term, Whole Life, ULIP, Endowment, Money-Back
```

#### **🔧 Knowledge Base Configuration Features**

##### **1. Easy Content Updates**

```bash
# Simply edit any markdown file in the knowledge base
nano backend/knowledge_base/data/policy_types.md

# Rebuild the vector store to reflect changes
cd backend
python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"
```

##### **2. Adding New Knowledge Categories**

```bash
# Add new markdown files for additional topics
echo "# Investment Plans" > backend/knowledge_base/data/investments.md
echo "# Senior Citizen Policies" > backend/knowledge_base/data/senior_policies.md

# The system automatically includes all .md files in the data directory
python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"
```

##### **3. Configurable Retrieval Parameters**

The retrieval system can be tuned via the `get_retriever()` function:

- **Chunk size**: Adjustable document segmentation
- **Search results**: Number of relevant chunks returned (default: 3)
- **Similarity threshold**: Minimum relevance score for results
- **Embedding model**: Switchable HuggingFace models

##### **4. Multi-format Support**

The knowledge base loader supports:

- **Markdown files** (.md) - Primary format
- **Text files** (.txt) - Plain text documents
- **Structured data** - Can be extended for JSON, CSV, etc.

#### **🔄 Dynamic Knowledge Base Updates**

**Hot-swappable Content:**

1. **Update content**: Modify any file in `backend/knowledge_base/data/`
2. **Rebuild index**: Run `build_vector_store()`
3. **Zero downtime**: No need to restart the application
4. **Instant availability**: New content is immediately searchable

**Configuration Examples:**

```python
# Custom knowledge base directory
KNOWLEDGE_BASE_PATH = "custom/knowledge/path"

# Custom embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Custom chunk size for document splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

#### **📊 Vector Store Pipeline**

1. **Document Loading**: Recursive scanning of markdown files with configurable directory paths
2. **Text Processing**: Configurable chunking strategies and preprocessing rules
3. **Embedding Generation**: Switchable HuggingFace models (`sentence-transformers/all-MiniLM-L6-v2` default)
4. **Vector Storage**: FAISS index with configurable similarity metrics
5. **Retrieval**: Semantic search with adjustable result count and relevance thresholds

#### **🎁 Configurable RAG Benefits**

- **✅ Fully Customizable**: Add/remove/modify knowledge without code changes
- **✅ Multi-domain Ready**: Easy to adapt for other insurance types or industries
- **✅ Performance Tunable**: Adjust retrieval parameters for speed vs. accuracy
- **✅ Content Versioning**: Track knowledge base changes and rollback if needed
- **✅ A/B Testing Ready**: Test different content versions seamlessly

---

## 💬 Sample Questions & Use Cases

### Policy Information Queries

- "What types of life insurance policies are available?"
- "What is the difference between term and whole life insurance?"
- "Explain ULIP policies and their benefits"
- "What riders can I add to my policy?"

### Eligibility & Underwriting

- "Am I eligible for a policy at age 55 with diabetes?"
- "What health conditions affect life insurance eligibility?"
- "Do I need medical tests for a ₹50 lakh policy?"

### Claims & Process

- "How do I file a claim after a policyholder passes away?"
- "What documents are needed for a claim?"
- "How long does claim settlement take?"
- "What happens if death occurs within 2 years of policy start?"

### Benefits & Coverage

- "What is covered under accidental death benefit?"
- "How does the maturity benefit work?"
- "What are the tax benefits of life insurance?"

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **"ModuleNotFoundError" when running Python commands**

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies if needed
pip install -r backend/requirements.txt
```

#### 2. **"GROQ_API_KEY not found" error**

```bash
# Check if .env file exists and contains your API key
cat .env | grep GROQ_API_KEY

# If missing, add it:
echo "GROQ_API_KEY=your_api_key_here" >> .env
```

#### 3. **FAISS index not found**

```bash
# Rebuild the vector store
cd backend
python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"
cd ..
```

#### 4. **Backend connection failed**

- Ensure FastAPI is running on port 8000
- Check for port conflicts: `lsof -i :8000`
- Verify firewall settings allow local connections

#### 5. **Streamlit frontend issues**

```bash
# Clear Streamlit cache
streamlit cache clear

# Restart with specific port
streamlit run frontend/app.py --server.port 8501
```

### Performance Tips

1. **Vector Store Optimization**: The FAISS index loads on first query - subsequent searches are much faster
2. **Memory Usage**: Each session maintains conversation history; consider periodic cleanup for long-running deployments
3. **API Rate Limits**: Groq has rate limits - implement request throttling for high-traffic scenarios

### Development Mode

For development with auto-reload:

```bash
# Backend with hot reload
cd backend && uvicorn app.main:app --reload

# Frontend with file watching
cd frontend && streamlit run app.py --server.fileWatcherType poll
```

### Knowledge Base Configuration

#### **Adding New Content**

```bash
# 1. Create new markdown file
echo "# New Insurance Topic" > backend/knowledge_base/data/new_topic.md

# 2. Add your content to the file
nano backend/knowledge_base/data/new_topic.md

# 3. Rebuild vector store
cd backend && python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"
```

#### **Updating Existing Content**

```bash
# 1. Edit any markdown file
nano backend/knowledge_base/data/policy_types.md

# 2. Rebuild to reflect changes
cd backend && python -c "from knowledge_base.loader import build_vector_store; build_vector_store()"
```

#### **Custom Configuration Options**

Modify `backend/knowledge_base/loader.py` to customize:

- Document chunk size and overlap
- Embedding model selection
- Retrieval result count
- Similarity search thresholds
