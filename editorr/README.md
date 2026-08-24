# Local AI Coding Assistant

A terminal-based coding assistant that runs entirely on your machine. It indexes your Python codebase into a local vector store, answers questions about your code, and proposes and applies edits — all powered by a local LLM via Ollama.

---

## Stack

| Component | Role |
|-----------|------|
| [Ollama](https://ollama.com) | Runs the LLM and embedding model locally |
| Qwen2.5-Coder 7B | Code chat and edit generation |
| nomic-embed-text | Embedding model for semantic search |
| ChromaDB | Local vector store for code chunks |
| FastAPI | Backend API server |
| Rich | Terminal UI rendering |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- The following Ollama models pulled:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

---

## Installation

```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn httpx chromadb rich pydantic
```

---

## Usage

**1. Start the backend:**

```bash
uvicorn main:app --port 8000
```

**2. Start the UI** (optionally pass a workspace path to auto-index):

```bash
python cli/ui.py /path/to/your/project
```

Or start without a path and index later by telling the assistant:

```
index /path/to/your/project
```

---

## What You Can Do

**Ask questions about your code:**
```
how does the order total get calculated?
where is the embedding model called?
what does reindex_files do?
```

**Request edits:**
```
rename display_name to full_name in the User class
add a created_at timestamp to the Order class
add a quantity check in add_item to not exceed stock
```

**Apply or discard proposed changes:**
```
yes        → applies the pending changes and reindexes affected files
no         → discards the changes
```

---

## How It Works

### Indexing

When you index a workspace, the assistant:

1. Recursively scans for all `.py` files
2. Parses each file with Python's `ast` module, splitting it into chunks by function, method, class, and module-level code
3. Embeds each chunk using `nomic-embed-text` (with file name and symbol type prepended for better retrieval)
4. Stores all chunks in a local ChromaDB collection

### Chat

Questions are answered by:

1. Embedding the question and retrieving the 15 most relevant chunks from ChromaDB
2. Passing those chunks as context to `Qwen2.5-Coder` with a system prompt that restricts answers to the provided context
3. Returning the answer along with source references (file, symbol, line numbers)

### Editing

Edit requests go through a multi-step pipeline:

1. **Multi-query expansion** — the LLM generates 3 sub-queries targeting callers, importers, and related symbols, which are run alongside the original request to surface all affected files
2. **Edit generation** — all candidate file contents are sent to the LLM with a strict prompt that forbids unrequested changes, added comments, or omitted code
3. **Diff rendering** — proposed changes are shown as unified diffs before anything is written to disk
4. **Apply + reindex** — on confirmation, files are written and reindexed so the vector store stays current

### Intent Classification

Every message goes through an LLM-based intent classifier that routes it to one of five actions: `index`, `edit`, `apply`, `reject`, or `chat`. This means you can just talk naturally — no commands to remember.

---

## Project Structure

```
.
├── main.py                # FastAPI server and intent dispatcher
├── cli/
│   └── ui.py              # Rich terminal UI and REPL loop
├── chat_service.py        # Question answering with RAG
├── edit_service.py        # Edit generation and application
├── workspace_indexer.py   # File scanning and embedding pipeline
├── ast_parser.py          # Python AST-based code chunking
├── retrieval.py           # Embedding query entry point
├── vector_store.py        # ChromaDB wrapper
├── ollama_client.py       # Ollama API client
├── diff_utils.py          # Unified diff generation
└── data/
    └── chroma/            # Persisted vector store (auto-created)
```

---

## Configuration

All model and URL settings are in `ollama_client.py`:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL      = "qwen2.5-coder:7b"
EMBED_MODEL     = "nomic-embed-text"
```

Temperature is controlled per call site via the `temperature` parameter on `chat_completion`:

| Call site | Temperature | Reason |
|-----------|-------------|--------|
| Intent classifier | 0.2 | Deterministic JSON output |
| Sub-query generation | 0.2 | Focused retrieval queries |
| Edit generation | 0.2 | Precise, consistent code changes |
| Chat answers | 0.4 | Natural, readable responses |

---

## Known Limitations

- Only indexes `.py` files (no JS, TS, etc.)
- Context window limits apply — very large files may be truncated
- Edit quality depends on how specifically you phrase the request; vague requests produce more spurious changes
- The vector store is wiped and rebuilt on every full index