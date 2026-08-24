import os

# Model Settings
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Search & Retrieval
TOP_K = 8
FTS_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
MIN_SCORE_THRESHOLD = 0.1

# Token Budget
MAX_CONTEXT_FRACTION = 0.30
MODEL_CONTEXT_WINDOW = 8192
CHARS_PER_TOKEN = 4

# Paths
DB_PATH = "data/codebase.db"
FAISS_INDEX_PATH = "data/faiss.index"
BACKUP_DIR = "data/backups/"
INDEX_ROOT = "."

# Patch Settings
PATCH_HISTORY_LIMIT = 50
BACKUP_RETENTION_LIMIT = 100