import chromadb
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "code_chunks"

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(DATA_DIR))
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection():
    global _collection
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict]):
    collection = get_collection()

    for i, chunk in enumerate(chunks):
        emb = chunk["embedding"]
            
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["chunk_text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def delete_chunks_by_file(file_path: str):
    collection = get_collection()
    results = collection.get(where={"file_path": file_path})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def query_chunks(embedding: list[float], n_results: int = 5) -> list[dict]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, count),
    )

    chunks = []
    for i, doc_id in enumerate(results["ids"][0]):
        chunks.append({
            "id": doc_id,
            "chunk_text": results["documents"][0][i], # type: ignore
            "metadata": results["metadatas"][0][i], # type: ignore
            "distance": results["distances"][0][i], # type: ignore
        })
    return chunks