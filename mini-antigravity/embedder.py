import json
from pathlib import Path

import faiss
import numpy as np

import config
import db

_index = None
_id_map = []  # FAISS row i -> symbol_id

_IDS_PATH = config.FAISS_INDEX_PATH + ".ids"


def rebuild_faiss_index(embed_model: str = config.EMBED_MODEL) -> int:
    """Build a fresh FAISS index from all embeddings in the DB."""
    global _index, _id_map

    rows = db.get_all_embeddings(embed_model)
    rows.sort(key=lambda r: r["symbol_id"])

    _id_map = [r["symbol_id"] for r in rows]

    index = faiss.IndexFlatIP(config.EMBED_DIM)

    if rows:
        matrix = np.stack([r["vector"] for r in rows]).astype(np.float32)
        faiss.normalize_L2(matrix)
        index.add(matrix)

    Path(config.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    Path(_IDS_PATH).write_text(json.dumps(_id_map))

    _index = index
    return len(_id_map)


def load_faiss_index(embed_model: str = config.EMBED_MODEL) -> bool:
    """Load FAISS index from disk, rebuilding if missing or corrupt."""
    global _index, _id_map
    try:
        if not Path(config.FAISS_INDEX_PATH).exists() or not Path(_IDS_PATH).exists():
            rebuild_faiss_index(embed_model)
            return True
        _index = faiss.read_index(config.FAISS_INDEX_PATH)
        _id_map = json.loads(Path(_IDS_PATH).read_text())
        if len(_id_map) != _index.ntotal:
            print("Warning: FAISS index/id_map out of sync. Rebuilding...")
            rebuild_faiss_index(embed_model)
        return True
    except Exception as e:
        print(f"Failed to load FAISS index ({e}), rebuilding...")
        rebuild_faiss_index(embed_model)
        return True


def get_index():
    """Return (index, id_map), loading lazily if needed."""
    global _index, _id_map
    if _index is None:
        load_faiss_index()
    return _index, _id_map


def semantic_search(query_vector: np.ndarray, top_k: int) -> list:
    """Search the FAISS index and return top_k results with cosine scores."""
    index, id_map = get_index()

    if index.ntotal == 0:
        return []

    qv = query_vector.astype(np.float32).copy()
    faiss.normalize_L2(qv.reshape(1, -1))
    qv = qv.reshape(1, config.EMBED_DIM)

    distances, indices = index.search(qv, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "symbol_id": id_map[idx],
            "cosine_score": float(dist),
        })
    return results