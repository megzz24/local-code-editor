import numpy as np

import config
import db
import embed
import embedder


def normalize_scores(scores: list) -> list:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _hydrate(symbol_ids: list, fts_scores: dict, sem_scores: dict, merged_scores: dict) -> list:
    """Fetch full symbol data and attach score fields."""
    symbols = db.get_symbols_by_ids(symbol_ids)
    results = []
    for sym in symbols:
        sid = sym["id"]
        sym["fts_score"] = fts_scores.get(sid, 0.0)
        sym["semantic_score"] = sem_scores.get(sid, 0.0)
        sym["merged_score"] = merged_scores.get(sid, 0.0)
        results.append(sym)
    return results


def hybrid_search(query: str, top_k: int = config.TOP_K, embed_model: str = config.EMBED_MODEL) -> list:
    """Run FTS + semantic search, normalize, merge, and return ranked symbols."""
    fetch_k = top_k * 2

    # FTS results
    fts_raw = db.fts_search(query, fetch_k)
    fts_ids = [r["symbol_id"] for r in fts_raw]
    fts_raw_scores = {r["symbol_id"]: r["fts_score"] for r in fts_raw}

    # Semantic results
    sem_raw = []
    try:
        qv = embed.generate_embedding(query, embed_model)
        sem_raw = embedder.semantic_search(qv, fetch_k)
    except Exception as e:
        print(f"Semantic search failed, falling back to FTS only: {e}")
        return keyword_only_search(query, top_k)

    sem_ids = [r["symbol_id"] for r in sem_raw]
    sem_raw_scores = {r["symbol_id"]: r["cosine_score"] for r in sem_raw}

    # Normalize each set independently
    norm_fts = normalize_scores([fts_raw_scores[sid] for sid in fts_ids])
    norm_fts_map = dict(zip(fts_ids, norm_fts))

    norm_sem = normalize_scores([sem_raw_scores[sid] for sid in sem_ids])
    norm_sem_map = dict(zip(sem_ids, norm_sem))

    # Merge
    all_ids = set(fts_ids) | set(sem_ids)
    merged = {}
    for sid in all_ids:
        f = norm_fts_map.get(sid, 0.0)
        s = norm_sem_map.get(sid, 0.0)
        merged[sid] = config.FTS_WEIGHT * f + config.SEMANTIC_WEIGHT * s

    # Filter and rank
    filtered = {sid: score for sid, score in merged.items() if score >= config.MIN_SCORE_THRESHOLD}
    ranked = sorted(filtered, key=lambda sid: filtered[sid], reverse=True)[:top_k]

    return _hydrate(ranked, norm_fts_map, norm_sem_map, filtered)


def keyword_only_search(query: str, top_k: int = config.TOP_K) -> list:
    """FTS-only search, used as fallback when Ollama is unreachable."""
    fts_raw = db.fts_search(query, top_k)
    fts_ids = [r["symbol_id"] for r in fts_raw]
    fts_raw_scores = {r["symbol_id"]: r["fts_score"] for r in fts_raw}

    norm_fts = normalize_scores([fts_raw_scores[sid] for sid in fts_ids])
    norm_fts_map = dict(zip(fts_ids, norm_fts))
    merged_map = {sid: config.FTS_WEIGHT * score for sid, score in norm_fts_map.items()}

    return _hydrate(fts_ids, norm_fts_map, {}, merged_map)


def semantic_only_search(query_vector: np.ndarray, top_k: int = config.TOP_K) -> list:
    """Semantic-only search using a pre-computed query vector."""
    sem_raw = embedder.semantic_search(query_vector, top_k)
    sem_ids = [r["symbol_id"] for r in sem_raw]
    sem_raw_scores = {r["symbol_id"]: r["cosine_score"] for r in sem_raw}

    norm_sem = normalize_scores([sem_raw_scores[sid] for sid in sem_ids])
    norm_sem_map = dict(zip(sem_ids, norm_sem))
    merged_map = {sid: config.SEMANTIC_WEIGHT * s for sid, s in norm_sem_map.items()}

    return _hydrate(sem_ids, {}, norm_sem_map, merged_map)