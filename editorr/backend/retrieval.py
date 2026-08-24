from ollama_client import get_embedding
from vector_store import query_chunks


def retrieve(query: str, n_results: int = 15) -> list[dict]:
    embedding = get_embedding(query)
    return query_chunks(embedding, n_results=n_results)