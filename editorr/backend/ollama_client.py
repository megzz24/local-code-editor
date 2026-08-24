import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    if "embeddings" in data:
        embedding = data["embeddings"][0]
    elif "embedding" in data:
        embedding = data["embedding"]
    else:
        raise ValueError(f"Unexpected Ollama response shape: {list(data.keys())}")

    if not embedding:
        raise ValueError(
            f"Ollama returned an empty embedding for model '{EMBED_MODEL}'. "
            "Check that the model is fully loaded: `ollama run nomic-embed-text`"
        )

    return embedding


def chat_completion(messages: list[dict], temperature: float = 0.2) -> str:
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]