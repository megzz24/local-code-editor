import numpy as np
import requests

import config


def generate_embedding(text: str, model: str = config.EMBED_MODEL) -> np.ndarray:
    """Generate a single embedding vector for the given text."""
    if not text.strip():
        return np.zeros(config.EMBED_DIM, dtype=np.float32)

    text = text.replace("\x00", "")  # strip null bytes

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {config.OLLAMA_BASE_URL}. Is it running?")
    except requests.Timeout:
        raise TimeoutError("Ollama embedding request timed out after 30s")

    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama error (model '{model}'): {data['error']}")

    vector = np.array(data["embedding"], dtype=np.float32)
    if vector.shape != (config.EMBED_DIM,):
        raise ValueError(
            f"Expected embedding dim {config.EMBED_DIM}, got {vector.shape[0]}. "
            f"Is '{model}' the right model?"
        )
    return vector


def batch_generate_embeddings(texts: list, model: str = config.EMBED_MODEL) -> list:
    """Generate embeddings for a list of texts sequentially."""
    results = []
    for i, text in enumerate(texts):
        if i > 0 and i % 10 == 0:
            print(f"Embedding progress: {i}/{len(texts)}")
        try:
            vec = generate_embedding(text, model)
        except Exception as e:
            print(f"Warning: embedding failed for item {i}: {e}")
            vec = np.zeros(config.EMBED_DIM, dtype=np.float32)
        results.append(vec)
    return results