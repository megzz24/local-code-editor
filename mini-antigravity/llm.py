import json

import requests

import config
import context_manager
import db


def get_chat_model() -> str:
    return db.get_setting("chat_model", config.CHAT_MODEL)


def stream_chat(messages: list, system_prompt: str, model: str = None):
    """
    Stream a chat response from Ollama.
    Yields string chunks. Raises StopIteration when done.
    """
    if model is None:
        model = get_chat_model()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": full_messages, "stream": True},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        yield "[ERROR: Ollama unreachable]"
        return

    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("error"):
            yield f"[ERROR: {data['error']}]"
            return

        chunk = data.get("message", {}).get("content", "")
        if chunk:
            yield chunk

        if data.get("done"):
            return


def chat_with_context(
    query: str,
    history: list,
    retrieved_symbols: list,
    base_system_prompt: str,
    model: str = None,
):
    """Main entry point for RAG chat. Assembles context, streams response."""
    ctx = context_manager.assemble_context(retrieved_symbols)
    system_prompt = context_manager.format_system_prompt(base_system_prompt, ctx)
    messages = history + [{"role": "user", "content": query}]
    yield from stream_chat(messages, system_prompt, model)


def generate_patch(prompt: str, model: str = None) -> str:
    """Non-streaming generation for structured patch output."""
    if model is None:
        model = get_chat_model()

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {config.OLLAMA_BASE_URL}")