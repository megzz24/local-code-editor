from ollama_client import chat_completion
from retrieval import retrieve


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    chunks = retrieve(question, n_results=15)

    if not chunks:
        return {
            "answer": "No indexed code found. Please index a workspace first.",
            "sources": [],
            "history": history or [],
        }

    context_parts = []
    sources = []
    seen_files: set[str] = set()

    for chunk in chunks:
        meta = chunk["metadata"]
        context_parts.append(
            f"# {meta['file_path']} — {meta['symbol_type']}: {meta['symbol_name']} "
            f"(lines {meta['start_line']}–{meta['end_line']})\n"
            f"{chunk['chunk_text']}"
        )
        if meta["file_path"] not in seen_files:
            sources.append({
                "file_path": meta["file_path"],
                "symbol_name": meta["symbol_name"],
                "symbol_type": meta["symbol_type"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
            })
            seen_files.add(meta["file_path"])

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful coding assistant. Answer questions about the codebase "
                "using ONLY the provided context. Rules:\n"
                "1. Only reference code that appears in the context — do not guess or invent functions or variables.\n"
                "2. If the answer is not in the context, say so explicitly and suggest which file to look in.\n"
                "3. Always mention the file path and line numbers when referencing specific code.\n"
                "4. If a question spans multiple files, explain how they connect.\n"
                "5. Be concise — do not over-explain code that wasn't asked about.\n"
                "6. Answer in plain English — do not quote raw code unless it is essential to the answer.\n"
                "7. If something doesn't exist in the codebase, say so directly rather than suggesting where it might be.\n"
            ),
        },
    ]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"Code context:\n\n{context}\n\nQuestion: {question}",
    })

    answer = chat_completion(messages, temperature=0.4)

    updated_history = list(history) if history else []
    updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources, "history": updated_history}