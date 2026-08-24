import config
import db


def estimate_tokens(text: str) -> int:
    """Rough token estimate: characters / CHARS_PER_TOKEN."""
    return len(text) // config.CHARS_PER_TOKEN


def get_token_budget() -> int:
    """Return max tokens available for injected RAG context."""
    window = int(db.get_setting("model_context_window", str(config.MODEL_CONTEXT_WINDOW)))
    return int(window * config.MAX_CONTEXT_FRACTION)


def assemble_context(symbols: list, budget_tokens: int = None) -> dict:
    """
    Pack as many top-ranked symbols as fit within the token budget.
    Returns a dict with context_str, included, excluded, token counts.
    """
    if budget_tokens is None:
        budget_tokens = get_token_budget()

    included = []
    excluded = []
    context_parts = []
    used_tokens = 0

    for sym in symbols:
        header = (
            f"### {sym.get('symbol_type', 'symbol')}: {sym.get('name', '')}\n"
            f"# File: {sym.get('file_path', '')}, "
            f"Lines {sym.get('start_line', '?')}-{sym.get('end_line', '?')}\n"
            f"# Score: {sym.get('merged_score', 0):.3f}\n"
            f"{sym.get('signature', '')}\n\n"
            f"{sym.get('docstring') or ''}\n\n"
        )
        source = sym.get("source", "")
        block = header + source + "\n---\n"
        block_tokens = estimate_tokens(block)

        if used_tokens + block_tokens <= budget_tokens:
            context_parts.append(block)
            included.append(sym)
            used_tokens += block_tokens
        else:
            # Try truncating source to fit remaining budget
            remaining = budget_tokens - used_tokens
            header_tokens = estimate_tokens(header + "\n---\n")
            if remaining > header_tokens:
                chars_for_source = (remaining - header_tokens) * config.CHARS_PER_TOKEN
                truncated_source = source[:chars_for_source]
                block = header + truncated_source + "\n---\n"
                context_parts.append(block)
                included.append(sym)
                used_tokens += estimate_tokens(block)
            else:
                sym_copy = dict(sym)
                sym_copy["excluded_reason"] = "token_budget"
                excluded.append(sym_copy)

    context_str = "".join(context_parts)
    return {
        "context_str": context_str,
        "included": included,
        "excluded": excluded,
        "total_tokens_used": used_tokens,
        "budget_tokens": budget_tokens,
    }


def format_system_prompt(base_prompt: str, context: dict) -> str:
    """Combine base instructions with retrieved code context."""
    if not context.get("context_str"):
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "## Retrieved Code Context\n"
        "The following symbols were retrieved from the codebase based on your query:\n\n"
        f"{context['context_str']}"
    )