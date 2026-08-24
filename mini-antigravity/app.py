import sys
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Codebase Assistant", page_icon="🧠", layout="wide")

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


@st.cache_resource(show_spinner="Starting up…")
def _init():
    import db, embedder
    db.init_db()
    embedder.load_faiss_index()


try:
    _init()
except Exception as e:
    st.error(f"Startup failed: {e}")
    st.stop()


if "history" not in st.session_state:
    st.session_state.history = []
if "pending_patch" not in st.session_state:
    st.session_state.pending_patch = None


# ── intent classification ─────────────────────────────────────────────────────
# Explicit rules first — only call LLM if rules don't match

PATCH_KEYWORDS = [
    "add ", "fix ", "refactor ", "rewrite ", "modify ", "update ",
    "change ", "remove ", "rename ", "implement ", "edit ",
]
SEARCH_KEYWORDS = [
    "show me", "find ", "list ", "search ", "which functions",
    "what functions", "where is", "look for", "all functions",
    "all classes", "show all",
]

CLASSIFIER_PROMPT = """Classify the user message into exactly one of:
SEARCH - finding or listing code symbols
PATCH  - modifying or fixing a specific function or class
CHAT   - asking a question or discussing code

Rules:
- If the message asks HOW something works, what something DOES, or WHY — it is CHAT
- If the message asks to SHOW, FIND, LIST, or SEARCH for symbols — it is SEARCH  
- If the message asks to ADD, FIX, CHANGE, REWRITE, or MODIFY code — it is PATCH

Reply with ONE word only: SEARCH, PATCH, or CHAT"""


def classify_intent(message):
    msg = message.lower().strip()

    # Rule-based fast path — avoids an extra LLM call for obvious cases
    if any(msg.startswith(kw) or f" {kw}" in msg for kw in PATCH_KEYWORDS):
        # Only patch if a file or function name looks present
        if "." in message or "def " in message or "function" in message or "class" in message:
            return "PATCH"

    if any(msg.startswith(kw) or msg.startswith(kw.strip()) for kw in SEARCH_KEYWORDS):
        return "SEARCH"

    # Questions about how/what/why → always CHAT
    for word in ["how does", "how do", "what does", "what is", "why does", "explain", "describe"]:
        if word in msg:
            return "CHAT"

    # Fall back to LLM classifier
    import llm
    chunks = llm.stream_chat(
        messages=[{"role": "user", "content": message}],
        system_prompt=CLASSIFIER_PROMPT,
    )
    result = "".join(chunks).strip().upper()
    for intent in ("SEARCH", "PATCH", "CHAT"):
        if intent in result:
            return intent
    return "CHAT"


# ── patch intent parser ───────────────────────────────────────────────────────

PATCH_PARSER_PROMPT = """Extract patch details from the user message.

Return ONLY a JSON object with:
- file_path: the file path (e.g. "sample_code/mathUtils.py"), or null
- symbol_name: the exact function or class name, or null
- instruction: the full change to make

Example:
{"file_path": "sample_code/mathUtils.py", "symbol_name": "divide", "instruction": "Add TypeError if inputs are not numbers"}

No extra text."""


def parse_patch_intent(message):
    import llm, json
    chunks = llm.stream_chat(
        messages=[{"role": "user", "content": message}],
        system_prompt=PATCH_PARSER_PROMPT,
    )
    raw = "".join(chunks).strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"file_path": None, "symbol_name": None, "instruction": message}


# ── handlers ──────────────────────────────────────────────────────────────────

def handle_search(query):
    import retriever
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            results = retriever.hybrid_search(query, top_k=8)

        if not results:
            msg = "No symbols found. Try different keywords or index your codebase first."
            st.write(msg)
            return msg

        st.markdown(f"Found **{len(results)} symbol(s)**:")
        for r in results:
            with st.expander(
                f"**{r['name']}** · `{r['file_path']}` · score `{r['merged_score']:.2f}`",
                expanded=False,
            ):
                if r.get("docstring"):
                    st.caption(r["docstring"])
                st.code(r.get("source", ""), language="python")

        return (
            f"Found {len(results)} symbol(s): "
            + ", ".join(f"`{r['name']}`" for r in results)
        )


def handle_patch(message):
    import retriever, patcher

    with st.chat_message("assistant"):
        with st.spinner("Understanding patch request…"):
            parsed = parse_patch_intent(message)

        file_path = parsed.get("file_path")
        symbol_name = parsed.get("symbol_name")
        instruction = parsed.get("instruction") or message

        if not file_path or not symbol_name:
            missing = []
            if not file_path:
                missing.append("the file path")
            if not symbol_name:
                missing.append("the function or class name")
            reply = (
                f"I need {' and '.join(missing)} to generate a patch. "
                f"Try something like:\n\n"
                f"> `add input validation to divide in sample_code/mathUtils.py`"
            )
            st.write(reply)
            return reply

        with st.spinner(f"Generating patch for `{symbol_name}` in `{file_path}`…"):
            hits = retriever.hybrid_search(f"{symbol_name} {instruction}", top_k=4)
            result = patcher.request_patch(
                file_path=file_path,
                symbol_name=symbol_name,
                instruction=instruction,
                context_symbols=hits,
            )

        if result.get("error"):
            reply = f"Patch failed: {result['error']}"
            st.error(reply)
            return reply

        st.success(f"Patch ready for `{symbol_name}` in `{file_path}`")
        st.code(result["diff"], language="diff")

        col_apply, col_reject = st.columns(2)
        with col_apply:
            if st.button("✅ Apply", key=f"apply_{result['patch_id']}", use_container_width=True):
                res = patcher.apply_patch(result["patch_id"])
                if res["success"]:
                    st.success(f"Applied. Backup id={res['backup_id']}")
                    # Show the updated file content to confirm
                    try:
                        updated = Path(file_path).read_text(encoding="utf-8")
                        with st.expander("Updated file", expanded=True):
                            st.code(updated, language="python")
                    except Exception:
                        pass
                    st.session_state.pending_patch = None
                else:
                    st.error(f"Apply failed: {res['error']}")
        with col_reject:
            if st.button("❌ Reject", key=f"reject_{result['patch_id']}", use_container_width=True):
                patcher.reject_patch(result["patch_id"])
                st.info("Patch rejected.")
                st.session_state.pending_patch = None
                st.rerun()

        st.session_state.pending_patch = result
        return f"Generated patch for `{symbol_name}` in `{file_path}`. Diff shown above."


def handle_chat(query):
    import retriever, context_manager, llm

    BASE_PROMPT = (
        "You are an expert code assistant. Answer questions about the codebase "
        "using the retrieved code context provided. "
        "If the context does not contain enough information, say so. "
        "Be concise and precise."
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context…"):
            hits = retriever.hybrid_search(query, top_k=8)
            ctx = context_manager.assemble_context(hits)

        response_box = st.empty()
        chunks = []

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.history
        ]

        for chunk in llm.chat_with_context(
            query=query,
            history=history,
            retrieved_symbols=hits,
            base_system_prompt=BASE_PROMPT,
        ):
            chunks.append(chunk)
            response_box.markdown("".join(chunks))

        full_response = "".join(chunks)

        if hits:
            with st.expander(
                f"📎 {len(ctx['included'])} symbol(s) used as context",
                expanded=False,
            ):
                for sym in ctx["included"]:
                    st.markdown(
                        f"**{sym['name']}** `{sym['file_path']}` "
                        f"· score `{sym['merged_score']:.2f}`"
                    )
                    st.code(sym.get("source", ""), language="python")

        return full_response


def dispatch(message):
    with st.spinner("Thinking…"):
        intent = classify_intent(message)

    if intent == "SEARCH":
        return handle_search(message), intent
    elif intent == "PATCH":
        return handle_patch(message), intent
    else:
        return handle_chat(message), intent


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Codebase Assistant")
    st.divider()

    try:
        import db, embedder
        conn = db.get_connection()
        sym_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        conn.close()
        idx, _ = embedder.get_index()
        st.markdown(f"**Symbols indexed:** {sym_count}")
        st.markdown(f"**Vectors in FAISS:** {idx.ntotal}")
    except Exception:
        st.warning("Index not ready")

    st.divider()
    st.markdown("**Index a folder**")
    root = st.text_input("Folder path", value="sample_code", label_visibility="collapsed")

    if st.button("Index now", use_container_width=True):
        import indexer, db as _db

        # Wipe existing symbols for this folder first to avoid duplicates
        root_resolved = str(Path(root).resolve())
        bar = st.progress(0, text="Scanning…")

        def cb(cur, total, fname):
            bar.progress(cur / total, text=f"[{cur}/{total}] {Path(fname).name}")

        with st.spinner("Indexing…"):
            r = indexer.index_directory(root, progress_callback=cb)
        bar.empty()
        st.success(f"{r['total_files']} files · {r['total_symbols']} symbols")
        if r["total_errors"]:
            st.warning(f"{r['total_errors']} errors")
        st.rerun()

    st.divider()

    # Clear index button to fix duplicate symbols
    if st.button("🗑 Clear index", use_container_width=True):
        import db as _db, embedder as _embedder
        with _db.get_connection() as conn:
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM symbols")
        _embedder.rebuild_faiss_index()
        st.success("Index cleared. Re-index your folder.")
        st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending_patch = None
        st.rerun()

    st.divider()
    st.caption(
        "Type naturally:\n"
        "- *how does X work?* → chat\n"
        "- *show me all math functions* → search\n"
        "- *add validation to divide in mathUtils.py* → patch"
    )


# ── main ──────────────────────────────────────────────────────────────────────

st.title("🧠 Codebase Assistant")

if not st.session_state.history:
    st.caption("Index a folder in the sidebar, then type anything.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("kind"):
            st.caption(f"_{msg['kind'].lower()}_")

if message := st.chat_input("Ask, search, or request a patch…"):
    with st.chat_message("user"):
        st.markdown(message)

    st.session_state.history.append({"role": "user", "content": message})
    response, intent = dispatch(message)
    st.session_state.history.append({
        "role": "assistant",
        "content": response,
        "kind": intent,
    })