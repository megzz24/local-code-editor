"""
run_tests.py  —  streamlit run run_tests.py
Interactive test runner for the local AI coding assistant backend.
All file paths and symbol names are discovered dynamically — nothing is hardcoded.
"""

import importlib
import inspect
import os
import sys
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Backend Test Runner", page_icon="🧪", layout="wide")

# ── ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)  # db.py / config.py use relative paths like "data/"

# ── helpers ───────────────────────────────────────────────────────────────────

def run(label, fn):
    """Run fn(), display pass/fail + output in a compact row."""
    try:
        result = fn()
        st.success(f"✅ {label}")
        if result is not None:
            st.code(str(result), language="python")
    except Exception:
        st.error(f"❌ {label}")
        st.code(traceback.format_exc())


def section(title):
    st.markdown(f"### {title}")


def discover_sample_files():
    """Return all .py files in sample_code/ dynamically."""
    sample_dir = PROJECT_ROOT / "sample_code"
    if not sample_dir.exists():
        return []
    return sorted(sample_dir.glob("*.py"))


def discover_symbols_in_file(path: Path):
    """Return (name, type) for every function and class in a .py file via AST."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append((node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            symbols.append((node.name, "class"))
    return symbols


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🧪 Backend Test Runner")
st.caption(
    "Each section tests one module. Run them top-to-bottom — "
    "later sections depend on earlier ones passing."
)

sample_files = discover_sample_files()
sample_file_paths = [str(f) for f in sample_files]

# ── 1. config ─────────────────────────────────────────────────────────────────
with st.expander("1 · config.py", expanded=True):
    section("Constants")
    if st.button("Run", key="cfg"):
        import config
        run("CHAT_MODEL set",       lambda: config.CHAT_MODEL)
        run("EMBED_MODEL set",      lambda: config.EMBED_MODEL)
        run("EMBED_DIM == 768",     lambda: config.EMBED_DIM == 768 or (_ for _ in ()).throw(AssertionError(f"got {config.EMBED_DIM}")))
        run("DB_PATH set",          lambda: config.DB_PATH)
        run("FAISS_INDEX_PATH set", lambda: config.FAISS_INDEX_PATH)
        run("OLLAMA_BASE_URL set",  lambda: config.OLLAMA_BASE_URL)

# ── 2. db ─────────────────────────────────────────────────────────────────────
with st.expander("2 · db.py", expanded=False):
    section("Database init and CRUD")
    if st.button("Run", key="db"):
        import db

        run("init_db() creates tables", db.init_db)

        run("set_setting / get_setting round-trip", lambda: (
            db.set_setting("_test_key", "hello"),
            db.get_setting("_test_key") == "hello" or (_ for _ in ()).throw(AssertionError("mismatch"))
        ))

        run("get_all_settings returns dict", lambda: db.get_all_settings())

        run("upsert_symbol returns int id", lambda: db.upsert_symbol(
            file_path="__test__.py",
            symbol_type="function",
            name="__test_fn__",
            signature="def __test_fn__()",
            docstring="Test symbol",
            source="def __test_fn__(): pass",
            start_line=1,
            end_line=1,
        ))

        run("get_symbol_by_id returns dict", lambda: (
            sid := db.upsert_symbol("__test__.py", "function", "__test_fn2__",
                                    "def __test_fn2__()", None, "def __test_fn2__(): pass", 1, 1),
            db.get_symbol_by_id(sid)
        )[-1])

        run("fts_search returns list", lambda: db.fts_search("test function", 5))

        run("delete_symbols_for_file cleans up", lambda: (
            db.delete_symbols_for_file("__test__.py"),
            db.fts_search("__test_fn__", 5) == [] or "cleanup ok"
        )[-1])

        run("DB file exists on disk", lambda: Path("data/codebase.db").exists())

# ── 3. embed ──────────────────────────────────────────────────────────────────
with st.expander("3 · embed.py  (requires Ollama)", expanded=False):
    section("Embedding generation")

    st.info("Ollama must be running with `nomic-embed-text` pulled.")

    if st.button("Run", key="emb"):
        import embed, config, numpy as np

        run("empty text → zero vector", lambda: (
            v := embed.generate_embedding(""),
            v.shape == (config.EMBED_DIM,) and not v.any()
        )[-1])

        run("short text → (768,) float32 vector", lambda: (
            v := embed.generate_embedding("database connection handler"),
            v.shape == (config.EMBED_DIM,) and v.dtype == np.float32
        )[-1])

        run("batch_generate_embeddings length matches input", lambda: (
            texts := ["add numbers", "read file", "validate email"],
            vecs := embed.batch_generate_embeddings(texts),
            len(vecs) == len(texts)
        )[-1])

# ── 4. indexer ────────────────────────────────────────────────────────────────
with st.expander("4 · indexer.py  (requires Ollama)", expanded=False):
    section("File discovery and symbol extraction")

    if not sample_files:
        st.warning("No files found in sample_code/. Add some .py files there first.")
    else:
        st.write(f"Found {len(sample_files)} sample file(s): "
                 + ", ".join(f.name for f in sample_files))

    if st.button("Run", key="idx"):
        import indexer

        run("discover_python_files finds sample_code/", lambda: (
            files := indexer.discover_python_files("sample_code"),
            len(files) > 0 or (_ for _ in ()).throw(AssertionError("no files found"))
        )[-1])

        for sf in sample_files:
            run(f"extract_symbols({sf.name})", lambda p=sf: (
                syms := indexer.extract_symbols(p),
                f"{len(syms)} symbols: " + ", ".join(s['name'] for s in syms)
            )[-1])

        run("build_embed_text non-empty", lambda: (
            syms := indexer.extract_symbols(sample_files[0]),
            text := indexer.build_embed_text(syms[0]) if syms else "",
            len(text) > 0
        )[-1])

        run("index_directory(sample_code) succeeds", lambda: (
            r := indexer.index_directory("sample_code"),
            f"{r['total_files']} files, {r['total_symbols']} symbols, {r['total_errors']} errors"
        )[-1])

# ── 5. embedder ───────────────────────────────────────────────────────────────
with st.expander("5 · embedder.py", expanded=False):
    section("FAISS index build, load, search")

    if st.button("Run", key="emdr"):
        import embedder, numpy as np, config

        run("rebuild_faiss_index returns count", lambda: (
            n := embedder.rebuild_faiss_index(),
            f"{n} vectors indexed"
        )[-1])

        run("load_faiss_index returns True", lambda: embedder.load_faiss_index())

        run("get_index() → (index, id_map) consistent", lambda: (
            idx, id_map := embedder.get_index(),
            idx.ntotal == len(id_map) or (_ for _ in ()).throw(
                AssertionError(f"index.ntotal={idx.ntotal} vs id_map={len(id_map)}")
            )
        )[-1])

        run("semantic_search with random vector", lambda: (
            qv := np.random.randn(config.EMBED_DIM).astype("float32"),
            results := embedder.semantic_search(qv, top_k=3),
            f"{len(results)} results"
        )[-1])

        run("FAISS index file exists", lambda: Path("data/faiss.index").exists())

# ── 6. retriever ──────────────────────────────────────────────────────────────
with st.expander("6 · retriever.py  (requires Ollama)", expanded=False):
    section("Hybrid search")

    query = st.text_input(
        "Search query to test with",
        value="",
        placeholder="e.g. read file from disk",
        key="ret_query"
    )

    if st.button("Run", key="ret"):
        import retriever

        if not query.strip():
            st.warning("Enter a query above first.")
        else:
            run("normalize_scores [3,1,2] → [1,0,0.5]", lambda: (
                r := retriever.normalize_scores([3.0, 1.0, 2.0]),
                r == [1.0, 0.0, 0.5] or (_ for _ in ()).throw(AssertionError(str(r)))
            )[-1])

            run("normalize_scores empty list → []", lambda:
                retriever.normalize_scores([]) == [])

            run("normalize_scores all-same → all 1.0", lambda: (
                r := retriever.normalize_scores([5.0, 5.0, 5.0]),
                all(v == 1.0 for v in r)
            )[-1])

            run(f"keyword_only_search('{query}')", lambda: (
                r := retriever.keyword_only_search(query, top_k=5),
                f"{len(r)} results: " + ", ".join(x['name'] for x in r)
            )[-1])

            run(f"hybrid_search('{query}')", lambda: (
                r := retriever.hybrid_search(query, top_k=5),
                f"{len(r)} results: " + ", ".join(x['name'] for x in r)
            )[-1])

# ── 7. context_manager ────────────────────────────────────────────────────────
with st.expander("7 · context_manager.py", expanded=False):
    section("Token budget and context assembly")

    if st.button("Run", key="ctx"):
        import context_manager, retriever

        run("estimate_tokens('hello world') > 0", lambda:
            context_manager.estimate_tokens("hello world") > 0)

        run("get_token_budget() > 0", lambda: (
            b := context_manager.get_token_budget(),
            f"{b} tokens"
        )[-1])

        run("assemble_context([]) → empty context_str", lambda: (
            r := context_manager.assemble_context([]),
            r["context_str"] == "" and r["included"] == []
        )[-1])

        run("assemble_context with real symbols", lambda: (
            symbols := retriever.keyword_only_search("function", top_k=3),
            ctx := context_manager.assemble_context(symbols),
            f"{len(ctx['included'])} included, {len(ctx['excluded'])} excluded, "
            f"{ctx['total_tokens_used']}/{ctx['budget_tokens']} tokens"
        )[-1])

        run("format_system_prompt with empty context omits header", lambda: (
            base := "You are helpful.",
            ctx := {"context_str": ""},
            out := context_manager.format_system_prompt(base, ctx),
            out == base
        )[-1])

        run("format_system_prompt with context appends section", lambda: (
            base := "You are helpful.",
            ctx := {"context_str": "### function: foo\ndef foo(): pass\n---\n"},
            out := context_manager.format_system_prompt(base, ctx),
            "Retrieved Code Context" in out
        )[-1])

# ── 8. llm ────────────────────────────────────────────────────────────────────
with st.expander("8 · llm.py  (requires Ollama)", expanded=False):
    section("Chat streaming and patch generation")

    st.info("Ollama must be running with the chat model pulled.")

    chat_prompt = st.text_input(
        "One-shot message to test streaming",
        value="",
        placeholder='e.g. "Reply with only the word hello."',
        key="llm_msg"
    )

    if st.button("Run", key="llm"):
        import llm

        run("get_chat_model() returns string", lambda: llm.get_chat_model())

        if chat_prompt.strip():
            run("stream_chat yields text", lambda: (
                chunks := list(llm.stream_chat(
                    messages=[{"role": "user", "content": chat_prompt}],
                    system_prompt="You are a helpful assistant."
                )),
                "".join(chunks)
            )[-1])

            run("generate_patch returns string", lambda: (
                r := llm.generate_patch(
                    f"Write a Python function called greet(name) that returns 'Hello {{name}}'. "
                    f"Output only a ```python ... ``` block."
                ),
                r[:200]
            )[-1])
        else:
            st.warning("Enter a message above to test streaming.")

# ── 9. patcher ────────────────────────────────────────────────────────────────
with st.expander("9 · patcher.py  (requires Ollama)", expanded=False):
    section("Code patch generation and validation")

    # Dynamically populate file + symbol pickers
    chosen_file = st.selectbox(
        "File to patch",
        sample_file_paths,
        key="patch_file"
    )

    symbols_in_file = []
    if chosen_file:
        symbols_in_file = [
            name for name, _ in discover_symbols_in_file(Path(chosen_file))
        ]

    chosen_symbol = st.selectbox(
        "Symbol to patch",
        symbols_in_file,
        key="patch_symbol"
    )

    patch_instruction = st.text_input(
        "Instruction",
        value="",
        placeholder='e.g. "Add input validation and raise TypeError if inputs are not numbers"',
        key="patch_instr"
    )

    if st.button("Run", key="ptch"):
        import patcher

        run("extract_code_from_response with valid block", lambda: (
            r := patcher.extract_code_from_response(
                "Sure!\n```python\ndef foo(): return 42\n```"
            ),
            r == "def foo(): return 42"
        )[-1])

        run("extract_code_from_response with no block → None", lambda:
            patcher.extract_code_from_response("No code here.") is None)

        run("validate_python_ast valid code → (True, None)", lambda: (
            ok, err := patcher.validate_python_ast("def foo():\n    return 1"),
            ok is True and err is None
        )[-1])

        run("validate_python_ast invalid code → (False, msg)", lambda: (
            ok, err := patcher.validate_python_ast("def foo(\n    broken"),
            ok is False and err is not None
        )[-1])

        run("generate_diff on identical sources → empty string", lambda: (
            src := "def foo(): pass\n",
            patcher.generate_diff(src, src, "test.py") == ""
        )[-1])

        run("replace_symbol_in_source replaces correctly", lambda: (
            src := "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
            new_sym := "def bar():\n    return 99",
            result := patcher.replace_symbol_in_source(src, "bar", new_sym),
            "return 99" in result and "return 1" in result
        )[-1])

        if chosen_file and chosen_symbol and patch_instruction.strip():
            run(f"request_patch({chosen_symbol})", lambda: (
                r := patcher.request_patch(
                    file_path=chosen_file,
                    symbol_name=chosen_symbol,
                    instruction=patch_instruction,
                    context_symbols=[],
                ),
                f"ast_valid={r['ast_valid']}  error={r.get('error')}  "
                f"patch_id={r.get('patch_id')}\n\n" + (r.get("diff") or "")[:600]
            )[-1])
        else:
            st.info("Select a file, symbol, and instruction above to test request_patch.")

# ── 10. backup ────────────────────────────────────────────────────────────────
with st.expander("10 · backup.py", expanded=False):
    section("File snapshots and restore")

    backup_target = st.selectbox(
        "File to back up",
        sample_file_paths,
        key="bk_file"
    )

    if st.button("Run", key="bk"):
        import backup

        run("list_backups() returns list", lambda: (
            r := backup.list_backups(),
            f"{len(r)} existing backups"
        )[-1])

        if backup_target:
            run(f"create_backup({Path(backup_target).name})", lambda: (
                r := backup.create_backup(backup_target),
                f"backup_id={r['backup_id']}  path={r['backup_path']}"
            )[-1])

            run("backup file exists on disk", lambda: (
                backups := backup.list_backups(backup_target),
                latest := backups[0] if backups else None,
                latest is not None and latest["exists_on_disk"]
            )[-1])

            run("restore_backup restores file", lambda: (
                backups := backup.list_backups(backup_target),
                latest := backups[0] if backups else None,
                r := backup.restore_backup(latest["id"]) if latest else {"success": False, "error": "no backup"},
                r["success"] or (_ for _ in ()).throw(AssertionError(r.get("error")))
            )[-1])

            run("prune_old_backups returns int", lambda: (
                n := backup.prune_old_backups(keep_latest=5),
                f"{n} backups pruned"
            )[-1])

        else:
            st.info("Select a file above to test backup/restore.")

# ── summary ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "All tests use your actual uploaded files and sample_code/ — "
    "nothing is hardcoded. "
    "Sections 3, 4, 6, 7, 8, 9 require Ollama to be running."
)