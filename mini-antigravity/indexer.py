import ast
import textwrap
from pathlib import Path

import config
import db
import embed
import embedder

EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "site-packages", "node_modules"}


def discover_python_files(root: str) -> list:
    """Return sorted list of all .py files under root, excluding common non-source dirs."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    files = []
    for f in root_path.rglob("*.py"):
        if not any(part in EXCLUDE_DIRS for part in f.parts):
            files.append(f)
    return sorted(files)


def _build_signature(node) -> str:
    """Build a human-readable signature string from an AST function node."""
    args = node.args
    parts = []
    # positional args
    for arg in args.args:
        annotation = ""
        if arg.annotation:
            annotation = f": {ast.unparse(arg.annotation)}"
        parts.append(f"{arg.arg}{annotation}")
    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)}){ret}"


def extract_symbols(file_path: Path) -> list:
    """Parse a Python file and extract function and class symbols."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = file_path.read_text(encoding="latin-1")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Skipping {file_path}: SyntaxError: {e}")
        return []

    source_lines = source.splitlines()
    symbols = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if it's a method (parent is ClassDef)
            name = node.name
            sig = _build_signature(node)
            docstring = ast.get_docstring(node)
            start = node.lineno
            end = node.end_lineno
            raw_source = textwrap.dedent("\n".join(source_lines[start - 1:end]))
            symbols.append({
                "symbol_type": "function",
                "name": name,
                "signature": sig,
                "docstring": docstring,
                "source": raw_source,
                "start_line": start,
                "end_line": end,
            })

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            docstring = ast.get_docstring(node)
            start = node.lineno
            end = node.end_lineno
            raw_source = textwrap.dedent("\n".join(source_lines[start - 1:end]))
            symbols.append({
                "symbol_type": "class",
                "name": node.name,
                "signature": sig,
                "docstring": docstring,
                "source": raw_source,
                "start_line": start,
                "end_line": end,
            })

    return symbols


def build_embed_text(symbol: dict) -> str:
    """Build the text string to embed for a symbol."""
    parts = [symbol.get("signature", "")]
    if symbol.get("docstring"):
        parts.append(symbol["docstring"])
    parts.append(symbol.get("source", ""))
    text = "\n\n".join(p for p in parts if p)

    if len(text) > 2000:
        text = text[:1000] + text[-1000:]  # keep start (signature) and end
    return text


def index_file(file_path: Path, embed_model: str = config.EMBED_MODEL, index_root: Path = None) -> dict:
    """Index a single Python file: extract symbols, embed, store in DB."""
    root = (index_root or Path(config.INDEX_ROOT)).resolve()
    try:
        rel_path = str(file_path.resolve().relative_to(root))
    except ValueError:
        # file is outside the root — use absolute path
        rel_path = str(file_path.resolve())
    errors = []

    db.delete_symbols_for_file(rel_path)
    symbols = extract_symbols(file_path)
    indexed = 0

    for sym in symbols:
        try:
            symbol_id = db.upsert_symbol(
                file_path=rel_path,
                symbol_type=sym["symbol_type"],
                name=sym["name"],
                signature=sym["signature"],
                docstring=sym["docstring"],
                source=sym["source"],
                start_line=sym["start_line"],
                end_line=sym["end_line"],
            )
            embed_text = build_embed_text(sym)
            vector = embed.generate_embedding(embed_text, embed_model)

            if not vector.any():  # skip zero vectors
                continue

            db.upsert_embedding(symbol_id, embed_model, vector)
            indexed += 1
        except Exception as e:
            errors.append(f"{sym['name']}: {e}")

    return {
        "file": rel_path,
        "symbols_found": len(symbols),
        "symbols_indexed": indexed,
        "errors": errors,
    }


def index_directory(root: str, embed_model: str = config.EMBED_MODEL, progress_callback=None) -> dict:
    """Index all Python files in a directory tree."""
    root_resolved = Path(root).resolve()
    files = discover_python_files(root)
    total_symbols = 0
    total_errors = 0
    per_file = []

    for i, f in enumerate(files):
        try:
            result = index_file(f, embed_model, index_root=root_resolved)
        except FileNotFoundError:
            result = {"file": str(f), "symbols_found": 0, "symbols_indexed": 0, "errors": ["File not found"]}

        per_file.append(result)
        total_symbols += result["symbols_indexed"]
        total_errors += len(result["errors"])

        if progress_callback:
            progress_callback(i + 1, len(files), str(f))

    embedder.rebuild_faiss_index(embed_model)

    return {
        "total_files": len(files),
        "total_symbols": total_symbols,
        "total_errors": total_errors,
        "per_file": per_file,
    }