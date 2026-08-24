import hashlib
from pathlib import Path

from ast_parser import parse_file
from ollama_client import get_embedding
from vector_store import add_chunks, delete_chunks_by_file, reset_collection


def scan_python_files(workspace_path: str) -> list[str]:
    root = Path(workspace_path)
    return [str(p.resolve()) for p in root.rglob("*.py") if p.is_file()]


def make_chunk_id(file_path: str, symbol_name: str, start_line: int) -> str:
    raw = f"{file_path}:{symbol_name}:{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()


def _embed_and_store(file_paths: list[str]) -> dict:
    total_chunks = 0
    failed_files = []

    for file_path in file_paths:
        try:
            chunks = parse_file(file_path)
            batch = []

            for chunk in chunks:
                if not chunk.chunk_text or not chunk.chunk_text.strip():
                    continue

                embed_text = (
                    f"file: {Path(chunk.file_path).name} "
                    f"type: {chunk.symbol_type} "
                    f"name: {chunk.symbol_name}\n"
                    f"{chunk.chunk_text}"
                )

                try:
                    embedding = get_embedding(embed_text)
                except Exception as e:
                    print(f"Skipping chunk {chunk.symbol_name} in {file_path}: {e}")
                    continue

                if not embedding:
                    continue

                chunk_id = make_chunk_id(file_path, chunk.symbol_name, chunk.start_line)
                batch.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "chunk_text": chunk.chunk_text,
                    "metadata": {
                        "file_path": chunk.file_path,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                })

            if batch:
                add_chunks(batch)
                total_chunks += len(batch)

        except Exception as e:
            failed_files.append({"file": file_path, "error": str(e)})

    return {"chunks_indexed": total_chunks, "failed_files": failed_files}


def index_workspace(workspace_path: str) -> dict:
    reset_collection()
    files = scan_python_files(workspace_path)
    result = _embed_and_store(files)
    return {
        "files_indexed": len(files) - len(result["failed_files"]),
        "chunks_indexed": result["chunks_indexed"],
        "failed_files": result["failed_files"],
    }


def reindex_files(file_paths: list[str]) -> dict:
    for fp in file_paths:
        delete_chunks_by_file(fp)
    result = _embed_and_store(file_paths)
    return {
        "files_reindexed": len(file_paths) - len(result["failed_files"]),
        "chunks_indexed": result["chunks_indexed"],
        "failed_files": result["failed_files"],
    }