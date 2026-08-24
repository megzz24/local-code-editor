import re
from pathlib import Path

from diff_utils import generate_unified_diff
from ollama_client import chat_completion
from retrieval import retrieve

def get_candidate_files(edit_request: str, n_results: int = 15) -> list[str]:
    chunks = retrieve(edit_request, n_results=n_results)

    # deduplicate chunks by id
    seen_ids: set[str] = set()
    unique_chunks = []
    for c in chunks:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            unique_chunks.append(c)

    unique_chunks.sort(key=lambda c: c["distance"])

    # extract unique files
    seen_files: set[str] = set()
    files = []
    for chunk in unique_chunks:
        fp = chunk["metadata"]["file_path"]
        if fp not in seen_files:
            seen_files.add(fp)
            files.append(fp)

    return files


def get_candidate_files_expanded(edit_request: str) -> list[str]:
    # ask the LLM for sub-queries that would surface callers / importers
    sub_query_msgs = [
        {
            "role": "system",
            "content": (
                "Output exactly 3 short search queries (one per line, no bullets) "
                "to find Python files that USE or CALL the symbol being changed. "
                "Focus on: files that access the attribute, files that pass it as an argument, "
                "files that store it in a dict or list. Do NOT generate queries about where it is defined."
            )
        },
        {"role": "user", "content": edit_request},
    ]
    raw = chat_completion(sub_query_msgs)
    sub_queries = [l.strip() for l in raw.splitlines() if l.strip()][:3]

    queries = [edit_request] + sub_queries
    print(f"[debug] expansion queries: {queries}")

    seen_files: set[str] = set()
    files: list[str] = []
    for q in queries:
        for fp in get_candidate_files(q, n_results=10):
            if fp not in seen_files:
                seen_files.add(fp)
                files.append(fp)

    print(f"[debug] candidate files: {files}")
    return files


def generate_edit(edit_request: str) -> dict:
    candidate_files = get_candidate_files_expanded(edit_request)

    if not candidate_files:
        return {"error": "No relevant files found. Index a workspace first.", "changes": []}

    file_contents: dict[str, str] = {}
    for fp in candidate_files:
        path = Path(fp)
        if path.exists():
            file_contents[fp] = path.read_text(encoding="utf-8")

    if not file_contents:
        return {"error": "Candidate files could not be read.", "changes": []}

    files_context = "\n\n".join(
        f"=== FILE: {fp} ===\n{content}"
        for fp, content in file_contents.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding assistant. Before making any edits, you must first output a plan, "
                "then output the code changes.\n\n"
                "Always use this exact format:\n\n"
                "PLAN:\n"
                "<list every file and exactly what changes are needed>\n\n"
                "CHANGES:\n"
                "FILE: <exact_file_path>\n"
                "```python\n"
                "<complete file content>\n"
                "```\n\n"
                "Rules:\n"
                    "1. Write the PLAN first — think through all affected files before writing code.\n"
                    "2. ONLY implement what is explicitly stated in the edit request. Nothing more.\n"
                    "3. Update ALL files that call or use changed code.\n"
                    "4. Adding unrequested functions, parameters, comments, imports, or blank lines is strictly forbidden.\n"
                    "5. Keep signatures consistent across all files.\n"
                    "6. Do not improve, optimize, or clean up any existing code you were not asked to change.\n"
                    "7. If you are unsure whether a change was asked for, do NOT make it.\n"
                    "8. Before writing each file ask yourself: did the user explicitly ask for this change? "
                    "If no, do not include it.\n"
                    "9. Every function, class, and method that exists in the original file MUST exist in your output. "
                        "Deleting or omitting existing code is strictly forbidden unless explicitly asked.\n"
                    "10. If you are only adding one thing, the rest of the file must be identical to the original.\n"
                    "11. Never add comments, docstrings, or inline annotations of any kind unless explicitly asked.\n"
                    "12. Never add code to functions or methods that were not mentioned in the edit request.\n"
                    "13. The ONLY lines that should appear as additions in the diff are the exact lines needed "
                        "to implement the edit request. Every other line must be identical to the original.\n"
                    "14. Before outputting each file, count the number of changes you are making. "
                        "If that number is greater than what the edit request requires, remove the extras.\n"
            ),
        },
            {
                "role": "user",
                "content": (
                    f"The most relevant file for this edit is: {candidate_files[0]}\n\n"
                    f"{files_context}\n\n"
                    f"Edit request: {edit_request}\n\n"
                    "IMPORTANT: Only make the changes described above. "
                    "Do not add, remove, or modify anything that was not explicitly requested. "
                    "Do not add comments. Do not add code to unrelated functions. "
                    "When in doubt, leave the code exactly as it is."
                ),
            }
    ]

    response = chat_completion(messages)
    changes = _parse_edit_response(response, file_contents)

    return {
        "changes": changes,
        "candidate_files": candidate_files,
        "raw_response": response if not changes else None,
    }


def _parse_edit_response(response: str, original_contents: dict[str, str]) -> list[dict]:
    pattern = r"FILE:\s*(.+?)\s*\n\s*```(?:\w+)?\s*\n(.*?)```(?:\s|$)"
    matches = re.findall(pattern, response, re.DOTALL)
        
    changes = []
    for raw_path, new_content in matches:
        file_path = raw_path.strip()
        new_content = new_content.rstrip()

        if file_path not in original_contents:
            for known_path in original_contents:
                if known_path.endswith(file_path) or file_path.endswith(known_path):
                    file_path = known_path
                    break

        if file_path not in original_contents:
            continue

        original = original_contents[file_path]
        diff = generate_unified_diff(original, new_content, file_path)

        if diff:
            changes.append({
                "file_path": file_path,
                "original_content": original,
                "new_content": new_content,
                "diff": diff,
            })

    return changes


def apply_changes(changes: list[dict]) -> dict:
    applied = []
    errors = []

    for change in changes:
        try:
            Path(change["file_path"]).write_text(change["new_content"], encoding="utf-8")
            applied.append(change["file_path"])
        except Exception as e:
            errors.append({"file_path": change["file_path"], "error": str(e)})

    return {"applied": applied, "errors": errors}