import ast
import difflib
import re
from pathlib import Path

import backup
import db
import llm


def build_patch_prompt(file_path, symbol_name, current_source, instruction, context_symbols):
    ctx_text = ""
    for sym in context_symbols:
        snippet = (sym.get("source") or "")[:500]
        ctx_text += f"\n# {sym.get('name', '')}\n{snippet}\n"

    return (
        f"You are a Python coding assistant. Modify the following code as instructed.\n\n"
        f"Instruction: {instruction}\n\n"
        f"File: {file_path}\n"
        f"Symbol: {symbol_name}\n\n"
        f"Current code:\n```python\n{current_source}\n```\n\n"
        f"Related context:\n{ctx_text}\n\n"
        "Output ONLY the replacement code in a ```python ... ``` block. "
        "Do not include any explanation or text outside the code block."
    )


def extract_code_from_response(llm_response):
    match = re.search(r"```python\s*(.*?)\s*```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_python_ast(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def generate_diff(original_source, new_source, file_path):
    diff = difflib.unified_diff(
        original_source.splitlines(keepends=True),
        new_source.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "".join(diff)


def replace_symbol_in_source(file_source, symbol_name, new_symbol_source):
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        return None

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol_name:
                target = node
                break

    if target is None:
        return None

    lines = file_source.splitlines(keepends=True)
    start = target.lineno - 1
    end = target.end_lineno

    new_lines = new_symbol_source.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    return "".join(lines[:start] + new_lines + lines[end:])


def request_patch(file_path, symbol_name, instruction, context_symbols):
    try:
        current_source = Path(file_path).read_text(encoding="utf-8")
    except OSError as e:
        return {"error": str(e), "ast_valid": False}

    prompt = build_patch_prompt(file_path, symbol_name, current_source, instruction, context_symbols)

    try:
        raw_response = llm.generate_patch(prompt)
    except Exception as e:
        return {"error": f"LLM error: {e}", "ast_valid": False}

    code = extract_code_from_response(raw_response)
    if code is None:
        return {"error": "LLM did not return a code block", "ast_valid": False}

    valid, ast_error = validate_python_ast(code)
    if not valid:
        return {"error": f"AST validation failed: {ast_error}", "ast_valid": False}

    new_source = replace_symbol_in_source(current_source, symbol_name, code)
    if new_source is None:
        return {"error": f"Symbol '{symbol_name}' not found in file", "ast_valid": True}

    diff = generate_diff(current_source, new_source, file_path)

    # Store new_source in metadata so apply_patch can write it directly
    # without parsing the diff — difflib.restore is unreliable
    metadata = {
        "instruction": instruction,
        "symbol": symbol_name,
        "new_source": new_source,
    }
    patch_id = db.save_patch(file_path, diff, metadata)

    return {
        "patch_id": patch_id,
        "diff": diff,
        "status": "pending",
        "ast_valid": True,
        "error": None,
    }


def apply_patch(patch_id):
    patches = db.get_patches(status_filter="pending")
    patch = next((p for p in patches if p["id"] == patch_id), None)
    if patch is None:
        return {"success": False, "error": "Patch not found or not pending", "backup_id": None}

    file_path = patch["file_path"]
    metadata = patch.get("metadata") or {}
    new_source = metadata.get("new_source")

    if not new_source:
        return {"success": False, "error": "Patch record is missing new_source — regenerate the patch", "backup_id": None}

    # Backup first — never write without one
    try:
        backup_result = backup.create_backup(file_path, patch_id)
    except IOError as e:
        return {"success": False, "error": f"Backup failed: {e}", "backup_id": None}

    try:
        Path(file_path).write_text(new_source, encoding="utf-8")
    except OSError as e:
        db.update_patch_status(patch_id, "rejected")
        return {"success": False, "error": str(e), "backup_id": backup_result["backup_id"]}

    db.update_patch_status(patch_id, "applied")
    return {"success": True, "backup_id": backup_result["backup_id"], "error": None}


def reject_patch(patch_id):
    db.update_patch_status(patch_id, "rejected")