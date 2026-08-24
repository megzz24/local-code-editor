import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chat_service import answer_question
from edit_service import apply_changes, generate_edit
from ollama_client import chat_completion
from workspace_indexer import index_workspace, reindex_files

app = FastAPI(title="Local AI Coding Assistant")


class IndexRequest(BaseModel):
    workspace_path: str


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class EditRequest(BaseModel):
    request: str


class ApplyRequest(BaseModel):
    changes: list[dict]


class ConverseRequest(BaseModel):
    message: str
    history: list[dict] = []
    pending_changes: list[dict] = []
    indexed: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index(req: IndexRequest):
    workspace = Path(req.workspace_path).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {workspace}")
    result = index_workspace(str(workspace))
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return answer_question(req.question, history=req.history)


@app.post("/edit")
def edit(req: EditRequest):
    if not req.request.strip():
        raise HTTPException(status_code=400, detail="Edit request cannot be empty")
    return generate_edit(req.request)


@app.post("/apply")
def apply(req: ApplyRequest):
    if not req.changes:
        raise HTTPException(status_code=400, detail="No changes provided")
    result = apply_changes(req.changes)
    if result["applied"]:
        reindex_result = reindex_files(result["applied"])
        result["reindex"] = reindex_result
    return result


@app.post("/converse")
def converse(req: ConverseRequest):
    """
    Single endpoint for the conversational UI.
    Uses the LLM to classify intent, then dispatches to the right action.
    Returns a structured response the UI renders directly.
    """
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    intent_messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier for a coding assistant. "
                "Given the user's message, output exactly one JSON object and nothing else.\n\n"
                "Schema:\n"
                '{ "intent": "<intent>", "value": "<extracted value or empty string>" }\n\n'
                "Intents:\n"
                '- "index"  → user wants to index/scan/load a workspace or directory path. '
                '  value = the file path they mentioned, or "" if none given.\n'
                '- "edit"   → user wants to change, add, refactor, fix, or modify code.\n'
                '  value = the edit request verbatim.\n'
                '- "apply"  → user is confirming they want to apply pending changes '
                '  (yes / ok / do it / apply / looks good / go ahead / sure).\n'
                '  value = "".\n'
                '- "reject" → user is declining pending changes '
                '  (no / cancel / discard / skip / nevermind).\n'
                '  value = "".\n'
                '- "chat"   → anything else: questions, explanations, "how does X work", etc.\n'
                '  value = the question verbatim.\n\n'
                "Output only valid JSON. No markdown, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"pending_changes_present: {bool(req.pending_changes)}\n"
                f"message: {msg}"
            ),
        },
    ]

    import json

    raw_intent = chat_completion(intent_messages)
    try:
        clean = raw_intent.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        intent = parsed.get("intent", "chat").lower()
        value  = parsed.get("value", msg).strip() or msg
    except Exception:
        intent = "chat"
        value  = msg

    if intent == "index":
        if not value or value == msg:
            return {
                "action": "need_path",
                "text": "Which directory would you like to index?",
                "history": req.history,
                "pending_changes": req.pending_changes,
                "indexed": req.indexed,
            }
        workspace = Path(value).expanduser().resolve()
        if not workspace.exists() or not workspace.is_dir():
            return {
                "action": "error",
                "text": f"Path doesn't exist or isn't a directory: {workspace}",
                "history": req.history,
                "pending_changes": req.pending_changes,
                "indexed": req.indexed,
            }
        result = index_workspace(str(workspace))
        return {
            "action": "indexed",
            "text": (
                f"Indexed **{result['files_indexed']}** files · "
                f"**{result['chunks_indexed']}** chunks."
            ),
            "index_result": result,
            "history": req.history,
            "pending_changes": [],
            "indexed": True,
        }

    if intent == "edit":
        if not req.indexed:
            return {
                "action": "error",
                "text": "Please index a workspace before making edits.",
                "history": req.history,
                "pending_changes": req.pending_changes,
                "indexed": False,
            }
        result = generate_edit(value)
        if result.get("error"):
            return {
                "action": "error",
                "text": result["error"],
                "history": req.history,
                "pending_changes": req.pending_changes,
                "indexed": req.indexed,
            }
        changes = result.get("changes", [])
        if not changes:
            return {
                "action": "no_changes",
                "text": "No changes were generated. Try rephrasing your request.",
                "raw_response": result.get("raw_response"),
                "history": req.history,
                "pending_changes": [],
                "indexed": req.indexed,
            }
        return {
            "action": "changes_ready",
            "text": f"Here {'are' if len(changes) > 1 else 'is'} the proposed change{'s' if len(changes) > 1 else ''}. Apply them?",
            "changes": changes,
            "history": req.history,
            "pending_changes": changes,
            "indexed": req.indexed,
        }

    if intent == "apply":
        if not req.pending_changes:
            return {
                "action": "chat",
                "text": "There are no pending changes to apply.",
                "history": req.history,
                "pending_changes": [],
                "indexed": req.indexed,
            }
        apply_result = apply_changes(req.pending_changes)
        reindex_result = None
        if apply_result["applied"]:
            reindex_result = reindex_files(apply_result["applied"])
        return {
            "action": "applied",
            "text": f"Applied {len(apply_result['applied'])} file(s).",
            "apply_result": apply_result,
            "reindex_result": reindex_result,
            "history": req.history,
            "pending_changes": [],
            "indexed": req.indexed,
        }

    if intent == "reject":
        return {
            "action": "rejected",
            "text": "Changes discarded.",
            "history": req.history,
            "pending_changes": [],
            "indexed": req.indexed,
        }

    if not req.indexed:
        return {
            "action": "chat",
            "text": (
                "I don't have any indexed code to search yet. "
                "Tell me a directory path to index and I'll get started."
            ),
            "history": req.history,
            "pending_changes": req.pending_changes,
            "indexed": False,
        }
    result = answer_question(value, history=req.history)
    return {
        "action": "chat",
        "text": result["answer"],
        "sources": result.get("sources", []),
        "history": result.get("history", req.history),
        "pending_changes": req.pending_changes,
        "indexed": req.indexed,
    }