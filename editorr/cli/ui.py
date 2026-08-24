import sys
import time
from pathlib import Path

import httpx
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.padding import Padding

# ── Theme ────────────────────────────────────────────────────────────────────

theme = Theme({
    "accent":     "bold #89B4FA",
    "dim_accent": "#6E9EF5",
    "ok":         "bold #A6E3A1",
    "warn":       "bold #F9E2AF",
    "err":        "bold #F38BA8",
    "muted":      "#9399B2",
    "code":       "#CDD6F4",
    "file":       "italic #89DCEB",
    "symbol":     "#A6E3A1",
    "prompt":     "bold #CBA6F7",
})

console = Console(theme=theme, highlight=False)
API_BASE = "http://localhost:8000"

# ── Helpers ──────────────────────────────────────────────────────────────────

def api_post(path: str, payload: dict, timeout: float = 300.0) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        if r.status_code != 200:
            console.print(f"[err]✗ API error {r.status_code}:[/] {r.text}")
            return None
        return r.json()
    except httpx.ConnectError:
        console.print(
            Panel(
                "[err]Cannot reach the backend.[/]\n"
                "[muted]Start it with:[/] [code]uvicorn main:app --port 8000[/]",
                border_style="err",
                padding=(0, 2),
            )
        )
        return None
    except Exception as e:
        console.print(f"[err]✗ Request failed:[/] {e}")
        return None


def rule(label: str = "", style: str = "dim"):
    console.print(Rule(label, style=style))


def show_index_result(result: dict):
    files  = result["files_indexed"]
    chunks = result["chunks_indexed"]
    failed = result.get("failed_files", [])

    summary = Table.grid(padding=(0, 3))
    summary.add_column(justify="right", style="muted")
    summary.add_column()
    summary.add_row("files",  f"[ok bold]{files}[/]")
    summary.add_row("chunks", f"[ok bold]{chunks}[/]")

    console.print(
        Panel(
            Padding(summary, (0, 1)),
            title="[ok bold]✓ Index complete [/]",
            border_style="ok",
            padding=(0, 2),
        )
    )
    if failed:
        console.print(f"\n  [warn]⚠  {len(failed)} file(s) skipped[/]")
        for f in failed[:5]:
            console.print(f"    [muted]{f['file']}[/]  [err]{f['error']}[/]")
        if len(failed) > 5:
            console.print(f"    [muted]…and {len(failed) - 5} more[/]")


def show_changes(changes: list[dict]):
    console.print()
    for change in changes:
        fp = change["file_path"]
        console.print(
            Panel(
                Syntax(
                    change["diff"],
                    "diff",
                    theme="github-dark",
                    line_numbers=False,
                    word_wrap=False,
                ),
                title=f"[file]{Path(fp).name}[/]  [muted]{fp}[/]",
                border_style="dim_accent",
                padding=(0, 1),
            )
        )


# def show_sources(sources: list[dict]):
#     if not sources:
#         return
#     console.print()
#     parts: list[Text] = []
#     for src in sources[:6]:
#         fp  = Path(src["file_path"]).name
#         sym = src["symbol_name"]
#         ln  = f"{src['start_line']}–{src['end_line']}"
#         t   = Text()
#         t.append(fp,        style="file")
#         t.append(f"  {sym}", style="symbol")
#         t.append(f"  :{ln}", style="muted")
#         parts.append(t)
#     console.print(Columns(parts, padding=(0, 2), expand=False))


def show_apply_result(apply_result: dict, reindex_result: dict | None):
    console.print()
    for fp in apply_result.get("applied", []):
        console.print(f"  [ok]✓[/]  [file]{fp}[/]")
    for err in apply_result.get("errors", []):
        console.print(f"  [err]✗[/]  [file]{err['file_path']}[/]  [muted]{err['error']}[/]")
    if reindex_result:
        console.print(
            f"\n  [muted]↺  Reindexed {reindex_result['files_reindexed']} file(s) "
            f"· {reindex_result['chunks_indexed']} chunks[/]"
        )
        for f in reindex_result.get("failed_files", []):
            console.print(f"  [warn]⚠  Reindex failed:[/] [file]{f['file']}[/]  [muted]{f['error']}[/]")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold]local ai coding assistant[/]\n"
                "[muted]ask questions · request edits · index /path/to/project[/]",
                justify="center"
            ),
            border_style="dim",
            padding=(1, 4),
        )
    )

    history: list[dict] = []
    pending_changes: list[dict] = []
    indexed = False

    # optional: auto-index from CLI arg
    if len(sys.argv) > 1:
        path = str(Path(sys.argv[1]).expanduser().resolve())
        console.print(f"\n  [muted]Indexing[/] [file]{path}[/]…\n")
        with console.status("[accent]Building embeddings…[/]", spinner="dots", spinner_style="accent"):
            result = api_post("/index", {"workspace_path": path})
        if result:
            show_index_result(result)
            indexed = True

    while True:
        console.print()
        
        if not indexed:
            console.print(f"[warn]Please index a workspace to get started")

        # show a hint when changes are waiting
        if pending_changes:
            console.print(
                f"  [warn]↑ {len(pending_changes)} change(s) pending[/]  "
                '[muted]say "yes" to apply or "no" to discard, or just keep going[/]'
            )

        try:
            raw = Prompt.ask("  [prompt]You[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [muted]Goodbye.[/]\n")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "bye", "q"):
            console.print("\n  [muted]Goodbye.[/]\n")
            break

        t0 = time.time()

        with console.status("[accent]…[/]", spinner="dots2", spinner_style="accent"):
            result = api_post(
                "/converse",
                {
                    "message":          raw,
                    "history":          history,
                    "pending_changes":  pending_changes,
                    "indexed":          indexed,
                },
                timeout=300.0,
            )

        if result is None:
            continue

        elapsed  = f"{time.time() - t0:.1f}s"
        action   = result.get("action", "chat")
        text     = result.get("text", "")
        history  = result.get("history", history)
        indexed  = result.get("indexed", indexed)
        pending_changes = result.get("pending_changes", pending_changes)

        console.print()

        # ── render by action type ──────────────────────────────────────────

        if action == "indexed":
            show_index_result(result["index_result"])

        elif action == "need_path":
            console.print(f"  [warn]{text}[/]")
            try:
                path = Prompt.ask("  [prompt]path[/]").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not path:
                continue
            with console.status("[accent]Building embeddings…[/]", spinner="dots", spinner_style="accent"):
                idx_result = api_post("/index", {"workspace_path": path})
            if idx_result:
                show_index_result(idx_result)
                indexed = True

        elif action == "changes_ready":
            show_changes(result.get("changes", []))
            console.print(f"\n  [accent]{text}[/]")

        elif action == "applied":
            show_apply_result(result["apply_result"], result.get("reindex_result"))
            console.print(f"\n  [ok]{text}[/]")
            # after applying, keep chatting about the same codebase
            if text and "question" in raw.lower() or any(
                w in raw.lower() for w in ("what", "how", "why", "where", "which", "does", "is", "are", "can")
            ):
                pass  # history already updated server-side

        elif action == "rejected":
            console.print(f"  [muted]{text}[/]")

        elif action == "no_changes":
            console.print(f"  [warn]{text}[/]")
            raw_resp = result.get("raw_response")
            if raw_resp:
                console.print(Panel(raw_resp, title="model output", border_style="dim", padding=(0, 2)))

        elif action == "error":
            console.print(f"  [err]✗  {text}[/]")

        else:
            # chat answer
            console.print(
                Panel(
                    Markdown(text),
                    title=f"[ok bold]assistant[/]  [muted]{elapsed}[/]",
                    border_style="dim_accent",
                    padding=(0, 2),
                )
            )
            # show_sources(result.get("sources", []))


if __name__ == "__main__":
    main()