import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeChunk:
    file_path: str
    chunk_text: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int


def parse_file(file_path: str) -> list[CodeChunk]:
    path = Path(file_path)
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    chunks = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        chunks.append(CodeChunk(
            file_path=file_path,
            chunk_text=source,
            symbol_name="<module>",
            symbol_type="module",
            start_line=1,
            end_line=len(lines),
        ))
        return chunks

    top_level_lines = []
    
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_text = "\n".join(lines[node.lineno - 1:node.end_lineno])
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_text=chunk_text,
                symbol_name=node.name,
                symbol_type="function",
                start_line=node.lineno,
                end_line=node.end_lineno, # type: ignore
            ))

        elif isinstance(node, ast.ClassDef):
            chunk_text = "\n".join(lines[node.lineno - 1:node.end_lineno])
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_text=chunk_text,
                symbol_name=node.name,
                symbol_type="class",
                start_line=node.lineno,
                end_line=node.end_lineno, # type: ignore
            ))

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_text = "\n".join(lines[item.lineno - 1:item.end_lineno])
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        chunk_text=method_text,
                        symbol_name=f"{node.name}.{item.name}",
                        symbol_type="method",
                        start_line=item.lineno,
                        end_line=item.end_lineno, # type: ignore
                    ))
                    
        else:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                top_level_lines.append(
                    "\n".join(
                        lines[node.lineno - 1: node.end_lineno]
                    )
                )

    module_text = "\n\n".join(top_level_lines).strip()
    if module_text:
        chunks.append(
            CodeChunk(
                file_path=file_path,
                chunk_text=module_text,
                symbol_name="<module>",
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
            )
        )

    if not chunks:
        chunks.append(CodeChunk(
            file_path=file_path,
            chunk_text=source,
            symbol_name="<module>",
            symbol_type="module",
            start_line=1,
            end_line=len(lines),
        ))

    return chunks