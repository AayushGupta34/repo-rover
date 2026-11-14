"""Deterministic structural pre-scanner for complexity and bloat.

Regexes over 200-line chunks cannot prove a nested loop or a query inside a
loop. This scanner can: it walks the Python AST, and for JavaScript and
TypeScript it tracks loop headers and brace depth on a code-only buffer
(comments and string contents blanked).

Checks, each reported with an exact file and line:
    nested_loops   a loop inside a loop (depth >= 2)
    io_in_loop     a database or HTTP call inside a loop
    god_function   a function longer than 80 lines, or a file longer than 600
    parser_bomb    a file that exhausts the parser (recursion or memory)

Results feed the detector registry through catalogue entries whose detector
type is `ast`, and are listed for the Optimizer as verified targets.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from repo_rover.sanitize import strip_comments

MAX_FUNCTION_LINES = 80
MAX_FILE_LINES = 600
MAX_HITS_PER_CHECK_PER_FILE = 5
CHECKS = ("nested_loops", "io_in_loop", "god_function", "parser_bomb")

_PY_STRICT_IO = {"execute", "executemany", "urlopen", "fetchall", "fetchone"}
_PY_OWNER_IO = {"get", "post", "put", "patch", "delete", "request", "filter", "find", "find_one",
                "query", "send", "read", "all", "first", "save", "commit"}
_PY_IO_OWNER = re.compile(r"(?i)\b(requests|httpx|aiohttp|session|client|conn|connection|cursor|cur|db|objects|query|redis|s3|boto3)\b")

_JS_LOOP = re.compile(r"\b(?:for|while)\s*\(|\.(?:forEach|map|filter|reduce|flatMap|some|every|find)\s*\(")
_JS_IO = re.compile(
    r"\bawait\s+[\w.$\[\]]+\.(?:find\w*|query|get|post|put|delete|fetch\w*|select|execute|save|update\w*|create\w*|insert\w*|count\w*|aggregate)\s*\("
    r"|\bfetch\s*\(|\baxios\s*(?:\.\w+)?\s*\(|\.query\s*\(|\.execute\s*\("
)
_JS_FUNCTION = re.compile(r"\bfunction\b[^{;]*\{|=>\s*\{\s*$|^\s*(?:async\s+)?[\w$]+\s*\([^)]*\)\s*\{\s*$")


def _hit(check: str, file: str, lines: list[str], line: int, end_line: int | None, detail: str) -> dict[str, Any]:
    line = max(1, min(line, len(lines) or 1))
    end = max(line, min(end_line or line, line + 2, len(lines) or line))
    snippet = "\n".join(l.strip()[:240] for l in lines[line - 1:end] if l.strip()) if lines else ""
    return {"check": check, "file": file, "line": line, "end_line": end, "snippet": snippet, "detail": detail}


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, file: str, lines: list[str]):
        self.file, self.lines = file, lines
        self.depth = 0
        self.hits: list[dict[str, Any]] = []

    def _loop(self, node: ast.AST) -> None:
        self.depth += 1
        if self.depth >= 2:
            self.hits.append(_hit("nested_loops", self.file, self.lines, node.lineno, node.lineno,
                                  f"Loop nested {self.depth} deep: work grows with the product of the collections."))
        self.generic_visit(node)
        self.depth -= 1

    visit_For = visit_AsyncFor = visit_While = _loop

    def _comprehension(self, node: ast.AST) -> None:
        extra = len(getattr(node, "generators", [])) or 1
        self.depth += extra
        if self.depth >= 2:
            self.hits.append(_hit("nested_loops", self.file, self.lines, node.lineno, node.lineno,
                                  f"Comprehension nested {self.depth} deep."))
        self.generic_visit(node)
        self.depth -= extra

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    def visit_Call(self, node: ast.Call) -> None:
        if self.depth >= 1 and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            try:
                owner = ast.unparse(node.func.value)
            except Exception:
                owner = ""
            if name in _PY_STRICT_IO or (name in _PY_OWNER_IO and _PY_IO_OWNER.search(owner)):
                self.hits.append(_hit("io_in_loop", self.file, self.lines, node.lineno, node.lineno,
                                      f"`{owner}.{name}(...)` runs once per loop iteration."))
        self.generic_visit(node)

    def _function(self, node: ast.AST) -> None:
        length = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            self.hits.append(_hit("god_function", self.file, self.lines, node.lineno, node.lineno,
                                  f"Function `{node.name}` is {length} lines long (limit {MAX_FUNCTION_LINES})."))
        outer, self.depth = self.depth, 0  # a nested def starts its own loop context
        self.generic_visit(node)
        self.depth = outer

    visit_FunctionDef = visit_AsyncFunctionDef = _function


def scan_python(file: str, lines: list[str]) -> list[dict[str, Any]]:
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    except (RecursionError, MemoryError, ValueError, OverflowError) as exc:
        return [_hit("parser_bomb", file, lines, 1, 1,
                     f"The file could not be parsed safely ({type(exc).__name__}): nesting or size exhausts the parser.")]
    visitor = _PythonVisitor(file, lines)
    try:
        visitor.visit(tree)
    except (RecursionError, MemoryError) as exc:
        return [_hit("parser_bomb", file, lines, 1, 1,
                     f"Walking the syntax tree failed ({type(exc).__name__}): the code is nested too deeply.")]
    return visitor.hits


def scan_javascript(file: str, lines: list[str], extension: str = ".js") -> list[dict[str, Any]]:
    """Loop-nesting heuristic for brace languages, on a code-only buffer."""
    code_lines = strip_comments("\n".join(lines), extension, blank_strings=True).split("\n")
    hits: list[dict[str, Any]] = []
    depth = 0
    loops: list[int] = []        # brace depth at which each open loop body started
    functions: list[tuple[int, int]] = []  # (start line, brace depth before its body)

    for number, code in enumerate(code_lines, start=1):
        loops = [d for d in loops if depth > d]
        headers = list(_JS_LOOP.finditer(code))
        inline_depth = len(loops)

        for index, header in enumerate(headers):
            if inline_depth >= 1 or index >= 1:
                hits.append(_hit("nested_loops", file, lines, number, number,
                                 "Loop or array iteration nested inside another loop."))
            inline_depth += 1
        in_loop_after = headers[0].end() if headers else (0 if loops else None)
        if in_loop_after is not None:
            io = _JS_IO.search(code, in_loop_after)
            if io:
                hits.append(_hit("io_in_loop", file, lines, number, number,
                                 f"`{code[io.start():io.end()].strip()}...` runs once per iteration."))

        if _JS_FUNCTION.search(code):
            functions.append((number, depth))
        opened = code.count("{")
        closed = code.count("}")
        if headers and opened > closed:
            loops.append(depth)  # the loop body stays open past this line
        depth = max(0, depth + opened - closed)
        while functions and depth <= functions[-1][1] and number > functions[-1][0]:
            start, _ = functions.pop()
            length = number - start + 1
            if length > MAX_FUNCTION_LINES:
                hits.append(_hit("god_function", file, lines, start, start,
                                 f"Function starting here is {length} lines long (limit {MAX_FUNCTION_LINES})."))
    return hits


def scan_file(file: str, lines: list[str]) -> list[dict[str, Any]]:
    lowered = file.lower()
    extension = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    if extension == ".py":
        hits = scan_python(file, lines)
    elif extension in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        hits = scan_javascript(file, lines, extension)
    else:
        return []
    if len(lines) > MAX_FILE_LINES and not any(h["check"] == "parser_bomb" for h in hits):
        hits.append(_hit("god_function", file, lines, 1, 1,
                         f"File is {len(lines)} lines long (limit {MAX_FILE_LINES})."))
    limited: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        key = (hit["check"], hit["line"])
        if key in seen:
            continue
        seen.add(key)
        counts[hit["check"]] = counts.get(hit["check"], 0) + 1
        if counts[hit["check"]] <= MAX_HITS_PER_CHECK_PER_FILE:
            limited.append(hit)
    return limited


def scan_repository(index: Any) -> dict[str, list[dict[str, Any]]]:
    """Run every check over a detector_engine.RepoIndex. Returns check -> hits."""
    results: dict[str, list[dict[str, Any]]] = {check: [] for check in CHECKS}
    for relative in index.files:
        if not relative.lower().endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
            continue
        loaded = index.load(relative)
        if loaded is None:
            continue
        for hit in scan_file(relative, loaded[1]):
            results[hit["check"]].append(hit)
    return results
