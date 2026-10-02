"""Splits source files into chunks for embedding.

Blind fixed-size chunking (e.g. "every 500 characters") regularly slices a
function in half, which wrecks retrieval — the half-chunk that gets embedded
often no longer represents what the code actually does. Instead, this finds
function/class boundaries per language via lightweight regex heuristics
(not a full AST parse — tree-sitter would be the "real" version of this,
noted as a stretch goal in the README) and chunks along those boundaries.
Oversized functions get sub-split with overlap so no chunk is unbounded.
"""

import re
from dataclasses import dataclass

from app.ingestion.loader import SourceFile

MAX_CHUNK_LINES = 60
OVERLAP_LINES = 8

# Matches top-level (non-indented) def/class starts in Python.
_PY_BOUNDARY = re.compile(r"^(def |class |async def )", re.MULTILINE)

# Matches function/class/arrow-function declarations in JS/TS. Deliberately
# loose — false positives just mean an extra split point, which is harmless.
_JS_BOUNDARY = re.compile(
    r"^(export\s+)?(default\s+)?(async\s+)?"
    r"(function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(\(|async)|"
    r"interface\s+\w+|type\s+\w+\s*=)",
    re.MULTILINE,
)

_JAVA_BOUNDARY = re.compile(
    r"^\s*(public|private|protected)\s+.*\b(class|interface|\w+\s*\()",
    re.MULTILINE,
)

_MD_BOUNDARY = re.compile(r"^#{1,6}\s", re.MULTILINE)

_BOUNDARY_BY_LANGUAGE = {
    "python": _PY_BOUNDARY,
    "javascript": _JS_BOUNDARY,
    "typescript": _JS_BOUNDARY,
    "java": _JAVA_BOUNDARY,
    "markdown": _MD_BOUNDARY,
}


@dataclass
class Chunk:
    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int

    @property
    def id(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def _split_on_boundaries(lines: list[str], boundary_pattern: re.Pattern) -> list[tuple[int, int]]:
    """Returns (start_line, end_line) index pairs, 0-indexed, splitting the
    file wherever boundary_pattern matches at the start of a line."""
    boundary_line_indices = [
        i for i, line in enumerate(lines) if boundary_pattern.match(line)
    ]
    if not boundary_line_indices or boundary_line_indices[0] != 0:
        boundary_line_indices = [0] + boundary_line_indices

    spans: list[tuple[int, int]] = []
    for idx, start in enumerate(boundary_line_indices):
        end = (
            boundary_line_indices[idx + 1]
            if idx + 1 < len(boundary_line_indices)
            else len(lines)
        )
        if end > start:
            spans.append((start, end))
    return spans


def _sub_split_if_oversized(start: int, end: int) -> list[tuple[int, int]]:
    """A single function/class can still be huge; break it into
    MAX_CHUNK_LINES windows with overlap so no chunk is unbounded."""
    span_len = end - start
    if span_len <= MAX_CHUNK_LINES:
        return [(start, end)]

    windows = []
    pos = start
    while pos < end:
        window_end = min(pos + MAX_CHUNK_LINES, end)
        windows.append((pos, window_end))
        if window_end == end:
            break
        pos = window_end - OVERLAP_LINES
    return windows


def chunk_file(source: SourceFile) -> list[Chunk]:
    lines = source.content.splitlines()
    if not lines:
        return []

    boundary_pattern = _BOUNDARY_BY_LANGUAGE.get(source.language)

    if boundary_pattern is None:
        # No known boundary strategy (json/yaml/plain text) — fall back to
        # fixed-size windows across the whole file.
        raw_spans = [(0, len(lines))]
    else:
        raw_spans = _split_on_boundaries(lines, boundary_pattern)

    chunks: list[Chunk] = []
    for start, end in raw_spans:
        for sub_start, sub_end in _sub_split_if_oversized(start, end):
            text = "\n".join(lines[sub_start:sub_end]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    file_path=source.path,
                    language=source.language,
                    start_line=sub_start + 1,  # 1-indexed for human-readable citations
                    end_line=sub_end,
                )
            )
    return chunks


def chunk_files(sources: list[SourceFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for source in sources:
        all_chunks.extend(chunk_file(source))
    return all_chunks
