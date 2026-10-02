"""Clones a git repo and yields the files worth indexing."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo

# Extensions we actually want to index. Deliberately excludes lockfiles,
# images, build output — noise that would pollute retrieval.
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java",
    ".md", ".json", ".yaml", ".yml",
}

# Directories to skip entirely, regardless of what's in them.
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "venv", ".venv",
    "__pycache__", ".next", "target", "coverage",
}

MAX_FILE_SIZE_BYTES = 200_000  # skip anything pathologically large (data dumps, etc.)


@dataclass
class SourceFile:
    path: str  # relative path within the repo, used in citations
    content: str
    language: str  # coarse language tag, used by the chunker to pick a strategy


def _detect_language(path: Path) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(path.suffix, "text")


def clone_and_load(repo_url: str) -> list[SourceFile]:
    """Clones repo_url to a temp dir, reads indexable files, then cleans up
    the clone (we only need the file contents, not the working tree)."""
    tmp_dir = tempfile.mkdtemp(prefix="rag_repo_")
    try:
        Repo.clone_from(repo_url, tmp_dir, depth=1)
        return _load_from_dir(Path(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_from_dir(root: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in INDEXABLE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable — skip rather than crash ingestion
        if not content.strip():
            continue
        files.append(
            SourceFile(
                path=str(path.relative_to(root)),
                content=content,
                language=_detect_language(path),
            )
        )
    return files
