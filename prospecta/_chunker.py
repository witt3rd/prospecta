"""
Text Chunking for Semantic Index

Sliding window chunker for RAG-style indexing.

Pure Python - no LLM tokens consumed. Runs as batch operation.

Ported from animus (animus/memory/chunker.py). No behavior changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Chunk:
    """A chunk of text with source metadata."""

    content: str
    source_path: Path
    chunk_index: int
    start_char: int
    end_char: int

    @property
    def id(self) -> str:
        """Unique ID for this chunk."""
        # Use path + chunk index for stable IDs
        path_str = str(self.source_path).replace("\\", "/").replace("/", "_")
        return f"{path_str}__chunk_{self.chunk_index}"

    def to_metadata(self) -> dict:
        """Build metadata dict for vector store."""
        return {
            "source_path": str(self.source_path),
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def chunk_text(
    content: str,
    source_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> Iterator[Chunk]:
    """Split text into overlapping chunks.

    Parameters
    ----------
    content : str
        Text content to chunk.
    source_path : Path
        Source file path (for metadata).
    chunk_size : int
        Target size of each chunk in characters. Default 1000.
    overlap : int
        Overlap between consecutive chunks in characters. Default 200.

    Yields
    ------
    Chunk
        Chunk objects with content and metadata.
    """
    if not content or not content.strip():
        return

    # For small content, return as single chunk
    if len(content) <= chunk_size:
        yield Chunk(
            content=content,
            source_path=source_path,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
        )
        return

    # Sliding window with overlap
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2  # Fallback if overlap >= chunk_size

    chunk_index = 0
    start = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to break at word boundary
        if end < len(content):
            # Look backwards for whitespace
            search_start = max(start + chunk_size - 100, start)
            last_space = content.rfind(" ", search_start, end)
            if last_space > search_start:
                end = last_space + 1

        chunk_content = content[start:end].strip()

        if chunk_content:
            yield Chunk(
                content=chunk_content,
                source_path=source_path,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
            )
            chunk_index += 1

        start += step

        # Avoid tiny final chunks
        if len(content) - start < overlap:
            break


def chunk_file(
    path: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    encoding: str = "utf-8",
) -> Iterator[Chunk]:
    """Read and chunk a file.

    Parameters
    ----------
    path : Path
        File to read and chunk.
    chunk_size : int
        Target chunk size in characters.
    overlap : int
        Overlap between chunks in characters.
    encoding : str
        File encoding. Default utf-8.

    Yields
    ------
    Chunk
        Chunk objects with content and metadata.
    """
    try:
        content = path.read_text(encoding=encoding)
    except (UnicodeDecodeError, OSError):
        # Skip files that can't be read
        return

    yield from chunk_text(content, path, chunk_size, overlap)
