"""
Text Chunking for Semantic Index

Sliding window chunker for RAG-style indexing.

Pure Python - no LLM tokens consumed. Runs as batch operation.

Ported from animus (animus/memory/chunker.py). No behavior changes.
"""

from __future__ import annotations

import re
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


@dataclass(frozen=True)
class ParagraphChunk:
    """A child chunk: content == original_text[char_start:char_end]."""

    ordinal: int
    char_start: int
    char_end: int
    content: str


_PARA_BREAK = re.compile(r"\n[ \t]*\n+")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Stripped (start, end) spans of the paragraphs of text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in list(_PARA_BREAK.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        nxt = m.end() if m else len(text)
        seg = text[pos:end]
        lead = len(seg) - len(seg.lstrip())
        stripped = seg.strip()
        if stripped:
            spans.append((pos + lead, pos + lead + len(stripped)))
        pos = nxt
    return spans


def _split_long(text: str, start: int, end: int, max_chars: int, overlap: int):
    """Split one over-long paragraph at whitespace into <= max_chars pieces."""
    pieces: list[tuple[int, int]] = []
    s = start
    while s < end:
        e = min(s + max_chars, end)
        if e < end:
            ws = max(text.rfind(" ", s + 1, e), text.rfind("\n", s + 1, e))
            if ws > s:
                e = ws
        piece = text[s:e]
        lead = len(piece) - len(piece.lstrip())
        stripped = piece.strip()
        if stripped:
            pieces.append((s + lead, s + lead + len(stripped)))
        if e >= end:
            break
        # next piece starts `overlap` chars back, moved forward to a word start
        ns = max(e - overlap, s + 1)
        while ns < e and not text[ns - 1].isspace():
            ns += 1
        s = ns if ns < e else e
    return pieces


def chunk_paragraphs(
    text: str, max_chars: int = 1000, overlap: int = 100
) -> list[ParagraphChunk]:
    """Child chunks of at most max_chars at paragraph boundaries.

    Paragraphs (blank-line separated) are packed greedily; a paragraph longer
    than max_chars is split at whitespace. Consecutive chunks share a small
    overlap: the trailing units of a chunk (paragraphs or pieces) totalling at
    most `overlap` chars are repeated at the start of the next. Offsets index
    the original text exactly.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars // 2))
    units: list[tuple[int, int]] = []
    for s, e in _paragraph_spans(text):
        if e - s <= max_chars:
            units.append((s, e))
        else:
            units.extend(_split_long(text, s, e, max_chars, overlap))
    chunks: list[ParagraphChunk] = []
    i = 0
    while i < len(units):
        j = i + 1
        while j < len(units) and units[j][1] - units[i][0] <= max_chars:
            j += 1
        chunks.append(ParagraphChunk(
            len(chunks), units[i][0], units[j - 1][1],
            text[units[i][0]:units[j - 1][1]],
        ))
        if j >= len(units):
            break
        # next chunk starts at the earliest unit within `overlap` of this end
        # (never at i itself: it must make progress)
        k = j
        while k - 1 > i and units[j - 1][1] - units[k - 1][0] <= overlap:
            k -= 1
        i = k
    return chunks
