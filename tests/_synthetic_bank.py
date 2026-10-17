"""A synthetic bank of invented text for Stage 1 performance checks.

`load_bank` writes `n_chunks` kind='chunk' items (random clustered vectors of
`dim` floats, Zipf-distributed pseudo-words) plus a few planted needles, builds
the HNSW indexes after the load and analyzes the tables. Everything is invented."""
from __future__ import annotations

import random

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from prospecta.db.queries import HNSW_INDEX_TEMPLATE, ensure_kind_hnsw_indexes, hnsw_index_name

SYLLABLES = ["ba", "ko", "ri", "tu", "ve", "lan", "mor", "sel", "dun", "fi", "gar", "hu",
             "jo", "kir", "lum", "nex", "or", "pal", "quo", "sta", "tem", "vin", "wex", "zo"]
WORDS_PER_CHUNK = 120
CHUNKS_PER_DOC = 3
# Ordinary prose is full of function words ("the", "and", ...): the very common words
# every real chunk has and the English configuration drops. `function_words` below
# gives that share of the chunks a natural sprinkle of them.
FUNCTION_WORDS = ["the", "of", "and", "a", "to", "in"]
FUNCTION_WEIGHTS = [5, 3, 3, 3, 3, 2]
FUNCTION_WORD_SHARE = 0.3      # of the word slots of a chunk that has them

# Planted needles: (document number, chunk text). Document numbers are small, so
# they exist at any corpus size of at least 100 chunks.
NEEDLES = [
    (1, "The bridge lamps were called electric dreams by the night crew of the old harbour."),
    (2, "Electric lamps burn late and the crew keeps its dreams for the morning shift."),
    (3, "The updater stopped with error 0x80070005 right after the v1.14 rollout."),
    (4, "Release v1.15 changed nothing about error 0x80070006 for the harbour crew."),
]


def vocabulary(seed: int = 7, size: int = 6000) -> list[str]:
    """The invented words, most frequent first (Zipf weights follow the order)."""
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def load_bank(url: str, bank: str, *, n_chunks: int, dim: int = 1536, seed: int = 7,
              function_words: float = 0.0) -> None:
    """`function_words`: the share of the chunks (0..1) that get function words mixed
    in; 0 leaves the text as the pseudo-words only (and the random stream untouched)."""
    rng = random.Random(seed)
    vocab = vocabulary(seed)
    noise = [(b - 127.5) / 255 for b in range(256)]      # a byte -> a value in [-0.5, 0.5]
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    n_docs = max(n_chunks // CHUNKS_PER_DOC, len(NEEDLES) + 1)
    centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(200)]

    with psycopg.connect(url, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES (%s, %s)", (bank, dim))
        conn.execute(
            "INSERT INTO documents (bank_id, source, original_text, content_hash, created_on) "
            "SELECT %s, 'doc-' || i, '', md5(i::text), DATE '2023-01-01' + (i %% 900) "
            "FROM generate_series(1, %s) i", (bank, n_docs))
        doc_ids = [r[0] for r in conn.execute(
            "SELECT id FROM documents WHERE bank_id = %s ORDER BY (regexp_replace(source, '\\D', '', 'g'))::int",
            (bank,))]
        needle_docs = {n: text for n, text in NEEDLES}
        with conn.cursor() as cur, cur.copy(
                "COPY memory_items (bank_id, document_id, content, original_chunk, embedding, "
                "kind, ordinal) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["text", "uuid", "text", "text", "vector", "text", "int4"])
            for i in range(n_chunks):
                doc_no = i // CHUNKS_PER_DOC + 1
                text = needle_docs.get(doc_no) if i % CHUNKS_PER_DOC == 0 else None
                if text is None:
                    words = rng.choices(vocab, weights, k=WORDS_PER_CHUNK)
                    if function_words and rng.random() < function_words:
                        words = [rng.choices(FUNCTION_WORDS, FUNCTION_WEIGHTS)[0]
                                 if rng.random() < FUNCTION_WORD_SHARE else w for w in words]
                    text = " ".join(words)
                vec = [c + noise[b] for c, b in zip(centers[rng.randrange(200)], rng.randbytes(dim))]
                copy.write_row([bank, doc_ids[min(doc_no, n_docs) - 1], text, text, Vector(vec),
                                "chunk", i % CHUNKS_PER_DOC])
        conn.execute("SET maintenance_work_mem = '1GB'")
        conn.execute("SET max_parallel_maintenance_workers = 0")   # parallel builds need a big /dev/shm
        conn.execute(HNSW_INDEX_TEMPLATE.format(
            index_name=hnsw_index_name(bank), bank_id_literal=bank, embedding_dim=dim))
        ensure_kind_hnsw_indexes(conn, bank, dim)
        conn.execute("ANALYZE memory_items")
        conn.execute("ANALYZE documents")


def random_query_vector(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]
