"""Synthetic Hindsight database: the real Hindsight table/column layout
(from the recorded schema), generic made-up content, scaled-down row counts."""
from __future__ import annotations

import hashlib
import json
import uuid

import psycopg

DIM = 384

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE alembic_version (version_num varchar PRIMARY KEY);
CREATE TABLE banks (bank_id text PRIMARY KEY, name text, disposition jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  mission text, last_consolidated_at timestamptz, mission_changed_at timestamptz,
  config jsonb NOT NULL DEFAULT '{}', internal_id uuid NOT NULL DEFAULT gen_random_uuid());
CREATE TABLE documents (id text NOT NULL, bank_id text NOT NULL, original_text text, content_hash text,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  retain_params jsonb, tags varchar[] NOT NULL DEFAULT '{}', file_storage_key text,
  file_original_name text, file_content_type text, PRIMARY KEY (id, bank_id));
CREATE TABLE chunks (chunk_id text PRIMARY KEY, document_id text NOT NULL, bank_id text NOT NULL,
  chunk_index int NOT NULL, chunk_text text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
  content_hash text);
CREATE TABLE memory_units (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text NOT NULL,
  document_id text, text text NOT NULL, embedding vector, context text, event_date timestamptz,
  occurred_start timestamptz, occurred_end timestamptz, mentioned_at timestamptz,
  fact_type text NOT NULL DEFAULT 'world', metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  chunk_id text, tags varchar[] NOT NULL DEFAULT '{}', proof_count int DEFAULT 1,
  source_memory_ids uuid[] DEFAULT '{}', consolidated_at timestamptz, observation_scopes jsonb,
  text_signals text, search_vector tsvector, consolidation_failed_at timestamptz, edited_at timestamptz);
CREATE TABLE entities (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), canonical_name text NOT NULL,
  bank_id text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', first_seen timestamptz NOT NULL DEFAULT now(),
  last_seen timestamptz NOT NULL DEFAULT now(), mention_count int NOT NULL DEFAULT 1,
  entity_kind text NOT NULL DEFAULT 'regular');
CREATE TABLE unit_entities (unit_id uuid NOT NULL, entity_id uuid NOT NULL, PRIMARY KEY (unit_id, entity_id));
CREATE TABLE entity_cooccurrences (entity_id_1 uuid NOT NULL, entity_id_2 uuid NOT NULL,
  cooccurrence_count int NOT NULL DEFAULT 1, last_cooccurred timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_id_1, entity_id_2));
CREATE TABLE memory_links (from_unit_id uuid NOT NULL, to_unit_id uuid NOT NULL, link_type text NOT NULL,
  entity_id uuid, weight float8 NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(),
  bank_id text NOT NULL);
CREATE UNIQUE INDEX idx_memory_links_unique ON memory_links
  (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid));
CREATE TABLE observation_history (id bigserial PRIMARY KEY, observation_id uuid NOT NULL,
  bank_id text NOT NULL, content jsonb NOT NULL, changed_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE mental_model_history (id bigserial PRIMARY KEY, mental_model_id varchar NOT NULL,
  bank_id text NOT NULL, content jsonb NOT NULL, changed_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE mental_models (id text NOT NULL DEFAULT gen_random_uuid()::text, bank_id text NOT NULL,
  name varchar NOT NULL, source_query text NOT NULL, content text NOT NULL, embedding vector,
  tags varchar[] DEFAULT '{}', last_refreshed_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(), search_vector tsvector, reflect_response jsonb,
  max_tokens int NOT NULL DEFAULT 2048, trigger jsonb NOT NULL DEFAULT '{}',
  last_refreshed_source_query text, structured_content jsonb, subtype varchar NOT NULL DEFAULT 'structural',
  description text NOT NULL DEFAULT '', entity_id uuid, observations jsonb, links varchar[],
  last_updated timestamptz, PRIMARY KEY (bank_id, id));
CREATE TABLE knowledge_pages (id varchar PRIMARY KEY, bank_id text NOT NULL, parent_id varchar,
  kind varchar NOT NULL, name text NOT NULL, mental_model_id varchar, sort_order int NOT NULL DEFAULT 0,
  managed bool NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE invalidated_memory_units (id uuid PRIMARY KEY, bank_id text NOT NULL, document_id text,
  text text NOT NULL, fact_type text NOT NULL DEFAULT 'world', metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(), invalidation_reason text,
  invalidated_at timestamptz DEFAULT now(), causal_links jsonb NOT NULL DEFAULT '[]');
CREATE TABLE directives (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text NOT NULL,
  name varchar NOT NULL, content text NOT NULL, priority int NOT NULL DEFAULT 0,
  is_active bool NOT NULL DEFAULT true, tags varchar[] DEFAULT '{}');
CREATE TABLE file_storage (storage_key text PRIMARY KEY, data bytea NOT NULL);
CREATE TABLE audit_log (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), action text NOT NULL,
  transport text NOT NULL, bank_id text, started_at timestamptz NOT NULL DEFAULT now(),
  request jsonb, response jsonb, metadata jsonb DEFAULT '{}');
CREATE TABLE llm_requests (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text, operation text,
  model text, status text NOT NULL, started_at timestamptz NOT NULL DEFAULT now(),
  input jsonb, output jsonb, error text);
CREATE TABLE async_operations (operation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text NOT NULL,
  operation_type text NOT NULL, status text NOT NULL DEFAULT 'pending');
CREATE TABLE bank_stats_cache (bank_id text PRIMARY KEY, payload jsonb NOT NULL);
CREATE TABLE graph_maintenance_queue (bank_id text NOT NULL, unit_id uuid NOT NULL, PRIMARY KEY (bank_id, unit_id));
CREATE TABLE webhooks (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text, url text NOT NULL,
  secret text, enabled bool NOT NULL DEFAULT true);
"""

# bank -> (units, links)
BANKS = {"alpha": (150, 2400), "beta": (40, 300)}


def _vec(seed: str, dim: int = DIM) -> str:
    h = hashlib.sha256(seed.encode()).digest()
    return "[" + ",".join(f"{((h[i % 32] + i) % 17 - 8) / 8:.3f}" for i in range(dim)) + "]"


def _u(seed: str) -> str:
    return str(uuid.UUID(bytes=hashlib.sha256(seed.encode()).digest()[:16]))


def build(url: str) -> dict:
    """Create the synthetic Hindsight schema+rows in `url`; return expected counts."""
    exp: dict = {}
    with psycopg.connect(url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("INSERT INTO alembic_version VALUES ('synthetic')")
        for bank, (n_units, n_links) in BANKS.items():
            cur.execute("INSERT INTO banks (bank_id, name, mission, config) VALUES (%s,%s,%s,%s)",
                        (bank, bank, "Remember generic things.", json.dumps({"k": 1})))
            for d in range(6):
                key = f"file-{bank}-{d}"
                cur.execute("INSERT INTO file_storage VALUES (%s, %s)", (key, b"\x00\x01bytes"))
                cur.execute("INSERT INTO documents (id, bank_id, original_text, content_hash, tags, "
                            "file_storage_key) VALUES (%s,%s,%s,%s,%s,%s)",
                            (f"doc-{bank}-{d}", bank, f"Source document {d} of {bank}.",
                             f"h{d}", ["src"], key))
                for ch in range(2):
                    cur.execute("INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,now(),%s)",
                                (f"chunk-{bank}-{d}-{ch}", f"doc-{bank}-{d}", bank, ch,
                                 f"Chunk {ch} text.", "ch"))
            ents = [_u(f"ent-{bank}-{i}") for i in range(30)]
            for i, e in enumerate(ents):
                cur.execute("INSERT INTO entities (id, canonical_name, bank_id, entity_kind) "
                            "VALUES (%s,%s,%s,%s)",
                            (e, f"Entity {i} {bank}", bank, "label" if i % 10 == 0 else "regular"))
            units = []
            for i in range(n_units):
                uid = _u(f"unit-{bank}-{i}")
                units.append(uid)
                ft = ("world", "experience", "observation")[i % 3]
                text = f"Generic fact number {i % 140} about topic {i % 7}."  # repeats => duplicate text
                emb = None if i % 25 == 0 else _vec(f"{bank}-{i}")
                cur.execute(
                    "INSERT INTO memory_units (id, bank_id, document_id, text, embedding, context,"
                    " fact_type, created_at, tags, proof_count, source_memory_ids, chunk_id)"
                    " VALUES (%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s)",
                    (uid, bank, f"doc-{bank}-{i % 6}", text, emb, "synthetic context", ft,
                     f"2025-01-{1 + i % 28:02d}T10:00:00Z", ["t1"] if i % 2 else [], 1 + i % 3,
                     units[:2] if ft == "observation" and i > 3 else [], f"chunk-{bank}-{i % 6}-0"))
                for e in (ents[i % 30], ents[(i * 7) % 30]):
                    cur.execute("INSERT INTO unit_entities VALUES (%s,%s) ON CONFLICT DO NOTHING", (uid, e))
            for i in range(29):
                cur.execute("INSERT INTO entity_cooccurrences VALUES (%s,%s,%s,now())",
                            (ents[i], ents[i + 1], 1 + i))
            made = 0
            k = 0
            while made < n_links:
                a, b = units[k % n_units], units[(k * 31 + 1 + k // n_units) % n_units]
                k += 1
                if a == b:
                    continue
                ent = ents[k % 30] if k % 3 == 0 else None
                cur.execute("INSERT INTO memory_links VALUES (%s,%s,%s,%s,%s,now(),%s) "
                            "ON CONFLICT DO NOTHING",
                            (a, b, ("semantic", "temporal", "entity")[k % 3], ent, 0.5, bank))
                made += cur.rowcount
            for j in range(3):
                cur.execute("INSERT INTO observation_history (observation_id, bank_id, content) "
                            "VALUES (%s,%s,%s)", (units[2], bank, json.dumps({"rev": j})))
            cur.execute("INSERT INTO mental_models (id, bank_id, name, source_query, content, embedding) "
                        "VALUES (%s,%s,'mm','q','c',%s::vector)", (f"mm-{bank}", bank, _vec("mm")))
            cur.execute("INSERT INTO mental_model_history (mental_model_id, bank_id, content) "
                        "VALUES (%s,%s,'{}')", (f"mm-{bank}", bank))
            cur.execute("INSERT INTO knowledge_pages (id, bank_id, kind, name) VALUES (%s,%s,'page','p')",
                        (f"kp-{bank}", bank))
            cur.execute("INSERT INTO invalidated_memory_units (id, bank_id, text, invalidation_reason) "
                        "VALUES (%s,%s,'old fact','superseded')", (_u(f"inv-{bank}"), bank))
            cur.execute("INSERT INTO directives (bank_id, name, content) VALUES (%s,'d','be brief')", (bank,))
            cur.execute("INSERT INTO audit_log (action, transport, bank_id) VALUES ('retain','http',%s)", (bank,))
            cur.execute("INSERT INTO llm_requests (bank_id, status) VALUES (%s,'ok')", (bank,))
            cur.execute("INSERT INTO async_operations (bank_id, operation_type) VALUES (%s,'consolidate')", (bank,))
            cur.execute("INSERT INTO bank_stats_cache VALUES (%s,'{}')", (bank,))
            cur.execute("INSERT INTO graph_maintenance_queue VALUES (%s,%s)", (bank, units[0]))
            cur.execute("INSERT INTO webhooks (bank_id, url, secret) VALUES (%s,'http://example.invalid','s3cret')", (bank,))
            exp[bank] = {"units": n_units, "links": n_links}
    return exp
