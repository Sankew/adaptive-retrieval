"""pgvector storage. Exact scan (no ANN index): at a few hundred chunks it's fast and has perfect recall."""
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .ingest import Chunk

DIM = 384  # bge-small-en-v1.5
DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/rag"

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks, documents, index_meta;
CREATE TABLE index_meta (
    embedder    text NOT NULL,                  -- query vectors must come from the same model
    corpus      text NOT NULL                   -- fingerprint of the files and chunking settings
);
CREATE TABLE documents (
    id          serial PRIMARY KEY,
    document_id text NOT NULL UNIQUE,
    title       text NOT NULL,
    version     int  NOT NULL,
    path        text NOT NULL
);
CREATE TABLE chunks (
    id          serial PRIMARY KEY,
    document    int  NOT NULL REFERENCES documents (id),
    section     text NOT NULL,
    ord         int  NOT NULL,
    content     text NOT NULL,
    embedding   vector({DIM}) NOT NULL
);
"""


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or os.environ.get("DATABASE_URL", DEFAULT_URL), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def reset(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def set_meta(conn: psycopg.Connection, embedder: str, corpus: str) -> None:
    conn.execute("INSERT INTO index_meta VALUES (%s, %s)", (embedder, corpus))


def meta(conn: psycopg.Connection) -> dict | None:
    """What the index was built with, or None if there is no index."""
    if conn.execute("SELECT to_regclass('index_meta')").fetchone()[0] is None:
        return None
    rows = _rows(conn.execute("SELECT embedder, corpus FROM index_meta"))
    return rows[0] if rows else None


def add_document(conn: psycopg.Connection, row: dict, chunks: list[Chunk], vectors: np.ndarray) -> None:
    with conn.transaction():
        doc = conn.execute("INSERT INTO documents (document_id, title, version, path) VALUES (%s, %s, %s, %s) RETURNING id",
                           (row["document_id"], row["title"], row["version"], row["path"])).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO chunks (document, section, ord, content, embedding) VALUES (%s, %s, %s, %s, %s)",
                            [(doc, c.section, c.ord, c.content, v) for c, v in zip(chunks, vectors, strict=True)])


def search(conn: psycopg.Connection, vector: np.ndarray, k: int) -> list[dict]:
    """Top-k chunks by cosine similarity, ties broken by insertion order."""
    return _rows(conn.execute(
        """SELECT d.document_id, d.version, d.title, c.section, c.content, 1 - (c.embedding <=> %(v)s) AS score
           FROM chunks c JOIN documents d ON d.id = c.document
           ORDER BY c.embedding <=> %(v)s, c.id LIMIT %(k)s""", {"v": vector, "k": k}))


def _rows(cursor: psycopg.Cursor) -> list[dict]:
    names = [d.name for d in cursor.description]
    return [dict(zip(names, r)) for r in cursor]
