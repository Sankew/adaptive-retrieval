"""Build the index. Run: uv run --group models python -m rag.pipeline index"""
from . import store
from .ingest import fingerprint, load_corpus


def index(conn, embedder) -> int:
    """Rebuild the tables from the manifest."""
    store.reset(conn)
    total = 0
    for row, chunks in load_corpus(embedder.count):
        store.add_document(conn, row, chunks, embedder.embed([c.content for c in chunks]))
        total += len(chunks)
    store.set_meta(conn, embedder.name, fingerprint())
    return total


def ensure_index(conn, embedder, reindex: bool = False) -> bool:
    """Build the index if it's missing or the corpus changed. True if rebuilt.

    An index built with a different embedder is replaced only with reindex=True, so a fake run
    can't silently swap out real embeddings.
    """
    built = store.meta(conn)
    if built == {"embedder": embedder.name, "corpus": fingerprint()}:
        return False
    if built and built["embedder"] != embedder.name and not reindex:
        raise SystemExit(f"this database's index was built with {built['embedder']}, not {embedder.name}. "
                         "Use another database (DATABASE_URL) or pass --reindex to replace it.")
    index(conn, embedder)
    return True


if __name__ == "__main__":
    import sys
    from .models import Embedder
    if sys.argv[1:] != ["index"]:
        raise SystemExit(__doc__)
    print(index(store.connect(), Embedder()), "chunks indexed")
