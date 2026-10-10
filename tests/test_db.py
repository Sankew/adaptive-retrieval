"""Against a real pgvector database. Skipped unless TEST_DATABASE_URL is set (CI starts one).

The tests rebuild the tables, so point TEST_DATABASE_URL at a database you don't serve from.
"""
import json
import os

import pytest

import router
from rag import store
from rag.models import LLM, FakeEmbedder, FakeReranker
from rag.pipeline import ensure_index, index
from tests.test_offline import FakeClient

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="needs a pgvector database")


@pytest.fixture(scope="module")
def conn():
    connection = store.connect(os.environ["TEST_DATABASE_URL"])
    index(connection, FakeEmbedder())
    yield connection
    connection.close()


def test_index_is_reused_and_search_returns_k_best_first(conn):
    assert ensure_index(conn, FakeEmbedder()) is False
    rows = store.search(conn, FakeEmbedder().embed(["sick days"], query=True)[0], 5)
    assert len(rows) == 5 and [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)

    class Other(FakeEmbedder):
        name = "other-model"
    with pytest.raises(SystemExit, match="built with fake-hashing"):
        ensure_index(conn, Other())


def test_routes_retrieve_nothing_three_or_ten(conn, tmp_path):
    llm = LLM(FakeClient(), tmp_path)
    q = {"question": "How many sick days?", "history": []}
    texts = {route: router.run_route(route, q, q["question"], conn, FakeEmbedder(), FakeReranker(), llm)["text"]
             for route in router.ROUTES}
    assert texts == {"none": "excerpts=0", "light": f"excerpts={router.LIGHT_K}", "deep": f"excerpts={router.DEEP_K}"}


@pytest.fixture
def labelling(conn, tmp_path, monkeypatch):
    """Run `label --fake` with a fake client; returns the label rows."""
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setattr(router, "RESULTS", tmp_path)

    def run(client: FakeClient) -> list[dict]:
        monkeypatch.setattr(router, "LLM", lambda: LLM(client, tmp_path / f"cache{id(client)}"))
        router.label(fake=True)
        return [json.loads(line) for line in (tmp_path / "labels_fake.jsonl").read_text().splitlines()]
    return run


def test_label_is_the_cheapest_route_the_judge_accepts(labelling, tmp_path):
    rows = labelling(FakeClient(verdict=lambda prompt: "excerpts=0" not in prompt))  # any retrieval passes
    assert len(rows) == 428 and {r["label"] for r in rows} == {"light"} and not any(r["all_failed"] for r in rows)
    followup = next(r for r in rows if r["type"] == "followup")
    assert followup["rewrite_usd"] > 0 and followup["query"] == "excerpts=0"
    assert json.loads((tmp_path / "env_fake.json").read_text())["fake_models"] is True
    assert not (tmp_path / "labels.jsonl").exists()  # fake labels never overwrite real ones

    rows = labelling(FakeClient(verdict=lambda prompt: False))
    assert {(r["label"], r["all_failed"]) for r in rows} == {("deep", True)}


def test_failed_answers_count_as_wrong_and_failed_rewrites_become_error_rows(labelling):
    rows = labelling(FakeClient(stop_reason="max_tokens"))
    errors = [r for r in rows if "error" in r]
    assert {r["type"] for r in errors} == {"followup"}  # the rewrite failed, so the question can't be routed
    answered = [r for r in rows if "error" not in r]
    assert all(r["all_failed"] and r["routes"]["deep"]["failed"] and r["routes"]["deep"]["usd"] > 0 for r in answered)
