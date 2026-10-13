"""Check the corpus manifest and eval/p3.jsonl against the data contracts. Exit 1 on any error.

    uv run python -m eval.validate
"""
import json
import re

from rag.ingest import ROOT, manifest, parse

TYPES = {"single", "multi", "general", "followup", "unanswerable"}
NO_SOURCES = {"general", "unanswerable"}
FIELDS = {"id": str, "question": str, "reference_answer": str, "sources": list, "type": str, "history": list,
          "group": str, "split": str, "origin": str}


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def check() -> list[str]:
    errors: list[str] = []
    rows = manifest()
    for r in rows:
        if not (ROOT / r["path"]).exists():
            errors.append(f"manifest: missing file {r['path']}")
        if r["status"] != "current" or r["effective_to"] is not None:
            errors.append(f"manifest: {r['document_id']} is not a current version (P3 indexes current documents only)")
    if len({r["document_id"] for r in rows}) != len(rows):
        errors.append("manifest: duplicate document_id")
    texts = {(r["document_id"], r["version"]): "\n".join(s.heading + "\n" + s.text for s in parse(ROOT / r["path"], r["title"]))
             for r in rows if (ROOT / r["path"]).exists()}

    split_of: dict[object, set[str]] = {}  # group, question text or fact -> splits it appears in
    seen: set[str] = set()
    path = ROOT / "eval" / "p3.jsonl"
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        where = f"p3.jsonl:{n}"
        q = json.loads(line)
        for field, kind in FIELDS.items():
            if not isinstance(q.get(field), kind):
                errors.append(f"{where}: {field} missing or not {kind.__name__}")
        if q.get("id") in seen:
            errors.append(f"{where}: duplicate id {q.get('id')}")
        seen.add(q.get("id"))
        if q.get("type") not in TYPES or q.get("split") not in ("train", "test") or q.get("origin") not in ("synthetic", "human"):
            errors.append(f"{where}: bad type, split or origin")
        split_of.setdefault(q.get("group"), set()).add(q.get("split"))
        # The same question, or the same fact from the same documents, in both splits leaks too.
        split_of.setdefault(("question", norm(q.get("question", ""))), set()).add(q.get("split"))
        if q.get("sources"):
            fact = ("fact", tuple(sorted(s["document_id"] for s in q["sources"])), norm(q.get("reference_answer", "")))
            split_of.setdefault(fact, set()).add(q.get("split"))
        history = q.get("history", [])
        if (q.get("type") == "followup") != bool(history):
            errors.append(f"{where}: follow-ups need history and only follow-ups have it")
        if [t.get("role") for t in history] != ["user", "assistant"] * (len(history) // 2) or len(history) % 2:
            errors.append(f"{where}: history must alternate user, assistant")
        sources = [(s["document_id"], s["version"]) for s in q.get("sources", [])]
        for s in sources:
            if s not in texts:
                errors.append(f"{where}: unknown source {s}")
        if q.get("type") in NO_SOURCES:
            if sources:
                errors.append(f"{where}: {q['type']} question has sources")
            continue
        if not sources or (q.get("type") == "multi" and len(set(sources)) < 2):
            errors.append(f"{where}: too few sources for a {q.get('type')} question")
        source_numbers = set().union(*(numbers(texts[s]) for s in sources if s in texts))
        if missing := numbers(q.get("reference_answer", "")) - source_numbers:
            errors.append(f"{where}: reference numbers {sorted(missing)} not in its sources")
    for key, splits in split_of.items():
        if len(splits) > 1:
            errors.append(f"{key} is in both splits: it would leak from train to test")
    return errors


if __name__ == "__main__":
    problems = check()
    print("\n".join(problems) or "ok")
    raise SystemExit(1 if problems else 0)
