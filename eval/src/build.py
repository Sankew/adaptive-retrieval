"""Assemble eval/p3.jsonl from the templates in this folder. Run: uv run python eval/src/build.py"""
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from facts_a import FACTS as FACTS_A  # noqa: E402
from facts_b import FACTS as FACTS_B  # noqa: E402
from other import AGGREGATE, CONVERSATIONS, GENERAL, MULTI, RESTATES, UNANSWERABLE  # noqa: E402

ROOT = HERE.parents[1]
VERSION = {row["document_id"]: row["version"]
           for row in map(json.loads, (ROOT / "corpus/manifest.jsonl").read_text().splitlines())}


# A conversation and the templates it restates form one unit (union-find), so a fact can't be asked
# in a test conversation and trained on as a template question.
UNIT: dict[str, str] = {}


def unit(group: str) -> str:
    while UNIT.get(group, group) != group:
        group = UNIT[group]
    return group


for conversation, templates in RESTATES.items():
    for template in templates:
        a, b = sorted((unit(conversation), unit(template)))
        UNIT[b] = a


def item(question, answer, sources, type_, group, history=()):
    """`group` is the unit kept in one split (and resampled together): a template, or a conversation with
    the templates it restates."""
    group = unit(group)
    return {"question": question, "reference_answer": answer,
            "sources": [{"document_id": d, "version": VERSION[d]} for d in sources], "type": type_,
            "history": list(history), "group": group, "split": "train" if zlib.crc32(group.encode()) % 10 < 7 else "test",
            "origin": "synthetic"}


rows = []
for group, doc, answer, phrasings in FACTS_A + FACTS_B:
    rows += [item(q, answer, [doc], "single", group) for q in phrasings]
for group, docs, answer, phrasings in MULTI:
    rows += [item(q, answer, docs, "multi" if len(docs) > 1 else "single", group) for q in phrasings]
for i, (q, answer) in enumerate(GENERAL, 1):
    rows.append(item(q, answer, [], "general", f"gk-{i:02d}"))
for group, opening, opening_answer, doc, turns in CONVERSATIONS:
    rows.append(item(opening, opening_answer, [doc], "single", group))
    history = [{"role": "user", "content": opening}, {"role": "assistant", "content": opening_answer}]
    for q, answer, source in turns:
        rows.append(item(q, answer, [source], "followup", group, history))
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
for group, docs, answer, q in AGGREGATE:
    rows.append(item(q, answer, docs, "multi", group))
for i, q in enumerate(UNANSWERABLE, 1):
    rows.append(item(q, "Not stated in the company documents.", [], "unanswerable", f"na-{i:02d}"))

with (ROOT / "eval/p3.jsonl").open("w") as f:
    for n, row in enumerate(rows, 1):
        f.write(json.dumps({"id": f"p3-{n:03d}", **row}) + "\n")
print(len(rows), "questions")
