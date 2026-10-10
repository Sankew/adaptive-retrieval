"""Adaptive retrieval: answer with no retrieval, light retrieval or deep retrieval, and learn which.

    uv run --group models python router.py label        # run every route on every question, judge, label
    uv run python router.py evaluate                    # train on train, compare systems on test
    uv run python router.py spotcheck                   # sample labels to check by hand
    uv run python router.py spotcheck --sheet results/spotcheck.csv   # score the checked sheet
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rag import store
from rag.answer import answer, judge, rewrite
from rag.ingest import ROOT
from rag.models import LLM, LLMError, transient

ROUTES = ("none", "light", "deep")  # cheapest first
LIGHT_K, DEEP_CANDIDATES, DEEP_K = 3, 30, 10
PRICE_IN = float(os.environ.get("GENERATOR_INPUT_USD_PER_MILLION", "0.10"))
PRICE_OUT = float(os.environ.get("GENERATOR_OUTPUT_USD_PER_MILLION", "0.50"))
RESULTS = ROOT / "results"


def labels_path(fake: bool) -> Path:
    """Labels from fake retrieval are kept apart: the model calls are real, but they aren't results."""
    return RESULTS / ("labels_fake.jsonl" if fake else "labels.jsonl")


def questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "p3.jsonl").read_text(encoding="utf-8").splitlines()]


def usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1e6


def standalone(llm: LLM, q: dict) -> tuple[str, dict | None]:
    """Follow-ups are rewritten into a standalone question before routing or retrieval."""
    if not q["history"]:
        return q["question"], None
    call = rewrite(llm, q["question"], q["history"])
    return call["text"].strip(), call


def run_route(route: str, q: dict, query: str, conn, embedder, reranker, llm: LLM) -> dict:
    """Answer one question on one route. Retrieval (embedding, search, rerank) is timed here as local_ms;
    llm_s is the model's latency, recorded on the first real call and kept in the cache. A refused or
    truncated answer is a failed route: it still costs, and it's marked `failed`."""
    start = time.perf_counter()
    if route == "none":
        chunks = None
    elif route == "light":
        chunks = store.search(conn, embedder.embed([query], query=True)[0], LIGHT_K)
    else:
        candidates = store.search(conn, embedder.embed([query], query=True)[0], DEEP_CANDIDATES)
        chunks = reranker.rerank(query, candidates)[:DEEP_K]
    out = {"local_ms": (time.perf_counter() - start) * 1000, "chunks": [f"{c['document_id']}:{c['section']}" for c in chunks or []]}
    try:
        result = answer(llm, q["question"], chunks, q["history"])
    except LLMError as e:
        return out | {"text": "", "usd": usd(e.input_tokens, e.output_tokens), "llm_s": None, "failed": str(e)}
    return out | {"text": result["text"], "usd": usd(result["input_tokens"], result["output_tokens"]),
                  "llm_s": result.get("latency_s")}


def environment(fake: bool) -> dict:
    """Where latency was measured: local_ms depends on the machine and on MPS vs CPU."""
    env = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
           "fake_models": fake}
    try:
        import torch
        env["torch"] = torch.__version__
        env["device"] = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        env["device"] = "none (fake models)"
    return env


def label(fake: bool, reindex: bool = False) -> None:
    """Cheapest route the judge marks correct. If none is, the label is deep and `all_failed` is set.

    A question whose rewrite or judging fails on a transient API error (or an unusable judge response) is
    written as an error row, left out of evaluation and retried next run. A bad key or model stops the run.
    """
    import anthropic

    from rag.models import Embedder, FakeEmbedder, FakeReranker, Reranker
    from rag.pipeline import ensure_index
    embedder, reranker = (FakeEmbedder(), FakeReranker()) if fake else (Embedder(), Reranker())
    conn = store.connect()
    ensure_index(conn, embedder, reindex)
    llm = LLM()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ("env_fake.json" if fake else "env.json")).write_text(json.dumps(environment(fake), indent=2) + "\n")
    path = labels_path(fake)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as out:
        for q in questions():
            row = {"id": q["id"], "type": q["type"], "split": q["split"], "group": q["group"]}
            try:
                query, rewrite_call = standalone(llm, q)
                row |= {"query": query, "routes": {},
                        "rewrite_usd": usd(rewrite_call["input_tokens"], rewrite_call["output_tokens"]) if rewrite_call else 0.0,
                        "rewrite_s": rewrite_call.get("latency_s") if rewrite_call else 0.0}
                for route in ROUTES:
                    result = run_route(route, q, query, conn, embedder, reranker, llm)
                    if "failed" in result:
                        verdict = {"correct": False, "reason": "no usable answer"}
                    else:
                        verdict = judge(llm, q["question"], q["reference_answer"], result["text"], q["history"])
                    row["routes"][route] = {**result, "correct": verdict["correct"], "reason": verdict["reason"]}
            except (LLMError, anthropic.APIError) as e:
                if not (isinstance(e, LLMError) or transient(e)):
                    raise
                out.write(json.dumps({**row, "error": f"{type(e).__name__}: {e}"}) + "\n")
                print(q["id"], "error:", e, flush=True)
                continue
            correct = [r for r in ROUTES if row["routes"][r]["correct"]]
            row["label"], row["all_failed"] = (correct[0], False) if correct else ("deep", True)
            out.write(json.dumps(row) + "\n")
            print(q["id"], row["label"], "(all failed)" if row["all_failed"] else "", flush=True)
    os.replace(partial, path)  # a crashed run leaves the previous labels in place


class Router:
    """TF-IDF + logistic regression over the standalone question. Below `threshold`, fall back to deep."""

    def __init__(self, threshold: float) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                                   LogisticRegression(max_iter=2000, class_weight="balanced"))
        self.threshold = threshold

    def fit(self, texts: list[str], labels: list[str]) -> "Router":
        self.model.fit(texts, labels)
        return self

    def route(self, texts: list[str]) -> list[str]:
        probs = self.model.predict_proba(texts)
        classes = self.model.classes_
        return [classes[p.argmax()] if p.max() >= self.threshold else "deep" for p in probs]


def group_bootstrap(rows: list[dict], values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, list[float]]:
    """Mean and a 95% interval from resampling whole groups: a template's paraphrases, or a conversation
    with the templates it restates, are correlated and move together."""
    by_group: dict[str, list[float]] = {}
    for r, v in zip(rows, values, strict=True):
        by_group.setdefault(r["group"], []).append(v)
    sums = np.array([sum(v) for v in by_group.values()])
    counts = np.array([len(v) for v in by_group.values()])
    picks = np.random.default_rng(seed).integers(0, len(sums), (n, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (round(float(sums.sum() / counts.sum()), 3),
            [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)])


def score(rows: list[dict], choice: list[str], charge_rewrite: bool) -> dict:
    """Accuracy and cost when each question is answered on the chosen route. The follow-up rewrite is
    charged when the route retrieves, and always for the router, which routes on the rewritten question."""
    correct = [float(r["routes"][c]["correct"]) for r, c in zip(rows, choice, strict=True)]
    charged = [charge_rewrite or c != "none" for c in choice]
    cost = sum(r["routes"][c]["usd"] + (r["rewrite_usd"] if paid else 0.0) for r, c, paid in zip(rows, choice, charged))
    llm = [r["routes"][c]["llm_s"] + ((r.get("rewrite_s") or 0.0) if paid else 0.0)
           for r, c, paid in zip(rows, choice, charged) if r["routes"][c]["llm_s"]]
    accuracy, ci = group_bootstrap(rows, correct)
    return {"accuracy": accuracy, "ci95": ci, "correct": int(sum(correct)), "n": len(correct),
            "usd_per_1k": round(1000 * cost / len(rows), 4), "routes": dict(Counter(choice)),
            "local_ms_p50": round(statistics.median(r["routes"][c]["local_ms"] for r, c in zip(rows, choice)), 1),
            "llm_s_p50": round(statistics.median(llm), 2) if llm else None}


def paired_delta(rows: list[dict], a: list[str], b: list[str]) -> dict:
    """Accuracy of choice b minus choice a on the same questions, with a group-bootstrap interval."""
    diffs = [float(r["routes"][y]["correct"]) - float(r["routes"][x]["correct"]) for r, x, y in zip(rows, a, b, strict=True)]
    delta, ci = group_bootstrap(rows, diffs)
    return {"delta": delta, "ci95": ci}


def compare(rows: list[dict], routed: list[str]) -> dict:
    """Baselines, router and oracle on the same questions. The oracle takes each question's label, and the
    least expensive measured route when every route fails. It maximizes accuracy first, then minimizes cost."""
    n = len(rows)
    baselines = {"never retrieve": ["none"] * n, "always light": ["light"] * n, "always deep": ["deep"] * n}
    # Select by measured generation cost, including retrieval's rewrite cost.
    # Route depth alone does not order token costs; even a failed call can be cheaper on deep.
    oracle = [min((route for route in ROUTES if r["routes"][route]["correct"]),
                  key=lambda route: r["routes"][route]["usd"] + (r["rewrite_usd"] if route != "none" else 0.0),
                  default=None) for r in rows]
    oracle = [route if route is not None else min(ROUTES,
              key=lambda choice: r["routes"][choice]["usd"] + (r["rewrite_usd"] if choice != "none" else 0.0))
              for r, route in zip(rows, oracle)]
    return {"systems": {name: score(rows, choice, charge_rewrite=False) for name, choice in baselines.items()}
            | {"router": score(rows, routed, charge_rewrite=True), "oracle": score(rows, oracle, charge_rewrite=False)},
            "router_minus_baseline": {name: paired_delta(rows, choice, routed) for name, choice in baselines.items()}}


def router_accuracy(rows: list[dict], routed: list[str]) -> dict:
    """How often the router picks the label. Questions every route fails are left out: their `deep` label
    is arbitrary, and the low-confidence fallback is also `deep`."""
    scored = [(p, r) for p, r in zip(routed, rows, strict=True) if not r["all_failed"]]
    light_deep = [(p, r) for p, r in scored if r["label"] in ("light", "deep")]
    by_type: dict[str, list[bool]] = {}
    for p, r in scored:
        by_type.setdefault(r["type"], []).append(p == r["label"])
    return {"overall": round(statistics.fmean(p == r["label"] for p, r in scored), 3) if scored else None,
            "light_vs_deep": round(statistics.fmean(p == r["label"] for p, r in light_deep), 3) if light_deep else None,
            "by_type": {t: round(statistics.fmean(v), 3) for t, v in sorted(by_type.items())},
            "questions": len(scored), "all_failed_left_out": len(rows) - len(scored)}


def evaluate(threshold: float, fake: bool = False) -> dict:
    path = labels_path(fake)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit(f"{path.name} contains duplicate question IDs; rerun label")
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if missing := {q["id"] for q in questions()} - {r["id"] for r in rows + errors}:
        raise SystemExit(f"{len(missing)} questions have no row in {path.name}, e.g. {sorted(missing)[:3]}; rerun label")
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise SystemExit(f"{path.name} needs labelled questions in both splits")
    queries = [r["query"] for r in test]
    router = Router(threshold).fit([r["query"] for r in train], [r["label"] for r in train])
    routed = router.route(queries)
    company = [i for i, r in enumerate(test) if r["type"] != "general"]
    curve = {}  # descriptive only: `threshold` was fixed before looking at test results
    for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        router.threshold = t
        result = score(test, router.route(queries), charge_rewrite=True)
        curve[t] = {"accuracy": result["accuracy"], "usd_per_1k": result["usd_per_1k"]}
    report = {
        "test_questions": len(test),
        "error_rows_skipped": len(errors),
        "label_counts_train": dict(Counter(r["label"] for r in train)),
        "label_counts_test": dict(Counter(r["label"] for r in test)),
        "all_routes_failed": {"train": sum(r["all_failed"] for r in train), "test": sum(r["all_failed"] for r in test)},
        **compare(test, routed),
        # The retrieval routes are told to answer from documents only, so they fail general-knowledge questions
        # by design. Without them, the comparison is about retrieval depth alone.
        "company_questions_only": compare([test[i] for i in company], [routed[i] for i in company]),
        "router_accuracy": router_accuracy(test, routed),
        "threshold": threshold,
        "threshold_curve": curve,
    }
    env = RESULTS / ("env_fake.json" if fake else "env.json")
    if env.exists():
        report["environment"] = json.loads(env.read_text())
    suffix = "_fake" if fake else ""
    (RESULTS / f"routing{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (RESULTS / f"misroutes{suffix}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "query", "label", "routed", "routed_correct", "label_correct"])
        for r, p in zip(test, routed):
            if p != r["label"]:
                writer.writerow([r["id"], r["type"], r["query"], r["label"], p,
                                 r["routes"][p]["correct"], r["routes"][r["label"]]["correct"]])
    return report


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for k successes in n independent samples."""
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 3), round(centre + half, 3)]


def spotcheck(n: int, seed: int) -> Path:
    """A random sample of labelled questions with every route's answer and verdict, to confirm labels by hand.
    Commit the checked sheet: it's the evidence the labels are right."""
    out = RESULTS / "spotcheck.csv"
    if out.exists() and any(r["label_ok"].strip() for r in csv.DictReader(out.open())):
        raise SystemExit(f"{out} already has hand marks; move it aside to sample a new sheet")
    rows = [r for r in map(json.loads, labels_path(False).read_text().splitlines()) if "error" not in r]
    by_id = {q["id"]: q for q in questions()}
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "history", "question", "query", "reference_answer", "label", "all_failed",
                         *(f"{r}_{field}" for r in ROUTES for field in ("answer", "judge")), "label_ok"])
        for r in random.Random(seed).sample(rows, min(n, len(rows))):
            q = by_id[r["id"]]
            history = "\n".join(f"{t['role']}: {t['content']}" for t in q["history"])
            writer.writerow([r["id"], history, q["question"], r["query"], q["reference_answer"], r["label"], r["all_failed"],
                             *(r["routes"][route][field] for route in ROUTES for field in ("text", "correct")), ""])
    print(f"wrote {out}: set label_ok to 1 or 0, then run `python router.py spotcheck --sheet {out}`")
    return out


def spotcheck_score(sheet: Path) -> dict:
    marks = [int(r["label_ok"]) for r in csv.DictReader(sheet.open()) if r["label_ok"].strip() in ("0", "1")]
    report = {"checked": len(marks), "labels_confirmed": sum(marks),
              "ci95": wilson(sum(marks), len(marks)) if marks else None}
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--fake", action="store_true",
                     help="hashing embedder and word-overlap reranker (still calls the API); writes labels_fake.jsonl")
    lab.add_argument("--reindex", action="store_true", help="replace an index built with a different embedder")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--threshold", type=float, default=0.5)
    ev.add_argument("--fake", action="store_true", help="evaluate labels_fake.jsonl")
    spot = sub.add_parser("spotcheck")
    spot.add_argument("--n", type=int, default=40)
    spot.add_argument("--seed", type=int, default=0)
    spot.add_argument("--sheet", type=Path, help="score a checked sheet instead of writing a new one")
    args = parser.parse_args()
    if args.command == "label":
        label(args.fake, args.reindex)
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.threshold, args.fake), indent=2))
    elif args.sheet:
        spotcheck_score(args.sheet)
    else:
        spotcheck(args.n, args.seed)


if __name__ == "__main__":
    main()
