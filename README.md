# Adaptive retrieval

[![Checks](https://github.com/Sankew/adaptive-retrieval/actions/workflows/checks.yml/badge.svg)](https://github.com/Sankew/adaptive-retrieval/actions/workflows/checks.yml)

Not every question needs retrieval. "What is the capital of Japan?" doesn't, "What's the annual sick-day allowance?" needs one passage, and "During the holiday deploy freeze, which named person signs off a SEV1 fix?" needs two documents. This project answers each question on one of three routes and trains a small classifier to pick the cheapest route that still gets it right.

| Route | Retrieval | Cost |
|---|---|---|
| `none` | nothing; the model answers from general knowledge and abstains on company questions | one short call |
| `light` | top 3 chunks by cosine similarity | one call, small prompt |
| `deep` | top 30, reranked by a cross-encoder, top 10 kept | one call, large prompt, plus local reranking |

It runs on a fictional company, Brackwater Robotics, so the model can't answer company questions from memory.

```mermaid
flowchart LR
    Q[Question<br/>+ conversation] --> W{Follow-up?}
    W -- yes --> R[Rewrite to a<br/>standalone question<br/>Configured generation model]
    W -- no --> C
    R --> C[Router<br/>TF-IDF + logistic regression]
    C -- confident: none --> N[Answer without<br/>documents]
    C -- confident: light --> L[Top 3<br/>pgvector]
    C -- unsure, or deep --> D[Top 30 → MiniLM<br/>cross-encoder → top 10]
    L --> A[Configured generation model<br/>cites document:section]
    D --> A
```

## How it's labelled and trained

- **Labels.** Every question is answered on all three routes, and Configured judging model judges each answer against the reference. The label is the cheapest route judged correct. When no route is, the label is `deep` and the question is counted in `all_routes_failed`.
- **Failures.** A refused or truncated answer is a wrong answer on that route, and its tokens still count toward cost.
- **Spot-check.** `router.py spotcheck` samples 40 labelled questions with the conversation, the rewritten query and every route's answer and verdict, for checking by hand. `--sheet` reports the share of labels confirmed. The checked sheet gets committed as evidence, and `spotcheck` won't overwrite one that has marks in it.
- **Router.** TF-IDF over word unigrams and bigrams, then logistic regression with balanced class weights, trained on the rewritten standalone question. If the top class probability is under 0.5, the question goes to `deep`: a wrong `deep` costs money, a wrong `none` costs the answer. The 0.5 was fixed before any test results; the threshold curve in the report is descriptive.
- **Follow-ups.** A follow-up ("And do they carry over?") is rewritten into a standalone question before routing and retrieval. The rewrite call counts toward cost. The answer and the judge both see the conversation.

## Evaluation

**Data:** `eval/p3.jsonl`, 428 questions, generated from the templates in `eval/src/`:

| Type | Count | Example |
|---|---|---|
| single-source | 269 | "What's the annual sick-day allowance?" |
| general knowledge | 60 | "What is the boiling point of water at sea level in Celsius?" |
| multi-source | 48 | "Who approves a 20% discount, by name?"; 2 of them need every document's version stamp and are built to fail on all routes |
| follow-up | 31 | in 25 conversations, with history |
| unanswerable | 20 | company questions the documents don't cover |

The corpus is 47 current Brackwater documents in Markdown, HTML and PDF, including stale FAQ pages that disagree with current policy.

**Splits** are by group, never by question: 125 groups and 318 questions train, 42 groups and 110 questions test. A group is a fact template with its paraphrases, or a whole conversation together with the templates its turns restate (listed in `eval/src/other.py`). Otherwise a fact asked in a test conversation could also be a training question, and the router would look better than it is. `eval.validate` fails if a group, a question's text, or a fact (same sources, same reference answer) appears in both splits.

**Systems compared on the test split:**

- never retrieve, always light, always deep;
- the router;
- the oracle: each question's label, and the free `none` route where every route fails, so no router can beat it on accuracy or cost.

The same comparison is repeated on company questions alone. The retrieval routes are told to answer from documents only, so they fail general-knowledge questions by design, and "router beats always-deep" would otherwise mostly mean "the router spots general-knowledge questions".

**Metrics:**

- **Accuracy** (judge), with k/n and a 95% interval from resampling whole groups, since paraphrases are correlated. Router minus each baseline is paired on the same questions, with the same kind of interval.
- **Cost:** API dollars per 1,000 questions, from the token counts the API returns, including the rewrite and excluding the offline judge.
- **Local retrieval time** (embedding, search, reranking) is reported in milliseconds, separately from model latency. `results/env.json` records the machine and whether models ran on MPS or CPU, since both change these numbers.
- **Router accuracy** is reported overall, for light vs deep alone, and by question type. Questions every route fails are left out (their `deep` label is arbitrary) and counted. General-knowledge vs company questions is easy to separate, so the overall number flatters the router; light vs deep is the hard part.
- `results/misroutes.csv` lists every test question the router sent somewhere other than its label, for error analysis.

## Results

**Pending.** Labelling needs `bge-small-en-v1.5`, the MiniLM reranker and an Anthropic API key, and hasn't been run yet. CI checks the data, the routing and scoring logic on synthetic labels, and a full fake labelling pass against Postgres; none of that is a result.

## Run

Python 3.12, [uv](https://docs.astral.sh/uv/), Docker:

```bash
docker compose up -d db
uv sync --group models --group dev
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag ANTHROPIC_API_KEY=...
export GENERATOR_MODEL="<generation-model-id>" JUDGE_MODEL="<judging-model-id>"
export GENERATOR_INPUT_USD_PER_MILLION="<input-rate>" GENERATOR_OUTPUT_USD_PER_MILLION="<output-rate>"
uv run python router.py label          # about 428 × (3 answers + 3 judgements) calls, cached in .cache/llm/
uv run python router.py evaluate       # writes results/routing.json and results/misroutes.csv
uv run python router.py spotcheck      # then fill label_ok and run with --sheet results/spotcheck.csv
```

Tests rebuild their tables, so they use a separate database:

```bash
docker compose exec db createdb -U postgres rag_test
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_test uv run pytest -q
```

Only complete responses are cached. When a rewrite or judge call fails on a transient error (timeout, rate limit, overload), the question is written as an error row, left out of evaluation (the count is in the report) and retried on the next run. A bad key or model name stops the run. Labels are written to a temporary file and moved into place at the end, so a crash leaves the previous labels intact, and `evaluate` refuses a labels file with questions missing. `label --fake` (hashing embedder, word-overlap reranker, real model calls) writes `labels_fake.jsonl`, never `labels.jsonl`. The questions are generated: edit `eval/src/`, then `uv run python eval/src/build.py` and `uv run python -m eval.validate`.

## Limitations

- The corpus, questions and reference answers are synthetic fixtures; answers are graded automatically. This can reinforce shared errors. Labels are only as good as the judge, which is why they are spot-checked by hand.
- An unanswerable company question is labelled `none` when the no-retrieval route correctly abstains. The router can learn that from topic words, which says nothing about whether the documents cover a question.
- The overlap between conversations and templates is listed by hand. The validator catches exact repeats, not every paraphrase.
- Conversation history holds the reference answers to earlier turns, so rewrites and answers always build on a correct conversation. Real conversations aren't that clean.
- TF-IDF sees words, not intent. It can't tell a single-hop question from a multi-hop one phrased with the same words; the misroutes file shows where that matters.
- Model latency comes from the first real call and is replayed from the cache, so it reflects network conditions at labelling time.
- Generation prices come from `GENERATOR_INPUT_USD_PER_MILLION` and `GENERATOR_OUTPUT_USD_PER_MILLION`. Defaults are illustrative (0.10 and 0.50); set the actual rates for the selected model before reporting costs.
- One small corpus. The labels, and so the router, are specific to it.
