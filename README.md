# Adaptive retrieval

[![Checks](https://github.com/Sankew/adaptive-retrieval/actions/workflows/checks.yml/badge.svg?branch=p3-routing)](https://github.com/Sankew/adaptive-retrieval/actions?query=branch%3Ap3-routing)

Some questions can be answered without documents. Others need one passage or evidence from several sources. This project compares three retrieval routes and trains a small classifier to choose the shallowest route judged correct for each question.

The experiment uses documents from a fictional company, Brackwater Robotics. This reduces the chance that company facts already appear in a model's training data, while leaving the model free to answer general-knowledge questions without retrieval.

**Status:** data validation and automated tests pass CI, including database tests with deterministic substitutes. Labelling and evaluation with real models are pending. There are no measured accuracy or cost improvements to report yet.

## Retrieval routes

| Route | Context supplied to the generator | Work involved |
|---|---|---|
| `none` | No documents; general knowledge is allowed, with instructions to abstain on company questions | One generation call |
| `light` | Top 3 chunks from exact cosine search in pgvector | Embedding, search and one generation call |
| `deep` | Top 30 chunks, reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, with the top 10 retained | Embedding, search, local reranking and one generation call |

Follow-ups are rewritten into standalone questions before routing. That adds a generation call. Retrieval depth changes the context size and local work; it does not guarantee a particular token cost.

```mermaid
flowchart LR
    Q[Question and conversation] --> W[Rewrite follow-ups into standalone questions]
    W --> R[TF-IDF and logistic regression]
    R --> N[No documents]
    R --> L[Top 3 chunks]
    R --> D[Top 30 chunks, reranked to 10]
    N --> G[Generate an answer]
    L --> G
    D --> G
```

## Setup

Use Python 3.12, uv and Docker. Run commands from the repository root.

```bash
uv sync --locked --group models --group dev
docker compose up -d --wait db
docker compose exec db createdb -U postgres adaptive_rag
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/adaptive_rag"
```

Create `adaptive_rag` once. The Compose service binds to local port 5432. If another project already runs pgvector on that port, reuse that server and create this project's separate database there. Indexing rebuilds the project's tables.

Generation and judging use the Anthropic SDK. Replace the placeholders with an API key and actual model IDs that support the request's effort settings and structured JSON output:

```bash
export ANTHROPIC_API_KEY="your-api-key"
export GENERATOR_MODEL="your-generation-model-id"
export JUDGE_MODEL="your-judging-model-id"
```

Local embeddings use `BAAI/bge-small-en-v1.5`; embedding and reranking models are downloaded on first use. Before reporting cost, set `GENERATOR_INPUT_USD_PER_MILLION` and `GENERATOR_OUTPUT_USD_PER_MILLION` to numeric rates for the chosen generation model. The defaults, 0.10 and 0.50, are illustrative.

## Label, train and evaluate

```bash
uv run --locked --group models python router.py label
uv run --locked python router.py evaluate
uv run --locked python router.py spotcheck
# Review results/spotcheck.csv and fill label_ok with 1 or 0, then:
uv run --locked python router.py spotcheck --sheet results/spotcheck.csv
```

1. **Label:** answer each question on all three routes and judge each answer against its reference. The training label is the first correct route in the order `none`, `light`, `deep`. This selects retrieval depth rather than the lowest measured cost. When all routes fail, the label is `deep` and `all_failed` is recorded.
2. **Train:** fit TF-IDF word unigrams and bigrams followed by logistic regression with balanced class weights. Training uses the standalone question. A top-class probability below 0.5 sends the question to `deep`.
3. **Evaluate:** compare the fitted router with fixed routes on the held-out test groups. The threshold curve describes alternative settings; the default threshold is fixed before test results.
4. **Review:** sample up to 40 questions with their history, rewritten query, route answers and verdicts. The completed sheet reports the share of labels confirmed. Sampling refuses to overwrite a sheet with existing review marks.

Labelling makes generation and judging calls for each route, plus rewrites for follow-ups. It can incur substantial API usage. Complete responses are cached by the full request in `.cache/llm/`; unchanged requests reuse them, while new or failed requests can incur additional charges.

### Data and splits

The corpus has **47 current documents** in Markdown, HTML and PDF. Some FAQ pages retain outdated statements that conflict with current policy.

`eval/p3.jsonl` contains **428 questions**:

| Type | Count | Purpose |
|---|---|---|
| Single-source | 269 | Find a fact in one document |
| General knowledge | 60 | Test when retrieval is unnecessary |
| Multi-source | 48 | Combine evidence across documents, including two cases designed to exceed the available context |
| Follow-up | 31 | Resolve context across 25 conversations |
| Unanswerable | 20 | Abstain when company information is undocumented |

The training split contains **318 questions in 125 groups**; the test split contains **110 questions in 42 groups**. Paraphrases of a fact stay in one group. Conversations are grouped with the fact templates they repeat, using the mapping in `eval/src/other.py`.

The validator rejects groups, identical question text and identical source/reference-answer combinations shared across splits. This catches exact overlap; it cannot rule out every semantic paraphrase.

Questions are generated from templates. Edit `eval/src/`, rebuild, then validate:

```bash
uv run --locked python eval/src/build.py
uv run --locked python -m eval.validate
```

### Comparisons and metrics

The report compares **never retrieve**, **always light**, **always deep**, the **router**, and an **oracle**. For each question, the oracle selects the cheapest measured correct route, including rewrite cost for retrieval routes. If every route fails, it selects the cheapest route. Its decisions use verdicts that a deployed router would not have.

The comparison is also reported for company questions alone. Retrieval routes are instructed to use documents only, so general-knowledge questions can otherwise make route selection look better without showing an improvement in retrieval depth.

- **Answer accuracy:** judged correctness, counts and 95% bootstrap intervals obtained by resampling whole groups. Router-versus-baseline differences are paired on the same questions.
- **Estimated cost per 1,000 questions:** generation token counts multiplied by configured rates, including applicable rewrites and excluding offline judging. The router pays for follow-up rewrites even when it selects `none`; fixed routes and the oracle pay for them when retrieving. Cached token counts remain in the estimate, so this is not the bill for an evaluation rerun.
- **Timing:** local embedding, search and reranking time are separate from model latency. Timings come from the labelling pass; cached model latency reflects the original call. The machine and model device are recorded in `results/env.json`.
- **Route-label accuracy:** overall, by question type, and for light versus deep. Questions where all routes fail are excluded from this metric and counted separately because their `deep` label is arbitrary.
- **Misroutes:** questions where the chosen route differs from the training label. A different route can still answer correctly; the CSV includes answer correctness to distinguish those cases.

### Output and failures

Labelling writes `results/labels.jsonl` and `results/env.json`. Evaluation writes `results/routing.json` and `results/misroutes.csv`. Review uses `results/spotcheck.csv`.

Refused or truncated answers count as wrong, with their generation tokens included in cost. Transient API failures and unusable rewrite or judge responses produce error rows, excluded from evaluation and retried on the next labelling run. Invalid credentials or model IDs stop the run. Label files are replaced atomically when a pass finishes; evaluation rejects duplicate IDs and missing question rows.

`label --fake` replaces embeddings and reranking with deterministic substitutes, but **still makes real generation and judging calls**. It writes separate `labels_fake.jsonl` and `env_fake.json` files. Use `evaluate --fake` for those labels. This option is not an API-free smoke test.

## Tests without model calls

The test suite uses mocked model clients and deterministic embeddings. Database tests require a dedicated pgvector database and rebuild its tables:

```bash
uv sync --locked --group dev
docker compose exec db createdb -U postgres adaptive_rag_test
TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/adaptive_rag_test" \
  uv run --locked --group dev pytest -q
```

Create the test database once. CI also checks that rebuilding the templates leaves `eval/p3.jsonl` unchanged.

## Limitations

- Synthetic documents, questions and references can share wording and mistakes. Automated verdicts need manual review before supporting performance claims.
- The router learns lexical patterns from one corpus. It does not directly reason about how many documents a question needs.
- Correct abstention can give an unanswerable company question a `none` label. Recognising its wording does not prove the router knows what the corpus covers.
- Conversation history contains reference answers to earlier turns. The experiment does not measure how routing behaves after an earlier answer is wrong.
- The conversation/template overlap map is maintained manually, and exact-match validation cannot detect all related facts.
- Local timing depends on hardware and model device. Cached model timing records earlier network conditions rather than fresh end-to-end request latency.
