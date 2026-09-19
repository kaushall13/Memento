# Memento

Long-term memory for conversational agents. Conversations are distilled into
typed, versioned memories rather than stored as raw turns, and those memories
are retrieved with calibrated ranking so a question can be answered from a
history far longer than the context window.

The reason the layer exists at all is cost: replaying a long history is slower
and more expensive than a memory system that answers from a handful of
distilled facts. That claim is treated as falsifiable and measured, not
assumed — see [Evaluation](#evaluation) and **EXP-01 (LongMemEval-M-100)**.

The Python package is `longmem`; Memento is the project.

## Status

| Area | State |
|---|---|
| Core library (`src/longmem/`) | Write path, read path, lifecycle, Postgres/Redis adapters, service — complete |
| Test suite | 178 passing, 4 skipped (integration tests skip without backends or keys) |
| Evaluation metrics (`evals/`) | Complete: 5 axes, 33 offline tests |
| Experiments | One experiment run end to end (EXP-01 — LongMemEval-M-100); see Evaluation |
| Absolute quality numbers | Early — one question measured per engine, not a benchmark score |

## The approach

Two flows share one storage layer:

```
write:  turns → judge every N turns → candidate → find related → classify
                                               → CREATE | MERGE | SUPERSEDE | DISCARD
read:   query → understand → hybrid search → hard filter → score → rerank
                                                → Top-K → assemble context
```

### Write path — memory formation

The write path decides what deserves to survive a conversation, and it is
deliberately split so the LLM only makes judgements while the code makes
decisions:

| Step | Who | What happens |
|---|---|---|
| Trigger | code | Every N turns (`memory.write_interval_turns`), plus a closing flush of the session tail |
| Judge | LLM | Returns a `MemoryCandidate`: store or not, distilled content, type, scope, `valid_from`, confidence basis |
| Enrich | code | Adds id, timestamps, confidence from config, status, version, and provenance — no LLM, no DB |
| Relate | code then LLM | Hybrid search over the **active head only**, then classify as `no_relation`, `similar`, or `contradiction` |
| Consolidate | code | `CREATE`, `MERGE` (raise confidence, add evidence), `SUPERSEDE` (close old, link new), or `DISCARD` |

`SUPERSEDE` never overwrites: the old memory keeps its content, gets
`status=superseded` and `valid_until`, and the new one points back through
`supersedes_id`. History stays answerable, which is what makes knowledge-update
and temporal questions work at all.

`MERGE` is the dangerous operation, because it keeps the existing row's wording
and only raises confidence — so anything the candidate says that the row does
not is **deleted**. It is guarded twice: the classify prompt states that
`similar` means the same fact restated and that a shared topic is not enough,
and `consolidate()` deterministically vetoes any merge whose candidate is not a
restatement (embedding cosine ≥ `merge_similarity_min`, or content-term
containment ≥ `merge_overlap_min` when no embedder is configured). A vetoed
merge becomes a `CREATE` and logs a `merge_vetoed` event.

### Read path — retrieval to context

1. **Understand** — the LLM rewrites the query, extracts entities, types, scope,
   keywords, and `temporal_confidence` (how strongly the query constrains
   *time*, not which era). The era itself, `question_time`, is always supplied
   externally by the caller and is never extracted by the model.
2. **Hybrid search** — vector similarity (pgvector, cosine) and lexical rank
   (Postgres full-text) over the candidate set, combined with Reciprocal Rank
   Fusion.
3. **Hard filter** — a gate, not a score: expired memories out (unless deep
   retrieval is asked for), wrong type and wrong scope out. Superseded memories
   stay in, because they answer historical questions.
4. **Score** — additive and temporally gated:

   ```
   final = w_relevance·relevance + w_confidence·confidence + w_temporal·temporal
         + w_recency·recency − w_decay_eff·decay_penalty
   w_decay_eff = w_decay · (1 − temporal_confidence)
   ```

   A stale memory is *penalised*, never multiplied to zero: on a question about
   2023, a memory from 2023 beats a fresher one from the wrong era, and when the
   query is about the present the penalty applies at full strength. Confidence,
   decay, and recency are kept as separate numbers and reported per hit.
5. **Rerank** — an optional stronger model over the filtered slice only, then
   Top-K (`retrieval.final_memory_count`).
6. **Assemble** — active memories, historical (superseded) memories, and
   supporting evidence are rendered as three fixed sections. No LLM call.

### Lifecycle

Activity is `max(last_retrieved_at, last_updated_at)`, falling back to
`created_at`, so a brand-new memory is never stale. Inactivity drives exponential
decay (`exp(-λ·days)`), and a soft expiry threshold archives rather than deletes:
expired memories leave normal retrieval but remain reachable on request. Being
retrieved refreshes activity and **never** raises confidence — confidence is
evidence strength, not popularity.

### Storage

| Store | Technology | Role |
|---|---|---|
| Durable | Postgres + pgvector + full-text search | Memories, provenance, structured metadata, temporal fields, versioning |
| Working | Redis | Session turns and state, expiring with the session |

`src/longmem/store.py` defines the `DurableMemoryStore` and
`WorkingMemoryStore` protocols first, with in-memory implementations used by
every test and by Phases 02–03 of development. The Postgres and Redis adapters
satisfy the same contracts, so swapping backends changes constructors, not call
sites. Embeddings are nullable throughout, so the system runs without vectors.

### Configuration and models

Every tunable lives in `config.yaml` — turn interval, judge window, decay rate,
expiry, retrieval sizes, scoring weights, confidence values, model ids — and the
settings loader validates and freezes them at startup. Code reads from
`Settings`; there are no tuning literals in the pipeline.

Chat backends are pluggable: OpenAI, OpenRouter, Groq, or a local GGUF through
`llama.cpp`. Embeddings are always local and free
(`Qwen/Qwen3-Embedding-0.6B`, 1024-dim). A run can mix a hosted chat model with
local embeddings, which is the cheapest useful configuration.

## Evaluation

### Metrics (`evals/`)

Pure functions over ids, strings, and numbers. They import nothing from
`src/longmem`, on purpose: a metric that reuses the system's own scoring cannot
detect the system's errors.

| Axis | Metrics | The failure it isolates |
|---|---|---|
| Retrieval | recall@k, precision@k, reciprocal rank, mean RR | The right memory missing versus ranking it badly. Recall cannot see order; precision cannot see omission |
| Answer | exact_match, semantic_similarity, abstention_report | Right-answer-wrong-words, and the questions that have no answer, where hallucination and omission are reported separately rather than averaged |
| Temporal | temporal_gate_lift, split by question type | Whether the temporal gate does real work, and whether a gain on historical questions is hiding a loss on current ones |
| Usefulness | memory_lift, paired win/loss/tie counts, lift per 1k extra tokens, ceiling check, sham control | Whether memory beats no memory *at all*, and whether that lift is real or just "more tokens" |
| Consolidation | duplicate_rate sampled per write boundary, as a history with a trend | Whether the write path keeps up as a conversation grows — a rising rate and a uniformly high one are different problems |

The headline is deliberately `memory_lift`: the system is only useful if
answering with memory beats answering without it, and that is the one number
that can come back negative.

### Experiment EXP-01 — LongMemEval-M-100

**Name.** *EXP-01 · LongMemEval-M-100* — the 100-question stratified slice of
LongMemEval-M, built to make the benchmark runnable on free tiers and CPUs.
The metric definitions it uses are the real ones in `evals/`.

**Why a smaller set.** LongMemEval-M pairs each question with a ~482-session
haystack: 500 questions cost roughly 584,000 judge calls, which is not runnable
on free tiers or on a CPU in any reasonable time. LongMemEval-M-100 keeps the
question distribution and cuts the haystack.

**What was built.**

| Component | Purpose |
|---|---|
| Dataset builder | Builds LongMemEval-M-100: 100 questions stratified across all six question types (seed 0), each haystack reduced to the gold sessions plus 15 hard distractors ranked by lexical overlap with the question, then temporal proximity, then session length. 1,691 sessions / 20,139 turns / ~5,000 judge calls — ~117× cheaper than the full set. Preserves the original JSON keys and adds a `_try` provenance block plus a sidecar `meta.json` (seed, quota, source hash). |
| Multi-key LLM client | For free tiers: rotates keys, honours the provider's own "try again in Xs" hint, paces against a token-per-minute budget, falls back lazily to a local GGUF, and reports per-key usage and failures. |
| Per-question runner | Fresh stores, ingest in date order, snapshot the store at every write boundary, answer with *and* without memory for the paired lift, optional cross-question abstention probe, resume support. |
| Cheap configuration | 5-turn judge windows and the LLM reranker bypassed — the two measured cost drivers. |

**What it measured.**

| Finding | Value |
|---|---|
| Groq free tier, tokens per minute | 8,000 — 10-turn windows were rejected outright (413) |
| Groq free tier, tokens per day | 200,000 — **one question consumes ~85% of it**, so ~1 question/day/key |
| Local run cost | ~29–40 minutes and ~5.4 GB RAM per question, ~90–105 LLM calls |
| Full set, local | ~49 hours; ~16 hours if distractors drop from 15 to 5 |

**What it found.** Tracing a single question (asked where the user keeps old
sneakers) end to end surfaced three write-path defects, each of which deleted a
fact the system had already identified:

1. `parse_judge_json` rejected an entire candidate when an optional field was
   unusable — local models emit `"type": ""` — so a valid memory became a
   discard. Optional fields are now coerced to their defaults. This was 12% of
   judge calls in the measured run.
2. A judge exception returned `DISCARD`, indistinguishable from a deliberate
   decision. Failures now return `ERROR` and are counted as `judge_errors`.
3. `MERGE` keeps the target's wording, so two different assertions about the
   same subject ("plans to take old sneakers to a cobbler" versus "keeps old
   sneakers in a shoe rack") merged into one row, the newer fact was erased, and
   the older row's confidence rose. Fixed by the classify rule and the
   deterministic merge backstop described above.

A prompt fix was also needed: the rule "general-knowledge questions do not count"
was read as "this window is instructional, so store nothing", which discarded the
user's own aside containing the answer.

**Results — the same question, across the fixes.**

| Metric | Baseline | Final |
|---|---|---|
| recall@10 | 0.5 | **1.0** |
| MRR | 0.333 | **1.0** |
| semantic similarity | 0.5388 | **0.6738** |
| memory lift (with vs without memory) | +0.0100 | **+0.1450** |
| judge errors | 4 | **0** |
| duplicate rate | 0.0 | 0.0 (across 11 vetoed merges) |

**Caveats.** One question, one run per configuration — these are traces that
locate defects, not a score. The mock-based runs establish plumbing only. No
multi-question comparison of engines exists yet, and the veto's tendency to
trade merge for duplicate has only been observed at n=1.

## Running it

```bash
# install into the project venv
uv pip install --python .venv/Scripts/python.exe -e .

# tests (offline; integration and live-LLM tests skip without backends/keys)
python -m pytest -q

# infrastructure
docker compose up -d          # pgvector + redis

# query the durable store
python scripts/ask.py --query "where do I keep my old sneakers?"
```

Chat provider and model ids are set in `config.yaml`; API keys live in `.env`
(never committed).

## Repository layout

```
README.md        this file: approach, evaluation, experiment, and how to run it
src/longmem/     Memento's library: config, schemas, lifecycle, store, llm,
                 formation, retrieval, context, service
evals/           evaluation metrics (retrieval, answer, temporal, usefulness,
                 consolidation), independent of the library
tests/           offline test suite; integration tests are marker-gated
scripts/         thin CLIs over the real backends
plans/           the phase contracts the implementation was built against
config.yaml      every tunable, validated and frozen at load
```

## Design decisions worth knowing

- **Store distilled facts, not conversation chunks.** The judge decides what
  deserves retention; the durable store is not a transcript cache.
- **Never overwrite history.** Contradictions produce a supersede chain, so both
  the old and new state stay answerable.
- **Staleness is an additive, gated penalty.** Never a multiplicative
  kill-switch; the gate on the query decides how much freshness matters.
- **Confidence, decay, and recency are separate signals.** Retrieval raises
  activity, never confidence.
- **Expiration is archival, not deletion.** Expired memories leave normal
  retrieval but remain reachable.
- **Hard filters are gates, not scores.** Wrong type or scope never enters
  ranking.
- **LLM judges, code decides.** The model produces candidates and relation
  labels; the pipeline computes confidence, versions, and history.
- **Metrics judge from outside.** `evals/` never imports `src/`.
