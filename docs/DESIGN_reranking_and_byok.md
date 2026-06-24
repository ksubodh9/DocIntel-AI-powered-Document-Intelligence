# Design: Re-ranking & Bring-Your-Own-Key (BYOK)

Status: **Proposed** — no code written yet.
Author: PDF Agent engineering
Scope: Two independent features. Re-ranking improves retrieval quality; BYOK lets
users supply their own LLM provider and API key. They share no code and can ship
separately. Recommended order: **re-ranking first** (contained, low risk), then
**BYOK** (larger refactor + security work).

---

## Feature 1 — Re-ranking

### Problem

Retrieval today is a single stage. In `app/rag/vectorstore.py`:

```
query → embed_query → ChromaDB cosine search (top_k=5) → sort by distance → return
```

The embedding model (`BAAI/bge-small-en-v1.5`, a **bi-encoder**) encodes the query
and each chunk *independently* into vectors and compares them. That is fast enough
to score the whole collection, but it is a coarse relevance signal: it never looks
at the query and a chunk *together*, so the top-5 it returns are often in the wrong
order, and a genuinely-best chunk can sit at rank 6–8 and never reach the LLM.

### Approach

Add a second, more accurate stage between vector search and the LLM. A
**cross-encoder** reads `(query, chunk)` as a single input and outputs a relevance
score. It is far more accurate at ordering but too slow to run over an entire
corpus — so we use the classic **retrieve-then-rerank** pattern:

```
query
  → embed_query
  → ChromaDB cosine search (fetch top-N candidates, e.g. N=20)   ← over-fetch cheaply
  → cross-encoder reranker scores the N candidates
  → keep the best rerank_top_k (e.g. 5)                          ← what the LLM sees
  → existing relevance threshold + citation logic (unchanged)
```

Cheap recall first (vector search casts a wide net), accurate precision second
(cross-encoder reorders the short list).

### Model choice

Use **fastembed's `TextCrossEncoder`** with `BAAI/bge-reranker-base` (or
`Xenova/ms-marco-MiniLM-L-6-v2` for a lighter option). Rationale:

- The project already depends on **fastembed** (ONNX runtime, no torch). The
  reranker runs on the same stack — no new heavy dependency, image stays small,
  free-tier friendly. This mirrors the deliberate choice already documented in
  `app/rag/embeddings.py` to avoid torch + sentence-transformers.
- Keeps the door open to swap models via config, same as embeddings.

### Implementation outline

1. **New module `app/rag/reranker.py`** — mirrors the `embeddings.py` factory
   pattern:
   - `Reranker` protocol with `rerank(query: str, docs: list[str]) -> list[float]`.
   - `BGEReranker` wrapping `fastembed.TextCrossEncoder`.
   - `@lru_cache get_reranker()` singleton (lazy-loaded, like `get_embedding_model`).

2. **Change `retrieve_chunks()` in `app/rag/vectorstore.py`:**
   - When reranking is enabled, request `n_results = rerank_candidates` (e.g. 20)
     from ChromaDB instead of `top_k`.
   - Pass the candidate texts through `get_reranker().rerank(query, texts)`.
   - Attach the rerank score to each chunk dict, sort by it descending, truncate
     to `rerank_top_k`.
   - When disabled, behaviour is exactly as today (feature flag, zero behaviour
     change).

3. **`retrieve_chunks_multi()`** — note its current comment says "re-rank" but it
   only re-sorts by cosine. Apply the real reranker to the merged candidate pool,
   and rename internal references so "rerank" unambiguously means the
   cross-encoder stage.

4. **Config (`app/config/settings.py`), next to the existing RAG block:**
   ```python
   rerank_enabled: bool = False          # feature flag, default off
   reranker_model: str = "BAAI/bge-reranker-base"
   rerank_candidates: int = 20           # N fetched from ChromaDB before rerank
   rerank_top_k: int = 5                 # final count passed to the LLM
   ```

5. **Tests** — unit-test that reranking reorders a known-good candidate above a
   distractor; integration-test the enabled/disabled paths return the same shape.

### Score-scale gotcha (must handle)

`min_relevance_score = 0.35` is tuned to **cosine similarity [0–1]**. Cross-encoder
outputs are **logits on a different, unbounded scale** — applying the 0.35 cutoff to
rerank scores would silently discard everything or nothing. Two clean options:

- **Recommended:** apply the existing cosine threshold *before* reranking (filter
  the candidate pool on cosine), then rerank only for *ordering*. The downstream
  `min_relevance_score` logic keeps working unchanged.
- Or introduce a separate `min_rerank_score` tuned to the chosen model. More
  accurate, but adds a second magic number to maintain.

### Trade-offs

- **Latency:** adds ~50–200 ms per query (N candidates through the cross-encoder on
  CPU). Hence the feature flag and a modest `rerank_candidates`.
- **Memory:** one extra ONNX model resident (small for `*-base`/MiniLM).
- **Accuracy:** materially better top-k ordering, especially on long or
  multi-topic documents — the main reason to do this.
- **Blast radius:** contained to `reranker.py` + `vectorstore.py` + config. Chat
  service, citations, and the threshold logic are untouched.

**Verdict:** Workable and low-risk. Ship first.

---

## Feature 2 — Bring-Your-Own-Key (BYOK)

Let users run chat/summarisation against **their own** LLM provider and API key
instead of the server's keys.

### The core obstacle

Credentials are currently **process-global**. `LLMService` and the embeddings layer
read a module-level `settings` singleton (`get_settings()` is `lru_cache`'d), and
every provider method reads keys directly:

```python
def _gemini_complete(...):
    genai.configure(api_key=settings.gemini_api_key)   # global
def _openai_complete(...):
    client = OpenAI(api_key=settings.openai_api_key)   # global
```

There is **no per-request or per-user key path anywhere**. So BYOK is not a bolt-on;
it is a **decoupling of `LLMService` from the global settings singleton**. That
refactor is the shared foundation for everything below.

### Foundational refactor (required by both modes)

Introduce an explicit credentials object and thread it through dependency
injection, which the codebase is already structured for
(`get_document_service → get_llm_service`).

1. **`LLMConfig` value object** — `provider`, `model` (optional), `api_key`, plus
   any provider-specific extras. One per request.

2. **`LLMService` takes an `LLMConfig`** in its constructor instead of reading the
   global `settings`. Provider methods change `settings.X_api_key` →
   `self.config.api_key` and `settings.X_model` → `self.config.model or <default>`.

3. **`get_llm_service` becomes a resolver**: build the `LLMConfig` from, in order
   of precedence:
   1. credentials supplied on the request (BYOK), else
   2. the user's saved credentials (persisted mode, below), else
   3. the server's `.env` defaults (current behaviour — nothing breaks for users
      who don't bring a key).

4. **Provider allowlist:** the incoming `provider` string must be validated against
   a fixed set (`gemini|openai|anthropic|groq|huggingface|ollama`). Never let a
   user-supplied string select an arbitrary code path.

Once this is in place, **session-only mode is essentially done** and persisted mode
is an additive storage layer on top — same execution path, keys just arrive from a
different source.

### Two storage modes — the user chooses the risk

Per the decision to surface the trade-off, the **end user** picks how their key is
handled at entry time. Default to the safer option.

| | Session-only (default) | Saved |
|---|---|---|
| Key lives | In memory for the session only | Encrypted at rest in the DB |
| On logout/expiry | Discarded | Retained until user deletes |
| Re-entry | Each session | Once |
| Risk | Low — never written to disk | Higher — server stores a secret |
| Server work | Minimal | Encryption, storage, rotation, delete |

UI must state the trade-off plainly at the point of entry, e.g.:

> *"Session-only keys are never stored and are cleared when you log out. Saved keys
> are encrypted but kept on our server so you don't re-enter them. Choose what
> you're comfortable with."*

#### Mode A — Session-only (ship first)

- Client sends `provider` + `api_key` with the chat/summary request (or exchanges
  them once for a short-lived server-side session token — preferable to resending
  the raw key on every call).
- Server builds a per-request `LLMService` from them and **never persists** the key.
- Smallest security surface; delivers the whole feature.

#### Mode B — Saved (second increment, additive)

- **`user_credentials` table** (or Supabase user metadata): `user_id`, `provider`,
  `ciphertext`, `created_at`, `last_used`. Store **ciphertext only** — never
  plaintext.
- **Encryption at rest:** Fernet (symmetric) with the key from a KMS / secret
  manager / env var that is *not* in the repo. Decrypt only in memory at call time.
- **Management surface** (scope in from the start):
  - `GET /credentials` — list stored providers, **masked** (`OpenAI ••••4f2a`),
    never returning the key.
  - `DELETE /credentials/{provider}` — revoke / rotate.
- Decryption happens inside the `get_llm_service` resolver when no request-supplied
  key is present.

### Security hardening (both modes — non-negotiable)

- **Never log keys.** `_call_provider` currently logs prompts, and
  `_classify_api_error` logs **raw SDK exception strings**, which can echo the key
  or auth header. Add scrubbing before any BYOK key flows through.
- **Never return a key** in any API response; mask everywhere it surfaces.
- **TLS only** for transport (already required) — keys ride on requests in
  session-only mode.
- **Validate provider** against the allowlist (above).
- **Rate-limit / scope** validation calls to avoid turning the endpoint into a
  key-testing oracle.

### Credential validation

Today a bad key only surfaces when a real call fails mid-request. Add a lightweight
`POST /credentials/validate` that makes one cheap provider call (e.g. a 1-token
completion or a models-list) and returns ok/❌, so users get immediate feedback when
entering a key.

### Scope boundary — keep embeddings server-side (initially)

Restrict BYOK to the **LLM / chat layer**. Do **not** let users bring an embeddings
provider yet: changing the embedding model changes vector dimensionality and
similarity space, which **invalidates every existing vector store** — you cannot
query (or rerank) across mismatched embeddings. Keeping embeddings server-side
sidesteps a whole class of corruption bugs. Revisit only with per-(user,model)
collections.

### Trade-offs

- **Effort:** larger than re-ranking — the `LLMService` decoupling touches every
  provider method, plus storage + security work for Mode B.
- **Risk:** session-only is low; saved-keys carries real responsibility (you now
  hold user secrets) — hence user-consented opt-in and encryption.
- **Payoff:** users run on their own quota/models; reduces server key costs and
  rate-limit contention.

**Verdict:** Workable. Do the shared refactor + Mode A first; add Mode B as an
opt-in second increment.

---

## Suggested sequencing

1. **Re-ranking** — `reranker.py`, wire into `vectorstore.py`, config flag, tests.
2. **BYOK foundation** — `LLMConfig` + decouple `LLMService` from global settings +
   resolver in `get_llm_service` + logging scrub + provider allowlist.
3. **BYOK Mode A** — session-only request path + `validate` endpoint.
4. **BYOK Mode B** — `user_credentials` storage, Fernet encryption, list/delete
   endpoints, masking.

Each step is independently shippable and reversible behind config/flags.

---

## Implementation status (built)

All four steps are implemented and covered by tests (92 passing).

**Step 1 — Re-ranking.** `app/rag/reranker.py` (fastembed `TextCrossEncoder`),
wired into `retrieve_chunks` / `retrieve_chunks_multi`. Config: `RERANK_ENABLED`,
`RERANKER_MODEL`, `RERANK_CANDIDATES`. Off by default; fails open to cosine order.

**Step 2 — BYOK foundation.** `LLMService` driven by `LLMConfig`
(`from_settings` / `for_byok`); provider allowlist; `_scrub()` redacts keys from
logs. Backward-compatible.

**Step 3 — Session-only BYOK.** Per-request headers `X-LLM-Provider`,
`X-LLM-Api-Key`, optional `X-LLM-Model`. `POST /credentials/validate`.

**Step 4 — Saved BYOK.** `user_credentials` table, Fernet encryption
(`CREDENTIALS_ENCRYPTION_KEY`), `CredentialService`. Endpoints:
`GET /session/mode`, `POST /credentials`, `GET /credentials` (masked),
`DELETE /credentials/{provider}`.

### Request resolution precedence (`_resolve_llm_service`)

1. `X-LLM-Provider` + `X-LLM-Api-Key` headers → session-only BYOK.
2. User's active saved key (decrypted in memory) → saved BYOK.
3. Server `.env` default.

`X-LLM-Use-Default: true` forces the server default even if a saved key exists.

### Post-login flow the client should implement

1. After login call `GET /session/mode`. It returns `server_default_available`,
   `persistence_available`, `has_saved_credentials`, `active_provider`,
   `supported_providers`.
2. Offer the user: **Continue** (server default — send no key headers, or
   `X-LLM-Use-Default: true`) or **Use my own key**.
3. If own key, offer **This session only** (hold the key client-side, send it via
   `X-LLM-*` headers on each request) or **Save it** (`POST /credentials`;
   afterwards it's used automatically with no headers). Only offer "Save" when
   `persistence_available` is true.
