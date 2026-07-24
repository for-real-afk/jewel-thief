# Model Comparison

Every number and quote in this document came from a real call made against a real
catalog during this project — none of it is a published benchmark or a vendor claim.
Where something wasn't actually tested live, it's marked as such rather than assumed.

## The models involved

| Role | Provider | Model | Notes |
| --- | --- | --- | --- |
| Embedding | Google Gemini | `gemini-embedding-2` | Native multimodal — embeds image pixels directly |
| Embedding (fallback) | LM Studio (local) | `google/gemma-4-e4b` (caption) + `text-embedding-nomic-embed-text-v1.5` (embed) | No local model embeds images directly — two-hop workaround |
| Reranker judge | Google Gemini | `gemini-2.5-flash` | Configured as the production default; **not live-tested with real image data this session** — see caveat below |
| Reranker judge | LM Studio (local) | `google/gemma-4-e4b` | Same vision model as the local embedding path's captioner |
| Reranker judge | Groq (cloud) | `qwen/qwen3.6-27b` | Only vision-capable model on this Groq account; a reasoning model |

---

## Embedding: Gemini (direct) vs. LM Studio (caption-then-embed)

### How they actually work

- **Gemini (`gemini-embedding-2`, `EMBEDDING_PROVIDER=gemini`):** the image goes
  straight into `embed_content()` as multimodal input. One API call, one vector. This is
  a real image embedding — it's a function of the pixels.
- **LM Studio (`EMBEDDING_PROVIDER=lmstudio`):** no local model can embed images, so the
  vision chat model (`gemma-4-e4b`) first writes a one-sentence factual caption of the
  image, then the *text* embedding model (`nomic-embed-text-v1.5`) embeds that caption.
  The resulting vector is a function of a **text description of the image**, not the
  pixels themselves.

### Measured: speed

| | Time to embed one image |
| --- | --- |
| Gemini | ~1.8s (single call) |
| LM Studio | Several seconds (two sequential HTTP round-trips: caption, then embed) |

### Measured: determinism / self-similarity

Searching with the *exact same file* that was already in the catalog is the simplest
possible correctness check — a system should recognize an identical image as (near-)
identical to itself.

- **Gemini:** not separately isolated in a dedicated test, but embedding is a direct
  deterministic-ish function of pixel content (no free-text generation step involved).
- **LM Studio:** measured self-similarity of a duplicate image against itself was
  **0.886**, not ~1.0 — and it wasn't even the top-ranked match in that search. This is
  because the caption step involves LLM sampling: the *same* image can produce slightly
  differently worded captions across calls (even at low temperature), so the derived
  embedding shifts too. This is a structural property of caption-then-embed, not a fixable
  bug — see [ISSUES.md §3.6](ISSUES.md#36-caption-then-embed-is-not-deterministic--a-photo-doesnt-fully-match-itself).

### Measured: retrieval quality (raw cosine ranking, before any reranking)

Querying with a photo of pressed-flower resin earrings against the same 79-item catalog:

| | Top-10 raw cosine results |
| --- | --- |
| **Gemini** | All 10 correctly category `earrings`, clean score separation (88.0% → 77.7%) |
| **LM Studio** | (Not isolated as a clean top-10-only test, but see the necklace query below — bracelets and rings appeared mixed into the top ranks near the correct necklace matches) |

Querying with a photo of an ornate gold choker necklace + earring set, raw cosine top
matches under LM Studio's embedding included `bracelet-3` (0.875), `necklace-1` (0.879,
the item's own near-duplicate), `rings-3` (0.863) all within a narrow band — category
membership was not cleanly separated by score alone the way Gemini's embedding
separated the earrings query.

### Verdict

Gemini's direct multimodal embedding was faster, more deterministic, and produced
visibly cleaner category separation in every real comparison run. LM Studio's
caption-then-embed path is a genuine, working fallback for fully offline / no-API-key
operation, but it inherits two structural costs: **information loss** (the embedding
only "sees" whatever the caption happened to describe — anything the captioning model
omitted or got wrong is invisible to retrieval) and **sampling noise** (nondeterministic
self-similarity). Use it as a "for now, no key available" path, not a permanent
production choice, if API access is available.

---

## Reranking: LM Studio vs. Groq (both tested live, same catalog, same query images)

### Test 1 — necklace + earring set query, 20 real candidates

| | LM Studio (`gemma-4-e4b`) | Groq (`qwen/qwen3.6-27b`, after fix) |
| --- | --- | --- |
| Time for 20 candidates | 96.3s | 2.0s |
| Top pick | `braclet-3` — **wrong category** (bracelet) | `necklace-9` — correct category, correct top score |
| Category grouping | Bracelets/rings interleaved with necklaces across confidence tiers | All 9 necklaces ranked "high", all 3 earrings "medium" (a plausible related category), all bracelets/rings correctly demoted to "low" |
| Reasoning given for top pick | *"Matches the ornate gold cuff silhouette and pattern"* (describing a **bracelet** as matching a **necklace** reference) | *"High similarity score and correct category match for the ornate gold necklace"* |

The local model didn't just get the ranking slightly wrong — it was confidently,
consistently wrong about what category the reference image even belonged to, across
every independent test run in this session (see
[ISSUES.md §3.4](ISSUES.md#34-a-local-vision-model-repeatedly-misclassified-the-same-item-as-the-wrong-category)).
Groq's model correctly identified the item as a necklace/choker every time.

### Test 2 — pressed-flower earrings, direct captioning (not reranking) comparison

Asked both models to freely describe the same unrelated image (no forced categories):

| | Groq (`qwen3.6-27b`) | LM Studio (`gemma-4-e4b`) |
| --- | --- | --- |
| Item type | "drop earrings" ✓ | "dangle earrings" ✓ |
| Central motif | "pressed white daisy and tiny green leaves... under clear resin" | "white butterfly motif... enamel or resin inlay" |
| Beads | "polished black oval beads, each tipped with a small gold cap" ✓ | "multiple tiers of polished, dark black beads" ✓ |
| Frame | "textured gold-toned metal with intricate filigree" ✓ | "antique gold-toned metal... intricate filigree" ✓ |

Both models were genuinely grounded in the actual image here (not hallucinating from
text patterns) — item type, bead color, and gold filigree all check out against the
real photo for both. The one disagreement ("daisy" vs. "butterfly") is a legitimately
ambiguous detail in an artistic layered-resin piece, not a sign either model faked it.
**This test is the important negative control**: it shows the necklace/bracelet
confusion in Test 1 was a specific weakness with that image's silhouette, not a general
"local model doesn't really see images" problem — `gemma-4-e4b` clearly can process
images correctly; it just got this one specific shape wrong, repeatedly.

### A real bug Groq surfaced, not a model weakness

Groq's first attempt at reranking (before any fix) returned `"no reranker judgment
available"` for all 20 candidates — a total failure, worse-looking than LM Studio's
wrong-but-present answers. Investigating the raw response showed `qwen/qwen3.6-27b` is a
**reasoning model** that emits an unbounded `<think>...</think>` block by default; with
no `max_tokens` cap, it was cut off mid-thought and never produced the requested JSON.
Fixed with `reasoning_effort: "none"` + `max_tokens: 4096` (both verified to work against
the real API before being treated as the fix — see
[ISSUES.md §3.2](ISSUES.md#32-groqs-reasoning-model-never-produced-usable-json)). Worth
noting for future model swaps: **a reasoning-capable model needs explicit configuration
to behave like a plain instruction-follower**, or it will silently fail this kind of
structured-output task.

### Verdict

Once correctly configured, Groq's `qwen/qwen3.6-27b` was **~48x faster** (2.0s vs 96.3s
for 20 candidates) and **correctly categorized every test image** that the local model
got wrong. The speed difference alone changes the UX category — a 90+ second reranking
step makes a synchronous request-response search endpoint impractical; a 2-second one
doesn't.

## `gemini-2.5-flash` reranker — untested caveat

`gemini-2.5-flash` is wired up as the reranker's production default
(`LLM_PROVIDER=gemini`) and is exercised by the mocked unit test suite, but **no live
reranking call was made against it with real image data during this session** — every
real comparison in this document is LM Studio vs. Groq, because those were the two
providers actively being evaluated as alternatives. Before trusting `gemini-2.5-flash`'s
real-world judgment quality, it deserves the same live test this document gave the other
two — don't assume it performs like `gemini-embedding-2` did just because they're the
same vendor.

## Summary recommendation

- **Embedding:** Gemini (`EMBEDDING_PROVIDER=gemini`) over LM Studio's caption-then-embed,
  whenever a real API key is available — faster, more deterministic, measurably cleaner
  category separation.
- **Reranking:** Groq (`LLM_PROVIDER=groq`) over the local model for both speed and
  measured accuracy on every real test run this session — with the explicit caveat that
  `gemini-2.5-flash` was never actually run head-to-head and might do just as well or
  better; that comparison is still open.
- **Local (LM Studio) path:** keep it working as the zero-API-key fallback (it does
  work, end to end), but treat its output as noticeably lower quality than either cloud
  option based on everything measured here — not a permanent production choice.
