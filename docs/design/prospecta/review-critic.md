# Prospecta v0.1 — Critic Review (Round 1)

**Role:** Critic (Round 1)
**Reviewer:** Forge ⚒️
**Date:** 2026-05-18
**Subject:** `stance-planner.md` + `review-architect.md` — challenging the framing, not compliance-checking

---

## Verdict

**REQUEST_CHANGES** (closable — none of the contests is structural; all can be folded into a Round-2 patch). The plan basically works. The Architect's eight reservations are real and addressable. What both stayed inside the frame for: the plan is *port-shaped* when the work is *translation-shaped*, ships a "bilateral" claim that the test gate doesn't actually prove, smuggles two pieces of animus ontology forward, silently fakes P13, and lacks a tracing primitive for debugging the spine when it misfires on a real corpus. None of these blocks dispatch; all should land before Round 2 closes.

---

## Framing contests

### F1 — "Implementation plan" is the wrong artifact for the work that's actually left

The orchestrator (me) picked **implementation-plan-shaped** because the assessment ran five rounds and locked five decisions. That framing pulled the Planner into Planner-shape: 11 sub-dimensions, 7 milestones, locked signatures, a self-graded APPROVE.

The work that's actually left is smaller and stranger than that. The spine is settled. The dependency surface is locked. The locked decisions cover the architectural forks. What remains is: (a) port ~1800 LOC of animus code with translation pressure on roughly a third of it, (b) rewrite three prompts away from Cookie-shape, (c) write *one* test (M7) that proves the spine is bilateral. That is a **~3-page MV checklist**, not a 658-line implementation plan.

The cost of the over-shape is real: the Planner spent budget on ceremony (sub-dim numbering, "what an ideal plan would have," self-graded verdict) that pushes the eye away from the two actually-hard things — the bilateral test corpus and the prompt rewrites. Both got the lightest treatment in the plan.

**Counter-proposal:** Round-2 retains the locked signatures section and the M7 spec, demotes everything else to a checklist, and explicitly relocates depth into the bilateral-test corpus design (which currently fits in one page and deserves three).

### F2 — The "port from animus is mostly mechanical" framing is optimistic

Architect's A1 (seven entanglement sites, not three) and A2 (prompts are Cookie-shaped at the narrative layer, not the surface layer) are the same observation surfacing twice: **animus's memory layer was not built generic and then specialized for Cookie. It was built for Cookie and the genericity has to be carved out.**

The Planner's mental model is "port = mechanical." The lived shape is "port = translation effort." These produce different plans. A port plan has milestones gated by tests-pass. A translation plan has milestones gated by *decisions made about each animus-ism*. M2 in the current plan opens against an undecided foundation (the seven `person` sites). The Architect's A1 fix — a port-decision sheet — is the right move, but it should be promoted from a deliverable inside M2 into a **gate before M2 opens**. Otherwise M2 inherits the ambiguity.

**Counter-proposal:** Insert M0 — "Port-decision sheet: every animus-specific call site in `index.py`, `queries.py`, and the three prompts gets a delete/generalize/defer verdict before any porting work starts." Half a day. M2's budget then holds at 2 days as planned.

### F3 — The bounded plugin section is doing different work than its bound was for

The orchestrator locked the plugin section at ≤2 pages to prevent depth-theater on what the assessment had already named as mechanical. The Planner honored the bound mechanically: 6 items in 2 pages. But the section's content shows the cost of bounding: item (f) is "what's NOT in this plan," and that list contains *the actual plugin design questions* — error handling, retry semantics, timeout config, per-session scoping, agent-context awareness. The bounded section is a fig leaf.

This is fine *if* "read hindsight and adapt" actually works. It probably does. But the plan should *say so explicitly*, not pretend it has a plugin section.

**Counter-proposal:** Cut the bounded plugin section in v0.1's plan entirely. Replace with one paragraph: "Plugin is derived in a separate ralplan after v0.1 lands. Implementer reads `/home/dt/src/ext/hermes-agent/plugins/memory/hindsight/__init__.py`, adapts to prospecta's API, ships as a ~400 LOC adapter PR. Estimated 1 day." Save the substance for when prospecta v0.1 is shippable and the actual plugin questions can be answered against a working library.

---

## Ontology contests

### O1 — `Query.path_contains` smuggles animus's scope ontology forward

The plan defines `Query.path_contains: str | None = None` as "optional caller-supplied scoping" (P4). The intent is honest — let the caller filter by path. The implementation is animus's `path_contains='person/<p>'` mechanism renamed.

Prospecta has **no concept of corpus structure**. A library that ships chroma + chunker + RAG should not know what a "path" is at the API surface. The caller knows their corpus shape; the library exposes a metadata filter. The right shape is:

```python
@dataclass(frozen=True)
class Query:
    text: str
    metadata_filter: dict[str, Any] | None = None   # chroma-shaped, caller-owned
```

`path_contains` becomes one possible filter (`{"path": {"$contains": "person/"}}`), not a privileged API surface. Animus's existing caller code can wrap with a helper; prospecta stays generic.

This is small but load-bearing for P2 (standalone first). Shipping `path_contains` as a first-class field is the kind of ontology smuggle that a v0.2 caller will hit and have to work around.

### O2 — `log-index` as a prompt name is animus's episode shape leaking

The Architect's A2 catches the narrative-layer animus-binding in `log-index.md`. The deeper move is: **the prompt is named for a thing prospecta does not have.** Animus generates index_text for `log.jsonl` episodes; prospecta generates index_text for arbitrary `retain`-supplied content. The right prompt name is `generate-index-text.md`, and the right framing is "you are generating index text for an arbitrary content artifact" — no log-episode pretense at all.

The Planner's "+30 min edit" budget treats this as a renaming exercise. The Architect's "+2-3 hours per prompt" is closer. The Critic's read: this is a *write from scratch using log-index as reference*, not an edit. Budget half a day.

### O3 — `log.jsonl` in the indexable file set is also leakage

The locked decisions specify the indexable set as `{.md, .py, .yaml, .yml, .txt} + log.jsonl`. The first five are file extensions; `log.jsonl` is a **filename convention from animus's episode pipeline**. Prospecta has no episodes. Why does prospecta know about a JSONL log filename?

Defensible answer: animus is the first caller, and animus has `log.jsonl` files in its substrate, and prospecta indexing those is useful. But that's caller-shaped, not library-shaped. The right move is: v0.1 indexes the five extensions; animus drops a hook that says "also index `log.jsonl` as line-per-document," or animus pre-processes its logs into `.md` before pointing prospecta at them. Either way, `log.jsonl` does not belong in the library's locked indexable set.

**This was settled in the assessment and is technically out-of-bounds for this Critic round.** Flagging anyway because P11 says re-litigate-settled-things is a smell, but ontology leakage is the kind of thing the orchestrator should reopen at v0.2 boundary.

### O4 — "Bilateral" oversells the symmetry of the spine

The plan and PRINCIPLES.md both speak of *bilateral* LLM-mediated retrieval as if write-side and read-side were mirror operations. They are not.

- **Write-side:** one LLM call that generates one-or-more question-shaped strings from a content artifact.
- **Read-side:** one LLM call that fans out N query-shaped strings from one user message, then runs N parallel searches, then a synthesis call.

These are not symmetric. The shared insight is *both sides cross-pollinate into question-space*, which is real and load-bearing. But calling it "bilateral synthesis" implies a structural mirror that doesn't exist. A more honest framing: **question-space embedding on both ends of the index/recall arc.**

This is a documentation contest, not an implementation contest, but it bears on the README (P15). If the README leads with "bilateral synthesis" the way the plan does, drive-by readers will look for the mirror and not find it.

---

## Missing dimensions

### M1 — No tracing primitive for spine debugging

When `recall_synth` fails on a real corpus (returns the wrong document, returns the right document but synthesizes nonsense, returns empty), what does the user inspect?

- `index_text` lives in chroma metadata: ✓ inspectable.
- The formulated queries: ✗ ephemeral, lost after the call.
- Per-query rank ordering: ✗ not surfaced.
- Which query matched which result: ✗ not surfaced.

`RAGResult.queries: list[Query]` is in the spec. That's a start. But there's no per-query → results mapping. The plan's logging answer ("logger.info") is not enough for debugging a P1-claim spine.

**Counter-proposal:** Add `RAGResult.trace: dict | None = None` carrying the per-query results map and rank scores. Optional and opt-in via `recall_synth(message, trace=True)`. ~20 LOC, gates against debugging-by-spelunking when the spine misfires.

### M2 — P13 is silently downgraded from per-call to directory-level

P13 says prompts are "overridable by the caller" — explicitly noting that animus, with refined prompts already, will override. The plan exposes `prompts_dir: str | Path | None = None` as a constructor arg.

That is *directory-level* override. To use different prompts for different calls, the caller must instantiate a new `Memory` (rebuilding the chroma client, the sweeper, everything). This is the opposite of "first-class caller override."

The right shape is per-call: `recall_synth(message, prompts: dict[str, str] | None = None)` where the dict maps prompt-name → prompt-text. Defaults to library prompts; caller-supplied wins. animus passes `{"formulate-queries": its_own_prompt}` per call.

This is a real P13 violation. The plan checks the box at the directory level and the principle gets quietly weakened.

### M3 — Concurrency model for two simultaneous retains is unspecified

Architect's A7 catches sweeper-vs-retain. What about retain-vs-retain? Two callers (notebook + cron, two threads in the same process, two processes sharing a chroma directory) calling `Memory.retain(content=..., index_text=None)` simultaneously. Both go through `index_single_file`. Is there a lock? Is chromadb's internal lock sufficient for write-write? What about file-on-disk collisions in `retain_dir` if both auto-generate filenames?

The plan is silent. The honest answer is probably "v0.1 documents single-writer assumption; multi-writer is v0.2." Say that.

### M4 — `llm` callable failure modes are partly addressed (A3) and partly not

Architect's A3 covers `formulate_queries` fallback. What about:

- `retain(content=..., index_text=None)` and `llm()` raises mid-call. Does retain partial-write (content on disk, no chroma upsert)? Drop the file? Retry?
- `recall_synth(message)` and the synthesis call (after retrieval succeeds) raises. Does the caller get partial results or an exception?

The plan is silent. The library needs a stated policy: **fail-loud-and-leave-substrate-clean** is the right default (don't half-write to chroma). Document it.

### M5 — No config schema versioning

`prospecta.toml` lands in v0.1. v0.2 adds `multi_corpus: dict` or `async_mode: bool`. What happens to v0.1 configs? Silent ignore? Loud error? `prospecta_config_version: 1` field?

Small but real. One line in the spec: "v0.1 configs accept unknown keys with a debug-log warning; breaking schema changes bump a `prospecta_config_version` field."

### M6 — Animus migration story is missing

Donald is the canonical caller. He has `~/animus/index/` with a chroma collection embedded by a specific model. Can prospecta point at that collection and read it? Or does animus migration require a rebuild?

The plan implicitly says rebuild (`Memory(index_dir=...)` constructs fresh). For the canonical caller, that's expensive and silent.

**Counter-proposal:** v0.1 README explicitly states: "if pointing prospecta at an existing chromadb index, use `chroma=...` with the same embedding model and collection_name; otherwise rebuild." Two sentences.

### M7 — README is gestured at, not specified

P15 is the README. The plan says M1 ships the README "leads with the spine." That is not a deliverable; it is an aspiration. P15 is a load-bearing principle and the plan ships zero README content.

The Planner self-graded "what an ideal plan would have" item 7 acknowledges this. That acknowledgment doesn't deliver P15.

**Counter-proposal:** Round-2 includes a 200-word README skeleton — headline, value-prop paragraph, install, three code examples, "why not LlamaIndex/Khoj" callout. That's the principle deliverable. Without it, P15 is fake-shipped.

---

## Simplicity challenges

### S1 — Ship the write-side spine alone in v0.1; read-side spine in v0.2

Radical alternative: v0.1 ships `retain` with LLM-generated `index_text`, plus `search(text) → results`, plus a single-query `recall(query) → RecalledMemory[]`. No `formulate_queries`. No `recall_synth`. The bilateral test becomes a **monolateral** test: prove that auto-generation of question-shaped `index_text` finds documents that lexical embedding misses, with the caller's query being a single human question.

Cost: gives up half the P1 claim in v0.1. Benefit: validates the *harder* half (write-side, where you have to convince yourself LLM-generated questions are better than chunk embedding) on a real corpus before doubling LLM cost on read.

I do not recommend this. But the Planner should have **considered and rejected** it explicitly, and didn't. The implicit assumption that v0.1 must ship both halves to "prove the spine" is exactly the kind of all-or-nothing that the simplicity test catches. The right answer is probably "ship both," but the answer should be earned.

### S2 — Collapse M1+M2+M3 into one milestone

M1 (scaffolding + types), M2 (chunker + parser + ignore + index porting + classical RAG), M3 (write-side spine) — these are not three independent merge gates. They are one continuous arc: *get retain → search working end-to-end*. The Planner pattern-matched milestone-shape to PR-shape. Three PRs of one day each is bureaucracy; one PR of three days is a deliverable.

Collapse to M1 (scaffolding through write-side spine, gated on the retain roundtrip test) + M2 (recall_synth with formulate_queries, gated on a recall test) + M3 (sweeper + CLI) + M4 (bilateral test). Four milestones, ~7 days. Same ship gate.

### S3 — Plugin section is one paragraph, not 2 pages

Already argued in F3. The 6-item bounded section is doing ceremony work. Replace with: "Plugin derived in follow-on after v0.1 ships. Read hindsight, adapt. 1 day."

---

## Principle audit (P1–P15)

| # | Plan honors? | Note |
|---|---|---|
| **P1 spine is the spine** | ⚠ partial | Honored at M7. Plan treats spine as the *last* gate, not the *load-bearing throughline*. Every milestone is leaf-test-shaped. |
| **P2 standalone first** | ✓ | No Hermes imports. CLI is argparse. Embedded chroma default. |
| **P3 LLM as injected callable** | ✓ | Strict. No provider imports. No "default LLM." |
| **P4 caller wins on override** | ⚠ partial | API level: ✓. Per-call prompt override: ✗ (see M2). |
| **P5 full content, no truncation** | ✓ | `RecalledMemory.content` carries full body. |
| **P6 let the LLM cook** | ⚠ partial | `_parse_scoped_queries` is a Python heuristic on LLM output (Architect's A8). P6 says no regex parsing where a prompt suffices — JSON-mode is the P6-honest answer. Plan defers. |
| **P7 single write path** | ✓ | `index_single_file` is the one writer. |
| **P8 embedded chroma zero-config** | ✓ | Default. |
| **P9 plugin is thin adapter** | ✓ | ≤2-page bound honored (but see F3). |
| **P10 surface adjacent mechanisms** | ✓ | jinja2 ports verbatim; chromadb owns its pin. |
| **P11 tests over prose** | ✓ | Each milestone gates on tests. M7 is the lock. |
| **P12 honest config** | ✓ | 6 fields. |
| **P13 prompts caller-overridable** | ✗ | Only directory-level (`prompts_dir`). Per-call override is not exposed. P13 says "callers can override per-call." Silent downgrade. |
| **P14 sweeper is safety net** | ✓ | Direct calls, not queued. Architect's A7 sharpens. |
| **P15 spine in README** | ✗ | Asserted but not delivered. No README content in plan. Self-graded "ideal plan" acknowledges. |

**Two outright misses (P13, P15), one partial that the Architect already caught (P6/A8), one partial on the framing of P1.** The plan does not mention P3/P5/P7/P10 explicitly anywhere in its body — they are honored by structure, not by named discipline. That is acceptable, but a Round-2 patch should include a P-by-P table that the Planner's self-grade item 8 already promised the Critic would do. (Doing it here.)

---

## Agreements with the Architect

- **A1** (entanglement sites): backed and sharpened — promote port-decision sheet from M2-deliverable to M0-gate.
- **A2** (Cookie-shape in prompts): backed and sharpened — these are rewrites, not edits. See O2.
- **A3** (formulate_queries fallback): backed. Don't merely test fallback; document the policy.
- **A4** (2x2 spine-isolation matrix): backed and **most important catch in the Architect's review**. Without this, M7 proves "spine helps," not "*bilateral* spine is load-bearing."
- **A5** (`_debug_*` kwargs off public init): backed.
- **A7** (chroma client thread-safety + concurrency test): backed and extended to M3 (two simultaneous retains, see M3 above).

## Disagreements with the Architect

- **A6** (embedding_model): Architect recommends defer-and-raise. Critic recommends **defer-silently with a debug-log warning, but keep the parameter accepting on the constructor as a v0.2-shaped affordance**. Raising NotImplementedError on a parameter the user passes is hostile; warning-and-using-default is the honest middle.
- **A8** (line-parser robustness): Architect undersells. The line parser is a **P6 violation** (regex parsing where a prompt suffices). The cheap fix (three more test cases) papers over the principle problem. The right fix is **JSON-mode in v0.1**, not v0.2. `complete_structured` is documented as available. The Hermes adapter wraps to a 5-line `llm_json` callable. The `LLMCallable` shape gains a second optional callable. This is a v0.1 scope decision the orchestrator should reopen.

---

## Must-fix before consensus (prioritized)

1. **M0 port-decision sheet** (Architect A1 promoted to gate).
2. **A4 spine-isolation matrix** in M7.
3. **P13 per-call prompt override** (`recall_synth(..., prompts={...})`) added to the locked signatures.
4. **README skeleton** (P15 — 200 words, see M7 above).
5. **`Query.path_contains` removed**, `metadata_filter: dict` added (O1).
6. **`log-index.md` renamed and rewritten** as `generate-index-text.md` (O2 + Architect A2).
7. **M3 retain-vs-retain concurrency policy stated** (single-writer assumption documented).
8. **JSON-mode decision reopened** (P6 conflict; A8 disagreement) — either bring `complete_structured` into v0.1 or document the line-parser as a known P6 compromise.

## Things I'd let slide

- The 7-milestone vs 4-milestone shape (S2). Bureaucracy, not breakage.
- The 2-page plugin section (S3). Cosmetic.
- Config schema versioning (M5). Honest deferral to v0.2.
- Animus migration story (M6). README sentence; not blocking.
- The "bilateral synthesis" terminology (O4). Honest documentation reframe at README time.
- Plan's omission of P3/P5/P7/P10 from explicit discussion. Honored by structure.

---

## Closing

The plan ships v0.1 essentially as written. The Architect's catches are the real load-bearing fixes; the Critic's job was to find what neither saw because both stayed inside the implementation-plan frame. What was silently absent: P13 per-call override, the README that delivers P15, the tracing primitive that makes spine debugging possible, a clean separation of caller-shape (scope, path filtering) from library-shape (text + metadata + chroma), and an honest reckoning with whether the line-parser is a P6 violation or a v0.2-acceptable compromise.

None of this blocks dispatch. All of it should land in Round 2 before consensus. The plan is stronger when it stops pretending the port is mechanical and starts naming the translations honestly.

⚒️
