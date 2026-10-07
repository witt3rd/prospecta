# Prospecta v0.1 — Critic Review (Round 2)

**Role:** Critic (Round 2)
**Reviewer:** Forge ⚒️
**Date:** 2026-05-18
**Subject:** `stance-planner-v2.md` against `review-critic.md` (Round 1) and `review-architect.md` (Round 1), under the orchestrator's curated treatment map.

---

## Verdict

**APPROVE_WITH_PROVISO.**

Round 2 closes the orchestrator's curated subset principled (not deferentially). The counterfactual deference test passes on 7 of 8 adoptions — defenses are shaped to the *specific* fix, not to "Critic-said-so." Budget growth is honest. The one tracer concern I'd been ready to raise against myself (Test 2: did I over-prescribe?) survives scrutiny — it's small, injected, default no-op, and structurally aligned with P3. The 2×2 matrix is a real strengthening of M7, not ceremony.

What I'd hold back on consensus for is *small*: two named gaps in §14's `llm` failure-mode contract that are silently absent in the NEW frame — both concerning raw-response propagation on the JSON-mode fallback path. Provisos detailed below. None blocks dispatch; all can be folded during distillation.

---

## Test 1 — Counterfactual deference, per major adoption

For each Round 2 adoption, the question: *would the Planner's defense also justify a counterfactual fix?*

| Adoption | Counterfactual alternative | Defense transfers? | Read |
|---|---|---|---|
| **M0 port-decision sheet as gate** | Keep sheet as M2 deliverable, not M0 gate | No. The §M0 defense is *"M2 inherits ambiguity if the sheet isn't a gate"* — this argument actively rejects the in-M2 placement. | **Principled.** |
| **2×2 bilateral matrix** | 3-cell (OFF/OFF + ON/ON + one mid) | No. §7's "what this proves" enumerates assertions that *require* both ON/OFF and OFF/ON to prove asymmetric load-bearing. Three cells can't separate write-side from read-side load. | **Principled.** |
| **JSON-mode to v0.1** | Document line-parser as known P6 compromise | No. §8 is shaped around "P6 line-parsing is gone." The defense doesn't carry to "keep line-parser, ship test cases." | **Principled.** |
| **Item (f) rewrite** | One-paragraph defer (Critic's actual S3 ask) | Partial. §10's defense is "implementer knows what they get for free" — that defense *could* justify any non-empty plugin section. S3 was let-slide; this is an inside-the-let-slide refinement. Acceptable. | **Mixed (deferential to the let-slide constraint, principled within it).** |
| **O1 metadata_filter rename** | Collapse all scoping to `path_contains` (i.e., keep the animus shape) | No. Defense is "library has no concept of corpus structure." This rejects `path_contains` as a privileged surface. | **Principled.** |
| **O2 prompt rename + translation** | Keep name, treat as +30min edit | No. §M1 enumerates per-prompt translation deltas; "+30min" can't deliver them. Budget delta is +1 day, which sizes to translation not rename. | **Principled.** |
| **Tracer primitive** | Document `logger.info` as the debug path | No. §12 lays out 6 named events with structured payloads; the OpenTelemetry-adapter hook in Open Question 3 explicitly treats payload as semi-public API. That defense doesn't justify "use logging." | **Principled.** |
| **Per-Memory `threading.RLock`** | Document single-threaded contract, no lock | No. §13 explicitly asserts "safe to call from multiple threads" and walks sweeper-vs-retain interaction with reentrancy. Defense doesn't justify single-thread-documented-only. | **Principled.** |

**Result:** 7 of 8 adoptions pass the counterfactual test cleanly; item (f) passes within the let-slide constraint. This is *not* a deferential pattern-match; the Planner reshaped each defense to fight the specific cut.

That changes my Round 1 worry. I'd been ready to find pattern-match. I don't.

---

## Test 2 — Did I over-prescribe in Round 1?

Walking my Round 1 catches against what Round 2 adopted.

### M1 — tracer primitive

Defensible in v0.1, not over-prescription. Three reasons:

1. The tracer is ~30 LOC (§12). Cost is real but bounded.
2. Default no-op lambda means zero runtime cost when unused.
3. Spine debugging is the *load-bearing* use case (P1). Without per-query rank visibility, "the recall returned the wrong document" requires re-running with prints. That's worse than acknowledged.

Open Question 3 (treating payload schema as semi-public) is the right honesty.

**Verdict:** kept. Not over-prescribed.

### M3 — per-Memory RLock

The lock solves retain-vs-retain in a single process. It does *not* solve two processes sharing a chroma dir (notebook + cron on the same `~/animus/index/`). Round 2's §13 acknowledges this implicitly by scoping to "per-instance" but doesn't loudly flag multi-process.

This is closer to papering than I want. But: (a) v0.1's caller surface is "single process, possibly threaded" — the canonical caller (animus) is one process; (b) chromadb's own SQLite layer handles some cross-process coordination; (c) v0.2 multi-writer is acknowledged as deferred.

**Verdict:** kept, but the §13 docstring should explicitly say "single-process contract; cross-process callers share at their own risk." That's a one-line addition I'd let slide if there's no Round 3.

### O2 — `log-index` → `generate-index-text` rename

Not theater. The prompt body is being *translated* (drop episode/spine language, drop "your memory bank" voice, regeneralize). The name change is downstream of the substance change. If the rename were Round 1's ask and the substance stayed Cookie-shaped, that would be naming theater. Round 2 does both.

**Verdict:** kept, real ontology.

---

## Test 3 — What did Round 2 make WORSE?

### Budget +67% (7.5 → 12.5 days)

Decomposing:
- M0 sheet: +0.5d (new)
- M1 prompt translation: +1.5d (Cookie-shape rewrite is real)
- M2 widening for 7 sites: +1.5d
- M5 JSON-mode + translation: +0.5d
- M7 2×2 corpus tuning: +0.5d
- M8 README: +0.5d
- Total additive: ~5d, matches +5d delta

This is honest growth, not sprawl. Round 1 was light on translation work; A1+A2 caught that. Round 2 prices it correctly.

**Verdict:** not worse; corrected.

### 9 milestones now — any ceremony?

Eight of nine carry a runnable demo + test gate. M8 (README, 0.5d, ship gate) is the soft one. It could plausibly fold into M7's ship gate as a deliverable rather than its own milestone. **This is the only ceremony-shaped milestone in the plan.** Marginal — let-slide.

### Item (f) rewrite

"Decisions inherited from settled context" actively serves the implementer better than "what's NOT in this plan." Reading the new (f), an implementer learns *what falls out of the locked decisions* (prefetch on, sync_turn no-op, per-session via MemoryManager). The Round 1 framing left them with a list of things to figure out. Round 2 lists what they already have. This is a real improvement.

**Verdict:** better, not worse.

### Anything lost from Round 1 that was good?

The Round 1 self-graded "what an ideal plan would have" was useful as a Planner-honest checklist of known gaps. Round 2 retires it. I think that's correct — those gaps are now closed or named as Open Questions. Nothing lost that was load-bearing.

---

## Test 4 — NEW silently-absent in the NEW frame

Walking P1–P15 against Round 2's new surfaces (tracer, concurrency, README, JSON-mode).

| # | Round 2 honors? | Note |
|---|---|---|
| P1 spine | ✓ | 2×2 strengthens; bilateral now provable, not asserted. |
| P2 standalone | ✓ | No new Hermes coupling. |
| **P3 LLM injected, no provider imports** | ⚠ | Tracer pattern is correctly injected (default no-op lambda, no `logging` opinions smuggled at the *tracer* level). §14 does call `logger.warning` for sweeper failures — Python `logging` is stdlib, not a provider import, so this is P3-compliant. But the README should clarify: the library uses stdlib `logging` for its own internal warnings; the tracer is for caller-side observability. The two channels exist; the docs should name them. |
| P4 caller wins | ✓ | Per-call overrides restored (M2). |
| **P5 full content, no truncation** | **✗ partial** | §14 row 6: malformed JSON → "falls back to single-query degraded mode." The raw malformed response is **dropped on the floor**. Tracer fires `parse_fallback=True` but the actual response bytes are not in the payload. A caller debugging "why is my formulate_queries always falling back?" can't see what the LLM actually returned. **This is a P5 violation in the NEW frame** — the JSON-mode promotion (A8) creates a new failure path that loses load-bearing content. See Proviso 1. |
| P6 let LLM cook | ✓ | JSON mode retires line-parser, as called for. |
| P7 single write path | ✓ | Unchanged. |
| P8 embedded chroma | ✓ | Unchanged. |
| P9 thin adapter | ✓ | Item (f) rewrite respects bound. |
| P10 surface adjacent | ✓ | Unchanged. |
| P11 tests over prose | ✓ | 2×2 sharpens. |
| P12 honest config | ✓ | Unchanged. |
| P13 per-call override | ✓ | Restored. |
| P14 sweeper safety net | ✓ | Lock contract clarifies. |
| **P15 spine in README** | ⚠ | §11 spec is sound. One small risk: the "what makes prospecta different from LlamaIndex/Khoj/mem0" section in the README spec is leaning competitor-takedown. Plan acknowledges in Open Question 4. Donald-reviewed-before-ship is the right gate. |

**Tracer-specific check (per delegated brief):** the tracer is injected, default no-op, structurally aligned with P3. *Not* silently smuggling `logging` opinions. The library's own internal warnings use stdlib `logging`, which is correct separation. No issue.

**JSON-mode raw-response check (per delegated brief):** see Proviso 1 below. The raw response should propagate. P5 implies it.

**README spec check (per delegated brief):** does it leak settled-decisions to users? The §11 README spec includes "v0.1 is sync only; embedding-model migration is v0.2; bilateral costs 2 LLM calls per retain and 1+N per recall_synth." These are *user-relevant* honest disclosures — not internal-architecture leaks. The chroma-embedded-default disclosure is appropriate. The version-cut disclosures are appropriate. **No leak.**

---

## Provisos (small, named, foldable during distillation)

1. **§14 malformed-JSON path must preserve the raw response (P5).** When `formulate_queries` JSON parsing fails and falls back to single-query mode, the raw LLM response string MUST land in the tracer payload (`parse_fallback=True, raw_response=<str>`) AND in a `logger.warning` log line. Anything less drops load-bearing debugging data and violates P5 silently. ~5 LOC. Spec change to §12 (`formulate_queries` tracer payload schema) and §14 (row 6 behavior column).

2. **§14 must cover schema-mismatch alongside malformed-JSON.** A well-formed JSON response that doesn't match `{"queries": [{"text": "..."}, ...]}` (e.g., LLM returns `{"results": [...]}` or `{"queries": "kelly birthday"}` as string-not-list) is currently silent on policy. Treat as malformed-JSON: same fallback, same logging, same raw-response capture. One row in the §14 table.

3. **§13 concurrency contract should explicitly state single-process scope.** Add a one-line note: *"v0.1's concurrency guarantee covers a single process. Multiple processes pointing at the same `index_dir` is undefined behavior in v0.1; v0.2 may address via chroma-server or file locking."* Without this, a future caller assumes the RLock covers cross-process; the lock does not. ~1 line addition to §13.

4. **(Optional / let-slide-eligible) M8 README ship-gate could fold into M7.** Saving 0.5d of ceremony. If the Planner prefers M8 as its own gate for milestone-counting hygiene, that's defensible too — it's not load-bearing either way.

Provisos 1–3 are real; Proviso 4 is cosmetic. All four together fit in <1 page of distillation diff. None requires reopening locked decisions or contesting the let-slide list.

---

## Things I'd let slide (in addition to the orchestrator's let-slide list)

- The 7th `index.py` site is named-by-count, not enumerated. Open Question 1 names this honestly; M0 surfaces it. Honest deferral.
- The 2×2 corpus tuning risk in M7. Acknowledged in §M7 with M8 as bumper. Acceptable.
- Anthropic-direct JSON-mode adapter complexity (10 LOC vs 3 for OpenAI/Hermes). Documented in README per Open Question 2. Not blocking.
- The "bilateral" terminology in PRINCIPLES.md and README (my Round 1 O4). Documentation reframe at README time; let it ride.

---

## Closing

The Planner did what counterfactual deference test is meant to catch: shaped each defense to the *specific* fix, not to "Critic said so." That's principled adoption. The budget grew honestly. The one silently-absent in the NEW frame is the JSON-mode fallback's raw-response loss — a P5 violation downstream of A8's promotion to v0.1. Small fix, foldable.

If Provisos 1–3 land in distillation, consensus is reached.

⚒️
