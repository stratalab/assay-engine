# Assay — Implementation Plan

*Milestones and epics for building Assay. Sequences the five specs
([vision](product/assay-vision.md) · [PRD](product/assay-prd.md) ·
[UX](product/assay-ux.md) · [engineering](product/assay-engineering.md) ·
[chisel-alignment](chisel-alignment.md)) into an ordered build with hard gates. Epics
trace back to the PRD's `A-*` requirements.*

---

## 0. Shape of the build

Three milestones plus a foundation:

| Milestone | What ships | Gate |
|---|---|---|
| **M0 — Foundations** | repo, CI + license gate, the embedded-store seam, the core data types | green skeleton; a trivial IR round-trips end to end through the store, keyed by its content hash (execution is M1) |
| **M1 — Deterministic spine (v0)** | the whole engine with **no model** — IR → resolve → execute → verify → reproducible answer | the v0 acceptance test (§2) |
| **M2 — The product (v1)** | NL via the embedded inference layer; the Chisel candidate→verified pipeline; plugin SDK; HTTP API | the v1 acceptance test (§3) |
| **M3 — Hardening & edge** | new domains, web UI, packaging polish, deeper integrations | per-epic |

**The critical path to v0** is short and linear: **E0.1 → E0.3 → E1.1 → E1.2 → E1.5 →
E1.6 → E1.7**. The resolver (E1.3), verification (E1.4), and the store (E0.2) run in
parallel and join at the v0 gate.

**Two external dependencies** (not Assay code — flagged per epic): the **embedded
StrataDB surface** (E0.2) and the **Strata inference layer** (E2.1). Both sit behind an
interface so Assay's engine develops without blocking on them — and Assay is a forcing
function for both to mature.

---

## 1. M0 — Foundations

### E0.1 — Repo, CI, and the license gate
- **Goal:** a green skeleton with the engineering doctrine mechanized from commit one.
- **Delivers:** package layout (engineering §4); `pyproject.toml` + `uv.lock`; CI running
  ruff + mypy + pytest on the Linux/macOS/Windows matrix; the **license-scan gate** that
  fails on any non-permissive runtime dep (engineering §2.4).
- **Depends:** —
- **Done:** CI green; the gate's classifier is proven by unit tests — permissive passes,
  copyleft and unknown-license fail closed (no need to poison the dep tree to watch CI fail).

### E0.2 — The embedded store seam  *(external dep: StrataDB)*
- **Goal:** the store behind an interface — StrataDB as the production backend, adopted on
  its own maturity clock, never blocking the spine.
- **Delivers:** the `store/` interface (namespaced KV + append-only lineage); an
  **in-memory backend** (the dev/test default, so compute epics don't block); the
  **StrataDB adapter** over the published `stratadb` PyO3 SDK — an **optional extra**
  (`assay[stratadb]`) until StrataDB's embedded surface is production-ready, with CI
  installing the extra so the adapter is exercised on all three platforms.
- **Delivers (reqs):** A-11 (provenance/versioning store); **stages** A-10 — the seam +
  adapter now; the flip to StrataDB-by-default is **E2.8**.
- **Depends:** E0.1.
- **Done:** an artifact round-trips through both backends — StrataDB via the extra, in CI,
  on all three platforms. *(Gating risk: StrataDB maturity — the in-memory default keeps
  M1 moving while it matures; Assay's engine stays a pure-Python wheel meanwhile.)*

### E0.3 — Core data types: the IR + the answer object
- **Goal:** the two objects everything else moves.
- **Delivers:** the IR schema (pydantic) + content-hash + `ir_version`; the four-part
  answer object (one shape for terminal + JSON), engineering §4, PRD §6/§10.
- **Delivers (reqs):** A-1 (IR is the execution contract), A-11.
- **Depends:** E0.1.
- **Done:** an IR validates, hashes stably (order-independent), and round-trips to/from JSON.

---

## 2. M1 — The deterministic spine (v0, no model)

### E1.1 — Template schema + `validate_template()`  *(the Chisel seam)*
- **Goal:** the declarative template contract + the shared validator.
- **Delivers:** the template schema (PRD §7.1); `validate_template()` — standalone,
  pydantic+stdlib only, the function Chisel imports (chisel-alignment §10); the template
  registry + the two tier fields (`license_tier`, `status`).
- **Delivers (reqs):** A-3, A-9, A-15.
- **Depends:** E0.3.
- **Done:** the golden beam template validates; a malformed template is rejected with a clear reason.

### E1.2 — The generic executor  *(security-critical)*
- **Goal:** run any declarative template, safely.
- **Delivers:** **safe** symbolic parse (`parse_expr`, locked namespace — never
  `sympify`/`eval`, engineering §7); unit-bound evaluation (Pint); dimension check of the
  result vs the declared output. `kind: formula` only (solver is later).
- **Delivers (reqs):** A-1, A-13.
- **Depends:** E0.3, E1.1.
- **Done:** the beam template evaluates to 0.50 mm; a formula string attempting code
  execution is rejected, not run (a security fixture proves it).

### E1.3 — The resolver + curated tables
- **Goal:** facts from trusted sources, never the model.
- **Delivers:** the resolver; a small curated `constants`/`materials` table with **per-value
  provenance + license** (engineering §2.5); missing-input handling — resolve / ask /
  fail-closed (PRD §8, UX §5.2).
- **Delivers (reqs):** A-2, A-8.
- **Depends:** E0.3.
- **Done:** `E` for steel resolves from the table with its source recorded; an unresolvable
  input is declared missing, never fabricated.

### E1.4 — The verification stage
- **Goal:** check the answer before returning it.
- **Delivers:** dimensional consistency, plausibility bounds, and one cross-method check;
  withhold-with-reason on failure (PRD §9, UX §5.6).
- **Delivers (reqs):** A-6.
- **Depends:** E1.2.
- **Done:** a wrong-units template fails the dimensional check; a cross-method disagreement withholds.

### E1.5 — The three golden declarative templates
- **Goal:** prove the schema + executor across real tasks.
- **Delivers:** `solve_equation`, `integrate`, `beam_deflection` as declarative templates
  with fixtures (PRD §16 v0); the **`kind: symbolic` method** the two symbolic goldens
  require — curated executor-owned operations (solve, integrate) on a gated `setup`
  problem, with built-in substitution/derivative verification (PRD §7). The beam is the
  golden shared with Chisel.
- **Delivers (reqs):** A-3.
- **Depends:** E1.1, E1.2, E1.3.
- **Done:** all fixtures green under the generic executor.

### E1.6 — The four-part answer + reproducible artifact
- **Goal:** the answer object, persisted and rerunnable.
- **Delivers:** result / interpretation / method / artifact assembly; `assay run <artifact>`;
  pinned-version capture; persistence to StrataDB (PRD §10, engineering §6 NFR-2).
- **Delivers (reqs):** A-7, A-11.
- **Depends:** E0.2, E1.2.
- **Done:** an answer's artifact reruns to an identical result (exact same-platform); the
  reproduction report states exact vs within-tolerance.

### E1.7 — The deterministic CLI
- **Goal:** the no-model surface.
- **Delivers:** `assay solve / integrate / units / run / show` (+ `--ir`, `--provenance`,
  `--method`), argparse-based; the concise answer rendering + progressive disclosure
  (UX §2, §5.1, §5.8).
- **Delivers (reqs):** A-4.
- **Depends:** E1.5, E1.6.
- **Done:** the UX §5.1 solve flow and §5.8 inspect flows work end to end.

### E1.8 — Rendering: plots & diagrams from verified data
- **Goal:** figures that are faithful views of verified data, never independent answers.
- **Delivers:** the `render/` module (Matplotlib); generic render primitives (function-plot,
  scatter, geometry-diagram) + the optional IR **render directive**; deterministic figures
  (fixed backend, no timestamps, SVG); `assay plot` (+ `--solve`); the compute→render split
  (PRD §10.1, UX §5.9); analytic geometry via SymPy `geometry`.
- **Depends:** E1.2 (verified data to render), E1.6 (figure as an answer artifact).
- **Done:** *"plot and solve x²−5x+6=0"* (deterministic form) renders a figure whose marked
  roots come from the verified solve; the figure reproduces byte-stably; a figure fixture
  asserts features from data, not pixels.

### ✅ M1 gate — the v0 acceptance test
`assay ask` is not built yet; this is all deterministic. **Done when:** each golden template
answers a worked example correctly, reproducibly, with per-fact provenance; **`beam_deflection`
declares `E` and `I` missing, resolves `E` for steel from the curated table (never
fabricated), and reproduces**; the E1.8 plot flow renders a byte-stable figure whose marked
features come from verified data; all fixtures green; ruff + mypy + full pytest green on the
platform matrix; **no model anywhere in the loop.**

---

## 3. M2 — The product (v1)

### E2.1 — The inference seam: NL → IR  *(staged, like the store seam E0.2)*
- **Goal:** natural language, contained.
- **Delivers:** the seam (`assay/inference`) — the `InferenceBackend` interface and
  **propose → validate → execute** (`validate_candidate` checks every candidate IR
  against its template contract *before anything runs*: hallucinated tasks, undeclared
  or wrong-dimension inputs, and ungated expressions are rejected, every reason stated);
  `assay ask`; the honest states — missing-input ask (interactive) / fail-clear
  (`--batch`), ambiguity fork + `--pick`, out-of-scope refuse, and `--emit-ir` →
  `assay run <ir.json>` for the correct-by-editing loop (UX §3, §5.2–§5.5). Two
  backends ship behind the seam: the **deterministic rule-based default** (no model —
  catalog-keyword task matching + dimension-keyed quantity extraction) and the
  **llama.cpp binding** (`assay[llm]` extra, `llama-cpp-python`, MIT — a local GGUF via
  `assay ask --llm model.gguf`; pinned seed/temperature, JSON-object output). The llama
  binding is the interim stand-in that the **embedded Strata inference layer**
  (llama.cpp + Lithos serving + optional OpenAI/Anthropic/Google routing, PRD §11)
  replaces behind the same interface when its SDK ships.
- **Delivers (reqs):** A-5, A-8, A-12.
- **Depends:** E1.* (the spine). The final stage — the Strata-layer binding — waits on
  that SDK.
- **Done:** a natural-language beam question yields the correct verified answer; an
  under-specified one asks; an out-of-scope one refuses. *(Met by the deterministic
  backend, proven in CI — `tests/test_inference.py`; the llama binding passes the same
  pre-execution gate, with proposal quality owned by the served model.)*

### E2.2 — The candidate → verified tier gate
- **Goal:** turn extraction into trusted content, safely.
- **Delivers:** the promotion path (`assay.templates.promote` — a submodule so the A-15
  seam stays dependency-light): `fixture_gate` (run the fixtures, report, no side
  effects — Chisel's local test-promote), `promote` (the only place `verified` is
  minted: a passing candidate gets a verified copy, pure; a failing one raises with
  every fixture's reason; a *claimed* `verified` label is re-earned, never imported),
  and `ingest` (validate through the A-15 seam → gate → register: passing serves,
  failing registers **quarantined** — visible, inspectable via the explicit
  `allow_candidate` opt-in, refused by default). Serve-only-`verified` stays enforced
  by `TemplateRegistry`; the two tier axes kept distinct (chisel-alignment §7):
  promotion never reads or writes `provenance.license_tier`. The CLI's point-of-use
  gate now delegates here.
- **Delivers (reqs):** A-14.
- **Depends:** E1.2, E1.5.
- **Done:** a candidate with failing fixtures stays quarantined; one with passing fixtures
  promotes and serves. *(Proven in `tests/test_promotion.py`.)*

### E2.3 — The Chisel content pipeline  *(cross-repo; renegotiated — round 2)*
- **Goal:** Chisel populates the first real domains.
- **Delivers:** ingest Chisel-emitted candidate templates + fixtures at the
  **pipeline fixture floor (≥3, `PIPELINE_FIXTURE_FLOOR`)**; the **cross-verification**
  (extracted formula vs Chisel's sandbox answer — both sides evaluate with Pint in
  base units, and Chisel runs it as a pre-emit gate too, round 2 §5); the first
  populated domains **in the negotiated order**: (a) the **34 AP-C pre-horizon tasks**
  as `newtonian_mechanics`/`em` — the end-to-end seam proof, some landing as fixtures
  on existing goldens, the rest as new candidates; then (b) **Mechanics of Materials**
  once a canon book is acquired (human-gated on Chisel's side) and their emitter
  ships. Ownership: Chisel owns extraction/abstain/eval-firewall; Assay owns the
  schema, the resolver key vocabulary (`assay facts --json`), and `verified`.
- **Depends:** E2.2; **E2.9** (the emitter builds against the post-E2.9 schema, not the
  current one); Chisel's emitter (their new work, not started until E2.9 lands).
- **Done:** a Chisel-emitted batch loads, cross-checks clean, and promotes to verified
  at the floor — first AP-C, then MoM.
- **Progress:** the first curated content waves shipped through the seam and are
  developed in the private content repository (batches, attachments, vocabulary
  requests, per-batch review records). The seam, loaders, gates, and tripwires
  are all here; the corpus is not.

### E2.4 — The plugin SDK
- **Goal:** domains without core changes.
- **Delivers:** templates as discoverable, versioned plugins (`assay.templates.plugins`)
  on the packaging ecosystem's own machinery — a distribution declares an
  `assay.templates` entry point whose target returns template *records* (data, not
  code); Assay validates each through the A-15 seam and serves nothing without the
  E2.2 fixture gate (**install grants presence, never trust**). Origin travels as the
  distribution's name + version; broken plugins (import failure, provider raising,
  invalid records, id collisions) are contained and reported per entry, never fatal.
  The CLI catalog becomes shipped + plugins (a plugin cannot shadow a shipped id), so
  `assay ask` routes to plugin templates — including into the ambiguity fork; `assay
  domains` lists everything covered with its source.
- **Delivers (reqs):** A-9.
- **Depends:** E1.1, E1.5.
- **Done:** a third-party template package installs and serves without touching core.
  *(Proven in `tests/test_plugins.py` with a real path-installed distribution: a
  `beamkit` package's cantilever template answers a natural-language question,
  verified, with facts still resolver-owned.)*

### E2.5 — Breadth: the v1 nucleus domains
- **Goal:** the ~dozen templates that make it a product.
- **Delivers:** templates across algebra / calculus / units / numerical methods / statistics /
  basic physics & engineering (PRD §15 nucleus) — hand-authored and/or Chisel-populated.
  Shipped: **12 goldens across 10 domains** — the E1.5 three plus `differentiate.univariate`
  (a new curated symbolic operation, verified by the numeric **difference quotient** — a
  genuine cross-method check), `projectile.range`, `kinetic_energy.point_mass`,
  `gravitational_potential_energy.point_mass`, `pendulum.period.simple`,
  `ideal_gas.pressure`, `resistor.voltage_drop`, `wave.speed`, `axial_stress.bar` — each
  with fixtures, bounds where natural, and `g`/`R` resolved from `assay.constants`
  (never baked in). Units remain the deterministic `assay units` verb. *Statistics and
  numerical methods need list inputs / `kind: solver` — they land with E3.5, not as
  shoehorned formulas.*
- **Depends:** E2.2 (so populated ones can verify).
- **Done:** the nucleus domains each answer a worked-problem set at target pass rate (PRD §17).
  *(Encoded in `tests/test_nucleus.py`: 12 worked problems, one per template, run
  NL → propose → validate → resolve → execute → verify — pass rate 100%, failures named.)*

### E2.6 — HTTP API + the agent contract
- **Goal:** the machine surface.
- **Delivers:** the answer object over HTTP (FastAPI/Starlette, the optional
  `assay[api]` extra; `assay serve`): `POST /v1/ask` (NL, batch semantics — never
  prompts, never fabricates), `POST /v1/run` (a caller-built IR — the agent's primary
  verb), `POST /v1/rerun` (reproduce an artifact: exact / within-tolerance / failed),
  `GET /v1/domains`, `GET /v1/health`. Every computed response carries the `Answer`
  object verbatim (per-check `verified`, per-fact `facts[].source`, `ir_hash` +
  pinned `versions` — UX §6) **plus the full artifact**, so the server stays
  stateless and the caller can reproduce bit-for-bit. Honest states are first-class
  `outcome` shapes (`missing_inputs` returns the understood-so-far IR to complete and
  re-POST; `ambiguous` + `pick`; `out_of_scope`); engine refusals are HTTP 400 with
  the reason (A-12). Same serving gate as every surface: promote-at-use (E2.2, A-14).
- **Delivers (reqs):** surfaces (PRD §12).
- **Depends:** E1.6.
- **Done:** an agent gets a stable, citable, reproducible JSON answer. *(Proven in
  `tests/test_api.py`: the beam answer cites `assay.materials steel.structural.E
  v0.1`; an artifact from one call reruns `exact` via another.)*

### E2.7 — Provider routing + IR-model attribution
- **Goal:** honest multi-provider inference.
- **Delivers:** record which model/provider produced an answer's IR (engineering §11 Q1);
  reproducibility caveat surfaced when a hosted provider is used. Shipped:
  `IR.produced_by` (`Producer`: provider + model) — every backend stamps its proposals,
  so the artifact names the NL→IR model; hand-built IRs carry `None` (no model to
  name). Attribution is **provenance, not content**: the content hash and cache key
  ignore it (the same IR is the same computation whoever wrote it). The caveat
  (`re-asking may read differently; the artifact reruns this exact IR`) surfaces on
  `assay ask` output and `assay show --provenance` for any non-`assay` producer — the
  deterministic rules carry no caveat (same question, same IR). *Provider routing
  itself lives in the Strata inference layer and arrives with its SDK; the attribution
  machinery here is ready for it.*
- **Depends:** E2.1.
- **Done:** an answer's artifact names the NL→IR model. *(Proven in
  `tests/test_attribution.py`, for both shipped backends and the save/load round trip.)*

### E2.8 — StrataDB adoption: default backend + hard dependency  *(closes A-10)*
- **Goal:** flip the store from "seam + optional adapter" to embedded StrataDB by default.
- **Delivers:** `stratadb` moves from the optional extra to a runtime dependency; the
  default backend flips `memory` → `stratadb` (persistent, under the user data dir); the
  in-memory backend stays for tests.
- **Depends:** E0.2; StrataDB's embedded surface declared production-ready.
- **Done:** a fresh `pip install assay` runs on embedded StrataDB with nothing extra to
  install; artifacts, cache, and lineage persist across runs.

### E2.9 — Emitter seam prep  *(round 2 blockers; the emitter builds against THIS schema)*
- **Goal:** the schema commitments made in round 2, shipped before Chisel's emitter starts.
- **Delivers:** **relative fixture `tol`** (default 1e-6, 1e-12 absolute floor for
  zero-expected — matching the task-bank semantics so the cross-check is mechanical;
  golden expects recomputed exact under the new semantics);
  **DAG-of-assignments** in `kind: formula` (ordered named `steps`, each through the
  same safe-parse gate — combined loading / Mohr's circle shapes); a written design
  note on `kind: table` + the `cases` discriminator (ships as a `schema_version` bump
  when the first mined batch needs it, designed against real rows). Already shipped
  ahead of this epic (round 2): the key vocabulary export, the fixture-floor
  machinery, the `unknown`-tier refusal, the vendored vocabulary pin.
- **Depends:** E1.1, E2.2.
- **Done:** the beam golden and one multi-step template validate and pass fixtures
  under relative tol; the seam pin can be taken by Chisel with no further semantic
  changes planned before MoM. *(Delivered: `_within_tolerance` — relative, 1e-12
  absolute floor, default 1e-6; the beam golden's expect recomputed exact;
  `FormulaMethod.steps` (ordered, gated, no forward references, no shadowing; last
  step is the result) with `principal_stress.plane.max` as the shipped multi-step
  golden + its worked problem; the `kind: table`/`cases` position in
  [`table-templates.md`](table-templates.md). The schema is now pinnable —
  `tests/test_templates.py`, `tests/test_nucleus.py`.)*

### E2.10 — The solve-for execution mode  *(confirmed round 5; the next Assay build)*
- **Goal:** answer the half of physics pedagogy that inverts equations silently.
- **Delivers:** formula template + target input + values for the other inputs *and*
  the output → symbolic inversion through the existing gated machinery, roots filtered
  by the target's declared dimension, **verification by forward substitution** at the
  *recovered input's* print precision (Chisel's tolerance-provenance rule); both-roots
  honesty (physical selection is a stated assumption or a fork, never silent); an
  optional `solve_for` fixture key. The printed-forms boundary stands: book-printed
  rearrangements remain content; solve-for covers the rest. Design set: Chisel's five
  round-5 fixtures (quadratic-in-t ramp, cyclotron sqrt, LC reciprocal-square, Van de
  Graaff linear — the two-regime consistency check — and the RC logarithmic case).
- **Depends:** E1.2, E2.9. — **Done:** the five design fixtures pass; the Van de Graaff
  case agrees with its printed-form template. *(Delivered: `assay/execute/solve_for.py`
  — inversion composes through E2.9 DAG steps too (recovering txy from a principal
  stress works); `Fixture.solve_for`/`output` + `IR.solve_for`/`given_output`
  (computational fields, in the content hash); the pre-execution gate, resolver,
  artifact, and rerun all carry it; verification is the independent
  **forward-substitution** check; refusals name every dropped root. All five design
  fixtures green through the ordinary fixture gate — `tests/test_solve_for.py`.)*

### E2.11 — Schema v2: list inputs, cases, tables  *(exhibits in hand, round 5)*
- **Goal:** the method shapes one textbook proved the schema still lacks.
- **Delivers:** `{"many": true}` list inputs + whitelisted reducers (`sum`,
  `sum_inverse`) — designed against the series/parallel resistor and capacitor rows;
  the `cases` discriminator (`moment_of_inertia.standard_bodies` is the exhibit);
  `kind: table` reserved for genuine interpolated chart reads (MoM bytes pending; the
  key→value split with resolver vocabulary is contract); the inequality/multi-output
  typology question. `schema_version: 2` validates alongside 1; arity-suffixed ids
  (`five_series` …) retire in a coordinated re-emit when this lands.
- **Depends:** E2.9; Chisel exhibits (in hand). — **Done:** the five-series row emits
  as one list-input template and gates green; the moment-of-inertia table emits as one
  `cases` object. *(Delivered: `schema_version: 2` — `TemplateInput.many` list inputs +
  the `REDUCERS` whitelist (`sum`, `sum_inverse`), structurally gated (a reducer takes
  exactly one bare list-input name; a list input appears **only** inside a reducer;
  no resolve hints on lists) and executed by **ast expansion to plain whitelisted
  arithmetic** (`sum(R_i)` → `R_i__0 + R_i__1 + …` — evaluation gains no new
  machinery); `kind: cases` — `setup[discriminator]` selects the case expression, every
  case is gated and every fixture must select a real case, with per-case fixture
  coverage; `IR.inputs` carries `Quantity | list[Quantity]` end-to-end (gate, executor,
  verify, artifact, rerun-exact); v1 stays **frozen** (v2 features on a v1 record are
  refused); solve-for over lists explicitly refused at the boundary. `kind: table`
  stays reserved pending MoM bytes ([`table-templates.md`](table-templates.md)).
  Acceptance = Chisel's round-5 exhibits verbatim: the 90 Ω five-series row, the
  parallel/series-capacitor `sum_inverse` rows, and the m58330 moment-of-inertia table
  as one `cases` object — `tests/test_schema_v2.py`.)*

### E2.12 — The catalog taxonomy  *(catalog governance at 400+ templates)*
- **Goal:** hierarchy before the flat domain list becomes a mess.
- **Delivers:** `templates/taxonomy.json` — **subject → topic → domain**, curated
  Assay-side data (the batches and their `domain` strings are untouched; the Chisel
  contract is unchanged): Mathematics / Physics / Chemistry / Engineering over the 22
  reconciled domains. The **lockstep rule**: every catalog domain is placed exactly
  once and every placement points at a live domain — an unplaced domain is a red CI,
  so the catalog cannot silently sprawl. One-time cleanup folded in: our three stray
  golden singletons aligned to the shared vocabulary (`oscillation→oscillations`,
  `electricity→electromagnetism`, `astronomy→gravitation`). Surfaces: `assay domains`
  renders the tree (subject → topic (count) → domain → templates), `/v1/domains`
  carries `subject`/`topic` per entry (additive), `GET /v1/taxonomy` serves the tree,
  and the web UI footer groups by it.
- **Depends:** E2.3 content. — **Done:** the lockstep test holds over the full
  catalog; `tests/test_taxonomy.py`.

### E2.13 — The mathematics & statistics extension  *(the three-book retrieval)*
- **Goal:** equip the schema for College Algebra, Calculus, and Introductory
  Statistics before Chisel extracts them.
- **Delivers:** the statistical reducer vocabulary — `count`/`mean`/`sum_sq`
  (arithmetic expansion, like E2.11's `sum`), `min`/`max`/`median` (order statistics:
  bound at input time on base-unit magnitudes — no expansion exists, the evaluation
  walk stays untouched), and paired `sum_product(x, y)` over two equal-length lists
  (the regression shape; unequal lengths refuse by name). **Multi-output**:
  `extra_outputs` on steps-DAG templates — named intermediate steps become reported,
  dimension-checked outputs (slope AND intercept, mean AND std, as one template);
  fixtures may expect every declared output. **New symbolic operations**: `limit`
  (setup `point`/`direction`, verified by a numeric approach sequence), `integrate`
  with `limits` `[lo, hi]` incl. `"oo"` (definite/improper — exact where SymPy is
  exact, `"oo"` states divergence; verified by independent quadrature / partial-
  integral growth), and `solve_inequality` (relational grammar; the answer is
  Assay-rendered canonical interval notation — `(-oo, -3] U [3, oo)` — stable across
  SymPy versions; verified by test-point membership). `erf` joins the safe namespace
  (the normal CDF; the widening ledger in `sandboxing.md`). **Backward
  compatibility**: reducer names are reserved only under `schema_version: 2` and only
  in call position — the shipped verbatim corpus (scalar inputs named `count`)
  validates unchanged.
- **Depends:** E2.11. — **Done:** mean+std and slope+intercept each emit as ONE
  template and gate green; sin(x)/x → 1 verified by approach; ∫₁^∞ dx/x states
  divergence and proves it; x² ≥ 9 answers in interval notation verified at 11 test
  points — `tests/test_math_stats.py`. *(Delivered same-day; reference artifacts
  regenerated on the `extra_outputs` serialization change.)*

### E2.14 — The coverage map  *(targeted growth, not bulk ingestion)*
- **Goal:** corpus growth is curated per-topic, never ingest-everything — the
  planning counterpart of the E2.12 taxonomy.
- **Delivers:** `templates/coverage.json` — **subject → field → topic** (2–3 levels),
  each topic marked **pending / in-progress / complete**, naming its intended source
  and (where blocked) the engine gate. Lockstep both ways: no non-pending topic
  claims an unshipped domain, and every shipped domain is claimed by a non-pending
  topic — a new domain forces a same-day coverage entry, like the taxonomy placement
  rule. `assay coverage` renders the map with per-topic template counts. The
  subject/field vocabulary is **anchored to CIP 2020** (NCES Classification of
  Instructional Programs — US government, public domain), not invented; engineering
  fields carry a `scope_reference` to the published **NCEES FE exam specification**
  whose topic list defines what "complete" means for practice-grade coverage.
- **Depends:** E2.12. — **Done:** the lockstep tests hold over the full catalog;
  `tests/test_coverage.py`. *(Delivered; the full coverage map — the curated subject universe with its
  sources and gates — lives in the private content repository. This repo ships a
  demo-catalog map with the same structure and lockstep tests.)*

### E2.15 — The execution trace  *(step-by-step that IS the computation)*
- **Goal:** show the work — as the literal computation record, never a narration.
- **Delivers:** `TraceStep` + `Answer.steps` (E0.3 additive): every steps-DAG
  execution records each step's name, authored expression, and computed base-unit
  value *from the same run that produced the answer*; cases record the selected
  case; solve-for records each recovered root with its inversion provenance;
  single-expression templates record their one evaluation. The trace travels in the
  answer object and artifact (byte-stable, reruns), renders as a numbered **Steps**
  band in the CLI and the web glass box, and is withheld with the answer (A-6) — a
  failed verification keeps its steps in the candidate. Upgrades every shipped
  steps template at once: steps were already data (E2.9), so no re-extraction.
  Deliberately out of scope: micro-derivation steps inside the algebra
  ("subtract 3 from both sides") — a curated rewrite-rule engine, its own epic
  series if the pedagogy market warrants; and cross-template macro-steps, which are
  E3.7 composition's presentation layer and reuse this machinery.
- **Depends:** E2.9, E2.13. — **Done:** the Mohr's-circle golden renders its three
  steps with the executor's actual intermediates; the trace round-trips through the
  artifact and reruns exact; a withheld answer carries no steps —
  `tests/test_trace.py`. *(Reference artifacts regenerated: `Answer.steps`.)*

### ✅ M2 gate — the v1 acceptance test
A fixed NL eval set produces correct, reproducible answers via the embedded inference
layer; a **Chisel-emitted candidate fails closed until its fixtures pass, then serves as
verified**; a third-party template loads with zero core changes; a fresh install runs on
**embedded StrataDB by default** (A-10, E2.8).

---

## 4. M3 — Hardening & edge

- **E3.1 — New domains** (chemistry / astronomy / thermodynamics / control / optimization) via
  the pipeline; permissive-solver discipline for optimization (engineering §2.2).
  *Smoke-sampled:* each planned domain ships 1–2 hand-authored declarative goldens
  (chemistry: `molarity.solution`, `molar_mass.from_sample`; astronomy:
  `escape_velocity.surface`, `schwarzschild.radius` — resolving `G` and `c` from
  `assay.constants`; thermodynamics: `heat.sensible` — `c_p` resolved per material;
  control: `rc.time_constant`, `spring_mass.natural_frequency`; optimization:
  `max_rectangle_area.fixed_perimeter`, a closed form), each with a worked problem in
  the eval. *Depth* — populated content via the pipeline (E2.3) and procedural cases
  via `kind: solver` (E3.5) — remains this epic's open remainder.
- **E3.2 — Web UI** — the glass box (UX §4, open question §9): IR / method / provenance
  drill-down without becoming a form. *Shipped (minimal):* one self-contained page
  (`assay/api/index.html` — inline CSS/JS, no external assets, no build step) served at
  `/` by the API app: concise answer + verified badge first; the IR and full artifact
  one `<details>` click deep (data, not a form — the §9.5 answer); every honest state
  rendered as itself (missing inputs show *only* what's missing and resume via
  `/v1/run`; the ambiguity fork is pick-buttons; out-of-scope refuses); artifact
  download; NL→IR attribution with the re-ask caveat. Richer UI (editing the IR in
  place, figures) remains open.
- **E3.3 — Packaging polish** — cross-platform wheels, the model-fetch UX (engineering §8),
  footprint budget in the README, the **Windows sandbox** for `kind: solver` (engineering §11 Q4).
  *Delivered:* a CI **wheel job** builds and smokes the pure-Python wheel (with its
  package data — goldens, tables, the web page) from a clean venv on all three
  platforms, uploading the dists; **`assay model fetch/list`** — explicit,
  sha256-gated fetch to the user data dir (fails closed, idempotent; `urllib` scoped
  to the fetcher by the gate test, and the engine provably never imports it), with
  `--llm` resolving explicit → `$ASSAY_LLM_MODEL` → single cached model; the
  **footprint budget** measured into the README (base ~304 MB; +api 3, +stratadb 11,
  +llm 55; model separate); the **sandbox position** written
  ([`sandboxing.md`](sandboxing.md)) — the boundary activates with model-influenced
  execution (E3.7), POSIX rlimits + Windows Job Objects when it does.
- **E3.4 — Reproducibility hardening** — the floating-point story (NFR-2): exact same-platform,
  within-tolerance cross-platform, reported honestly.
  *Delivered:* **committed reference artifacts** (`tests/reference_artifacts/`, ten of
  them spanning symbolic/formula/DAG/all four solver bindings, generated by
  `scripts/make_reference_artifacts.py` — regenerated only deliberately, never to
  silence a red) rerun by CI on Linux/macOS/Windows: `failed` is never acceptable, a
  matching environment must rerun `exact`, and symbolic results are exact everywhere;
  **drift diagnostics** — every non-exact rerun names its environment drift
  (versions/python/platform), and a non-exact rerun with *no* drift says "this may be
  a reproducibility bug" out loud; **hash-seed independence proven** (subprocesses
  with contradictory `PYTHONHASHSEED`s write byte-identical artifacts); rebuilding a
  reference from its own IR is byte-identical on a matching environment; CI pins
  `OPENBLAS_NUM_THREADS=1` (NFR-1 hygiene). The reference set doubles as the
  version-drift tripwire: a dependency bump that shifts any value shows up as a named
  `within-tolerance` — or a red — on the next push.
- **E3.5 — `kind: solver` templates** — the Assay-authored code bindings for procedural cases
  (ODE / optimization / root-finding), sandboxed (PRD §7.1, engineering §7).
  *Delivered:* the curated registry (`assay.execute.solvers` — a binding is a **name in
  a whitelist**, never an import path; problems pass the same safe-parse gate; SymPy
  `subs` evaluation, no eval/lambdify; SciPy joins the runtime nucleus) with four
  bindings, each carrying a built-in independent check: `root_find.brentq`
  (substitution — the quintic the symbolic path withheld now answers verified),
  `integrate.quad` (cross-method Simpson), `minimize.scalar_bounded` (local
  optimality), `ode.solve_ivp` (RK45 cross-checked by DOP853); four solver goldens +
  worked problems (25 goldens, eval 25/25). *Open remainder:* the OS-level sandbox for
  model-influenced execution (E3.3/engineering §11 Q4) — today's bindings are
  Assay-authored trusted code, so the gate is the grammar, not a process boundary.
- **E3.6 — Optional external integrations** — deeper Chisel, Verity for any real tools Assay
  drives, Petra on the inference model; versioned online data sources (PRD §19).
- **E3.7 — Composition & leaf synthesis** *(designed, not scheduled —
  [`composition-and-synthesis.md`](composition-and-synthesis.md))*: multi-step answers
  on a strict precedence ladder — curated template → model-proposed **composition of
  curated templates** (`kind: composition`: the model wires, never writes math; every
  edge dimension-checked, every node verified) → minimally-synthesized **leaf**
  formulas (⚠ candidate forever until *independent* fixtures promote them) → refuse.
  The invariant: **nothing synthesized ever overrides or shadows a curated template**.
  Depends: E2.9 (step semantics), E3.5 (solver kind). Build the simple ones first.

---

## 5. Traceability (epic → requirement)

| A-req | Delivered by |
|---|---|
| A-1 IR is the contract | E0.3, E1.2 |
| A-2 no fabricated facts | E1.3 |
| A-3 template owns method | E1.1, E1.5 |
| A-4 deterministic mode | E1.7 |
| A-5 inference via embedded layer | E2.1 |
| A-6 verification stage | E1.4 |
| A-7 reproducible artifact | E1.6 |
| A-8 missing-input handling | E1.3, E2.1 |
| A-9 nucleus + plugin SDK | E1.1, E2.4 |
| A-10 built on Strata (embedded) | E0.2 (the seam), E2.8 (the adoption) |
| A-11 provenance & versioning | E0.2, E0.3, E1.6 |
| A-12 explainable failure | E2.1 |
| A-13 declarative + generic executor | E1.2 |
| A-14 candidate/verified tier | E2.2 |
| A-15 shared template-validator | E1.1 |

---

## 6. Risks & sequencing notes

- **StrataDB embedded surface (E0.2 → E2.8)** is the one foundation that can slip; the
  store *interface* + in-memory default keep M1 unblocked while the adapter (the optional
  `assay[stratadb]` extra, exercised in CI) proves the seam against the real SDK. The flip
  to StrataDB-by-default (E2.8) waits on StrataDB being production-ready — and the
  hard-requirement (A-10) means Assay drives that maturation rather than waiting on it.
- **The Strata inference layer (E2.1)** gates M2's NL surface; M1 (v0, deterministic) is
  fully independent of it, so real value ships before the inference layer is required.
- **Chisel (E2.3)** is cross-repo; the alignment doc's §14 questions should be answered
  before E2.3 starts, but E2.2 (the gate) and E2.4 (SDK) don't wait on Chisel.
- **Correctness is a standing gate, not a milestone** — every template ships fixtures, every
  milestone stays green on the platform matrix, and the license gate runs on every commit.
