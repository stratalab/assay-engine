# Assay — Product Requirements Document

*The concrete "what we build" to the vision's "why." Companion to
[`assay-vision.md`](assay-vision.md) (thesis, the one claim, positioning). This
document pins the architecture, the intermediate representation, the requirements, the
scope, and a phased roadmap with done-criteria — enough to start v0 from.*

---

## 1. Summary

Assay is an **open-source computational answer engine**: ask a scientific or
mathematical question — in natural language, on the CLI, or as a structured object —
and receive an **executable, inspectable, reproducible** answer, computed by
established open-source libraries (SymPy, SciPy, Pint, …) rather than by a language
model's memory. Assay builds no new solver. Its product is the **curated layer between
language and computation** — a structured intermediate representation (IR), a library
of hand-authored task templates, and one hard guarantee: **facts come from trusted
sources, never from the model.**

The build order embodies the thesis: **the deterministic spine ships first, without any
model** (structured input → IR → execute → reproducible answer), and the language-model
front-end — served by the embedded Strata inference layer — is added on top as an optional
producer of IRs. A user who
never touches an LLM still gets a programmable computational engine; a user who brings
any model gets natural language on top.

---

## 2. Users and use cases

| User | How they use Assay | What they need |
|---|---|---|
| **Developer** | The structured Python API / CLI, no model — `assay solve "x^2+3x-4=0"` | Deterministic, fast, scriptable, reproducible results |
| **Scientist / engineer** | A natural-language question through a model of their choosing | Correct answer + method + assumptions + a rerunnable artifact |
| **Agent / tool integrator** | Assay as a *tool* an LLM agent calls | A structured, verifiable computation the agent can trust and cite |
| **Ecosystem (optional)** | Lithos as the reference model provider; Chisel curating templates; Strata caching | Clean interfaces — never a hard dependency |

The through-line: every user gets a result they can **open, verify, and rerun** — the
property that separates Assay from both a chatbot (opaque, unverified) and Wolfram
Alpha (correct but closed and unscriptable).

---

## 3. The problem (in brief)

Two failure modes bracket the status quo; Assay is defined by refusing both (full
treatment in the vision, §3):

- **The chatbot** — fluent, confident, and unfalsifiable: an invented formula, a
  hallucinated constant, no execution, no units check, no provenance, no reproduction.
- **The black box** — Wolfram computes correctly but is closed, unscriptable at the
  reasoning layer, and gives an answer you cannot open, rerun, or port to your own model
  or data.

The open computational libraries exist but are fragmented; nothing unifies them into a
curated, callable, honest engine. That engine is the gap.

---

## 4. Principles

Two cornerstones, and the commitments they force:

1. **The model never sources a fact.** The model may interpret and plan; it may never
   supply a number. Every constant, material property, and formula is resolved from a
   trusted library/dataset, or declared a missing input. → forces the *resolver* (§8)
   and the *provenance record* on every resolved fact.
2. **The intermediate representation is the product.** Between language and computation
   there is an inspectable, executable structured object — never an LLM emitting Python
   and hoping. → forces the *IR* (§6) as the single execution contract, and the
   *task-template library* (§7) as the curated content behind it.

Derived commitments: **template-owns-method** (the template defines the formula and
plan; the model only fills task + inputs); **verify before you trust** (§9);
**reproducible by construction** (§10); **bounded, not inferior** (§15);
**self-contained on embedded Strata** (§19).

---

## 5. Architecture

```
   producers                          the spine                         result
 ┌───────────────┐
 │ NL (via model)│──┐
 ├───────────────┤  │   ┌─────┐   ┌──────────┐   ┌─────────┐   ┌────────┐   ┌──────────────┐
 │ CLI / struct. │──┼──▶│ IR  │──▶│ validate │──▶│ resolve │──▶│ execute│──▶│ verify + emit│
 ├───────────────┤  │   └─────┘   └──────────┘   └─────────┘   └────────┘   └──────────────┘
 │ hand-authored │──┘   (against the task template's contract)                 result + method
 └───────────────┘                                                             + interpretation
                                                                               + artifact
```

The **IR is the entry point and the single execution contract.** Three producers can
create one — a model from natural language, the CLI/structured API, or a hand-authored
file — and exactly one consumer executes it. Natural language is *a* producer, never
the mandatory front door. Every stage after the IR is deterministic.

---

## 6. The Intermediate Representation

The IR is a structured, inspectable, executable object. A question —

> *"The maximum deflection of a simply supported steel beam under a 5 kN central load over 2 m."*

— compiles to:

```yaml
assay_version: "0.x"
ir_version: 1
query: "max deflection of a simply supported steel beam, 5 kN central load, 2 m span"
domain: structural_mechanics
task: beam_deflection                       # a template id (§7)
setup:                                       # the problem configuration
  member: simply_supported_beam
  loading: { type: point_load, position: center }
inputs:
  P: { value: 5000, unit: N }                 # load
  L: { value: 2,    unit: m }                 # span
missing_inputs: [ E, I ]                      # required, not yet supplied
resolved:                                     # filled by the resolver — never by the model
  E:
    value: 200e9
    unit: Pa
    source: { library: "assay.materials", key: "steel.structural.E", version: "0.x" }
assumptions: [ euler_bernoulli, small_deflection, linear_elastic ]
execution_plan: [ resolve_material_property, validate_units, evaluate_formula, generate_plot ]
outputs: [ max_deflection ]
```

**Naming fix vs the vision.** The vision sketch used `model:` for the problem
configuration; the PRD renames it to **`setup`** to avoid collision with the
language-*model* layer. The two are unrelated and must not share a key.

**Input naming — the template owns the vocabulary.** The keys of `inputs`,
`missing_inputs`, and `resolved` are the **template's declared input names** (§7.1:
`P`, `L`, `E`, `I`). Every producer — model, CLI, hand-authored file — binds quantities
to those names, and the executor rejects keys the template does not declare. Human-facing
renderings may expand a name ("second_moment_of_area (I)"), but the IR carries the
template's canonical symbols.

**Properties the IR guarantees, structurally:** inspectability (a human reads exactly
what will run), reproducibility (§10), model-portability (any producer, same execution),
deterministic execution, caching (keyed by IR content hash), and verifiability (§9).

**Provenance & versioning (requirement A-11).** Every IR carries `assay_version`,
`ir_version`, a **content hash**, and — for every entry in `resolved` — the exact
`source` (library + key + version). The answer is therefore auditable to the fact.

**Content-hash scope.** The hash covers the *computational* fields only: it is canonical
JSON (sorted keys) → sha256 over the IR **excluding `assay_version`** (software
provenance, not content) **and `query`** (NL phrasing). Two phrasings of the same
computation — or the same IR run under two Assay versions — hash identically; any change
to task, setup, inputs, resolved facts, assumptions, plan, or outputs changes the hash.
Because the hash deliberately does not identify the software that ran it, **cache entries
key on the content hash *plus* the pinned execution versions**, never the hash alone.

---

## 7. The task-template library (the moat)

The IR *schema* is a day's work; the **curated catalog of task templates behind it is
the multi-year, compounding moat.** A template is a hand-authored unit of domain
knowledge. **The template owns the method and the plan; the model only maps a question
to `task` + `inputs` + `missing_inputs`.** That bounds the model to the one job it is
good at and keeps its failure surface small.

**Template contract** — every template declares:

- `id`, `domain`, and a human description.
- **input schema** — required and optional inputs, each with an expected *dimension*
  (so units are checked, not assumed).
- **resolution hints** — an input may name its trusted source (a `resolve` ref: a
  curated table + a key pattern filled from `setup`); the resolver honors it (§8) and
  the model never supplies the value. An input without a hint can only be user-supplied.
- **method** — `kind: formula` (a symbolic expression Assay's generic executor
  evaluates — e.g. `P*L**3/(48*E*I)` — the declarative, Chisel-populatable form),
  `kind: symbolic` (a *curated symbolic operation* the executor owns — solve,
  integrate — where the problem expression arrives in `setup`, passes the same safe
  parse, and verification is built in: substitution/derivative), or `kind: solver`
  (an Assay-authored code binding for genuinely procedural cases: ODE solve,
  optimization, root-finding). The template owns it; the model never writes it.
- **assumptions** — the modeling assumptions, surfaced in every answer.
- **execution_plan** — the ordered deterministic steps.
- **verification hooks** — dimensional expectation, plausibility bounds, and a
  cross-method check where feasible (§9).
- **output schema** — the target quantities and their dimensions.
- **fixtures** — worked examples with known answers. *A template without passing
  fixtures does not ship* (the golden-fixture discipline).

**Plugin SDK.** Templates are plugins: discoverable, independently versioned, grouped
into domains. A new domain is a set of templates authored against the SDK — **never a
change to the core.** Third parties can add domains without touching the engine.

### 7.1 Declarative templates and the generic executor

Most templates are **fully declarative** — a symbolic formula plus dimensioned inputs,
outputs, assumptions, and fixtures — and Assay runs *any* of them through **one generic
executor**: parse the formula (SymPy), bind the resolved inputs with units (Pint),
evaluate, check the result's dimension against the declared output, and run the fixtures.
**Adding a declarative domain is adding data, not code.** Only `kind: solver` templates
carry Assay-authored code; the declarative path covers the physics/engineering
formula-family bulk. The beam of §6, as a declarative template:

```yaml
id: beam_deflection.simply_supported.center_point
domain: structural_mechanics
inputs:
  - { name: P, dimension: force }
  - { name: L, dimension: length }
  - { name: E, dimension: pressure,         # resolved from materials, never the model
      resolve: { library: assay.materials, key: "{material}.E" } }
  - { name: I, dimension: length**4 }
method: { kind: formula, expr: "P * L**3 / (48 * E * I)" }
output: { name: max_deflection, dimension: length }
assumptions: [euler_bernoulli, small_deflection, linear_elastic]
fixtures:
  - inputs: { P: [5000, N], L: [2, m], E: [200e9, Pa], I: [8.33e-6, "m**4"] }
    expect: { max_deflection: [5.0e-4, m] }   # 0.50 mm
    tol: 1e-6
provenance: { source: "<book / paper / question-paper id>", license_tier: lawful, status: candidate }
```

### 7.2 The Chisel → Assay content pipeline

Domain breadth is **populated, not hand-written** one template at a time.
[Chisel](https://github.com/stratalab/chisel) processes books, papers, Wikipedia, and
question papers into **candidate declarative templates** against this exact schema — its
formula registry becomes the `method`, and question papers become the **fixtures**. The
seam is the same shape as the Lithos↔Chisel one (`tir_validate`): a shared, standalone
**`validate_template()`** that both repos import (Assay owns it), so Chisel cannot emit a
template Assay would reject.

The trust boundary lives in the schema as a trust **`status`** — a separate field from
the license axis, `provenance.license_tier` (two tiers, never conflated —
chisel-alignment §7):

- Chisel emits `status: candidate`.
- Assay promotes candidate → **`verified`** only when the template's **fixtures pass
  under the generic executor** — optionally cross-checked against an independently
  *derived* form, where a disagreement between the extracted and derived formula is a
  curation bug caught before it ships. Assay serves only `verified` templates by default.

So **extraction never becomes authority silently**: every template carries its own
correctness proof, and Chisel floods in candidates while the fixture gate holds the
floor. Which Chisel input feeds which part: *question papers → fixtures* (cleanest),
*textbooks → methods + assumptions*, *papers → advanced methods + constants*, *Wikipedia
→ resolver constants/definitions* (never a canonical formula). Chisel is a **build-time**
accelerator that expands Assay's baked-in content; it is never a runtime dependency
(A-10).

---

## 8. The no-fabricated-facts guarantee

The single most important reliability mechanism. **The model may never supply a numeric
fact.** Constants, material properties, and formulas are handled by the **resolver**:

- Given a required input the IR left in `missing_inputs`, the resolver attempts to
  resolve it from a **trusted source** — a curated Assay table (`assay.materials`,
  `assay.constants`) or a nucleus library (`scipy.constants`, and later CoolProp,
  `periodictable`, …) — and records the exact `source` + `version` + `value` in
  `resolved`.
- If it **cannot** be resolved, the input stays missing, and the engine either
  **asks** the user (interactive mode) or **fails with a clear "missing required
  input"** (batch/deterministic mode). It **never** lets the model fill it, and never
  silently defaults it.

This is the same principle as Lithos masking tool-results from the training loss and
Verity's determinism: **facts come from the world, judgment comes from the model.** Get
it wrong and Assay is a confident hallucination machine wearing a SciPy logo.

---

## 9. Deterministic verification

Verification is a first-class pipeline stage, not a formatting step. Assay checks its
own work in ways a chatbot structurally cannot:

- **Dimensional consistency** — the computed result's dimension must equal the
  template's declared output dimension (Pint). A wrong formula usually fails here.
- **Plausibility bounds** — template-declared ranges reject the absurd (3 km of beam
  deflection) before it is reported.
- **Cross-method agreement** — where feasible, compute symbolically *and* numerically
  and compare within tolerance.

A failed check **withholds or flags** the answer with an explanation; it never silently
returns a result that failed verification.

---

## 10. Output and reproducibility

Every answer has four parts:

1. **Result** — the value, with units. *(Maximum deflection: 0.50 mm)*
2. **Interpretation** — what was understood and assumed.
3. **Method** — the formula and assumptions. *(δ = PL³/48EI; simply supported, central point load, linear-elastic, small deflection)*
4. **Executable artifact** — the IR + resolved facts + pinned library versions,
   rerunnable: `assay run result.json`.

**Reproducibility is the deliverable.** `assay run <artifact>` on the same pinned
versions reproduces the result bit-for-bit — the clearest line between Assay and both a
chatbot and a black box.

### 10.1 Rendering — verified data, not verified pictures

Plots, graphs, and diagrams are **renderings of verified data, never answers.** Assay
separates **compute** (verified — the result) from **render** (a faithful visualization of
that result). The model never draws a curve; it selects a *view*, and every mark traces to
a computed, verified quantity.

- **The IR carries an optional `render` directive** — a view spec (plot type, range, what
  to mark) — validated and executed *after* compute + verify. For a compound query like
  *"plot and solve x² − 5x + 6 = 0"*, the IR holds one computation (solve → roots x=2, x=3)
  plus one render directive (sample y = x²−5x+6 over a root-derived range, mark the verified
  roots and vertex).
- **Generic render primitives** (function-plot, scatter, geometry-diagram) are built-in and
  domain-agnostic; **domain diagrams** (a Mohr's circle, a free-body diagram) are
  template-provided renderings. Both render only verified data.
- **Verification applies to the data, not the pixels.** You cannot dimension-check an image
  and you do not need to: trust flows *verified data → deterministic renderer → honest
  figure*. Fixtures assert the figure's *features from the data* ("roots marked at x=2, x=3"),
  never a pixel comparison.
- **Figures are deterministic** (fixed backend, no timestamps, SVG preferred) so the same
  data + view spec render identically; regressions are diffable.
- **Geometry splits the same way:** analytic geometry is *compute* (SymPy `geometry`, exact —
  intersections, areas, circles); a diagram is a *render* of those computed objects.

The figure is an artifact attached to the answer and **labelled a rendering** — the *result*
remains the verified value. (Matplotlib and SymPy `geometry` are both permissive — no license
concern.)

---

## 11. Inference via the embedded Strata inference layer

Assay does not call model providers directly. NL→IR inference runs through **Strata's
inference layer**, which ships **embedded** (§19) — a complete solution: **llama.cpp built
in** (serves a local Lithos model on-device by default, so there is nothing external to
deploy or key) **plus optional routing to OpenAI/Anthropic/Google**. Assay depends on that
one layer; the provider flexibility lives in the layer, not in Assay. **Not in the path of
the deterministic spine.**

Routing to a hosted provider is *runtime inference, not training* (no teacher-doctrine
conflict), and the no-fabricated-facts rule (§8) **contains any model** — even a closed
one only produces a *candidate IR*, validated before execution, and never supplies a fact.

**Propose → validate → execute.** The layer's only job is to produce a *candidate IR*
(task selection + input extraction + missing-flags) from natural language. That candidate
is **validated against the task template's contract before anything runs** — a malformed
or hallucinated IR is rejected, not executed. The model influences *which template and
inputs*, never the *method*, the *facts*, or the *execution*.

Assay's single inference dependency is that one embedded layer; model selection and
serving are its concern, not Assay's — swap the *served model* (Lithos by default), not
the layer. **Deterministic mode** bypasses inference entirely (§12). The IR is an ideal
tool-use training target, which is why Lithos is the natural model the layer serves.

---

## 12. Surfaces

- **Python API** — `assay.solve(...)`, `assay.run(ir)`, the resolver and template
  registry. The foundational surface. *(v0)*
- **CLI (deterministic mode)** — no model in the path: *(v0)*
  ```
  assay solve     "x^2 + 3x - 4 = 0"
  assay integrate "sin(x)^2"
  assay units     "30 psi to kPa"
  assay run       result.json
  ```
- **NL CLI / one-shot** — `assay ask "..."` routed through a configured provider. *(v1)*
- **HTTP API** — the engine behind a service boundary. *(v1)*
- **Web UI** — inspectable answers in a browser. *(later)*

---

## 13. Functional requirements

- **A-1 — IR is the sole execution contract.** Nothing executes except a *validated*
  IR. All producers converge on it; one executor consumes it.
- **A-2 — No fabricated facts.** Every constant/property/formula is resolved from a
  trusted source or declared missing; the model never supplies a numeric fact; every
  resolved fact records its source + version (§8, §6).
- **A-3 — Template owns method.** Each template declares inputs, method, assumptions,
  plan, verification, outputs, and ships passing fixtures; the model supplies only
  `task` + `inputs` + `missing_inputs` (§7).
- **A-4 — Deterministic mode.** A structured/CLI path executes with no model in the
  loop, deterministically.
- **A-5 — Inference via the embedded Strata inference layer.** NL→IR runs through
  Strata's embedded inference layer (serving a local Lithos model by default), which emits
  a candidate IR validated before execution; Assay calls no provider directly (§11).
- **A-6 — Verification stage.** Dimensional + bounds + (where feasible) cross-method;
  a failed check withholds/flags the answer with a reason (§9).
- **A-7 — Reproducible artifact.** Four-part output; `assay run <artifact>` reproduces
  the result on pinned library versions (§10).
- **A-8 — Missing-input handling.** Under-specified problems are flagged and either
  asked or failed-clear; never fabricated, never silently defaulted.
- **A-9 — Bounded nucleus + plugin SDK.** Core libraries fixed to the v1 nucleus
  (§15); new domains arrive as plugins, not core edits (§7).
- **A-10 — Built on Strata, embedded (self-contained).** StrataDB is the store and the
  Strata inference layer is the NL→IR path — both **required and bundled**, so there is
  nothing external to deploy; `pip install` and it works. Chisel/Verity are optional
  external accelerators (§19).
- **A-11 — Provenance & versioning.** Each IR carries `assay_version`, `ir_version`, a
  content hash, and per-fact sources; results are auditable and cacheable.
- **A-12 — Explainable failure.** When Assay cannot map a query, resolve a fact, select
  a solver, or verify a result, it states *why* — it never guesses.
- **A-13 — Declarative templates + generic executor.** A `kind: formula` template runs
  through one domain-agnostic executor (symbolic eval + unit check + fixtures); a new
  declarative domain adds data, not code (§7.1).
- **A-14 — Candidate/verified trust status.** Every template carries a trust `status`
  (`candidate` → `verified`), distinct from its `provenance.license_tier`; only
  `verified` templates serve by default; a `candidate` is promoted only when its
  fixtures pass under the executor (§7.2).
- **A-15 — Shared template-validator seam.** A standalone `validate_template()` (Assay
  owns; Chisel imports) is the Chisel→Assay contract — Chisel cannot emit a template
  Assay would reject (§7.2).

---

## 14. Non-functional requirements

- **Determinism** — same IR + pinned versions → identical result; no wall-clock, no RNG
  in the compute path.
- **Reproducibility** — the artifact reruns; library versions are pinned in it.
- **Offline compute** — facts resolve from bundled/curated data; the compute path makes
  no network calls (online data sources are a later, explicitly-versioned option).
- **Safety** — any model-influenced or generated execution runs sandboxed; curated
  template code is trusted, arbitrary generated code is not.
- **Performance** — the deterministic path is fast enough for interactive CLI use.
- **Portability** — stable results across model providers (same NL → same IR/answer
  within tolerance).

---

## 15. Scope

**In scope (the product):** the IR + executor; the **declarative template schema +
shared `validate_template()` + generic executor**; the resolver + curated constant/property
tables; the task-template library + plugin SDK; deterministic verification; the
four-part reproducible output; **rendering — plots + analytic-geometry diagrams from
verified data (§10.1)**; the embedded Strata inference layer; the CLI/API surfaces.

**The v1 nucleus (bounded, not inferior).** A *small, reliable* library set —
**SymPy** (symbolic), **NumPy/SciPy** (numeric + `scipy.constants`), **mpmath**
(high precision), **Pint** (units), **Matplotlib** (plots), **statsmodels**
(statistics) — covering: arithmetic & algebra, equation solving, calculus, unit
conversion, numerical methods, statistics, plotting, and the common physics/engineering
formula families.

**Explicitly out of scope:** a new CAS or symbolic engine; a generic coding agent; an
LLM wrapper around arbitrary Python; a giant scientific database (Assay *resolves*
facts, it is not the source); a Wolfram Language clone; a database or model *server* the
user must deploy (Strata ships embedded); a hard dependency on Lithos, Chisel, or Verity
as *external* services; geometric *proofs* and general spatial reasoning (a different product).

**Later (not v1):** chemistry (RDKit), astronomy (Astropy), thermodynamics (CoolProp),
control (`control`), optimization (CVXPY), graphs (NetworkX), differential equations,
materials data, deeper domain solvers, **3D / interactive plots**, the web UI, and online
(versioned) data sources.

---

## 16. Roadmap

| Phase | What | Done-criteria |
|---|---|---|
| **v0 — Prove the spine (no model)** | IR schema + validator + executor **on the embedded-store seam** (StrataDB adapter + in-memory default; adoption staged, §19); the **declarative template schema + `validate_template()` + generic executor**; the resolver + a small curated `materials`/`constants` table; **3 declarative templates** — `solve_equation`, `integrate`, `beam_deflection`; verification (dimensional + one cross-method); four-part output + `assay run`; **rendering from verified data + `assay plot` (§10.1)**; deterministic CLI | Each template answers a worked example **correctly, reproducibly, with per-fact provenance**; `beam_deflection` correctly declares E and I missing, then **resolves E for steel from the curated table (never fabricated)**; all fixtures green; a figure renders **byte-stably from verified data**; **no model anywhere in the loop** |
| **v1 — The product** | The embedded Strata inference layer wired to NL→IR (Lithos served by default); NL→IR with pre-execution validation; broaden to the nucleus domains (~a dozen templates); the plugin SDK + one externally-authored template; **the Chisel candidate pipeline + the candidate→verified fixture gate**; HTTP API | A fixed NL eval set produces **correct, reproducible answers via the embedded inference layer**, same IR/answer across reruns within tolerance; a third-party template loads with zero core changes; a **Chisel-emitted candidate fails closed until its fixtures pass, then serves as verified** |
| **later** | New domains (chemistry/astro/thermo/control/optimization); web UI; optional Strata/Lithos/Chisel/Verity integrations; versioned online data | Per-domain worked-problem sets pass; integrations remain removable |

---

## 17. Success criteria

- **Correctness** — pass rate on a held-out worked-problem set, per domain.
- **Reproducibility** — fraction of artifacts that rerun to an identical result (target: 100%).
- **Zero-fabrication audit** — every reported numeric fact traces to a resolver source (target: 100%).
- **Verification catch rate** — fraction of injected wrong formulas/units caught by the verify stage.
- **Cross-provider agreement** — same NL question → same IR/answer across providers, within tolerance.
- **Missing-input calibration** — under-specified problems flagged (asked/failed), not silently defaulted.

---

## 18. The hard parts (the arithmetic is the easy part)

- Mapping ambiguous language to a *precise* task + inputs.
- Deciding when an input is missing and must be asked for vs resolved.
- **Preventing fabricated constants and material properties** (the §8 spine).
- Selecting the correct template/solver.
- Sandboxing any generated execution.
- Validating units and domain applicability.
- Explaining failures clearly instead of guessing.
- Producing stable results across different model providers.

---

## 19. The Strata stack — embedded, not deployed

Assay is **built on Strata, which ships embedded** — the user installs nothing extra and
manages no services (the SQLite pattern). Two parts are bundled and **required**:

- **StrataDB** (embedded, like SQLite) — the store for artifacts, cache, and IR/result
  lineage. Not a different database, and not a server you stand up.
- **Strata inference layer** (embedded) — the NL→IR inference path; a complete solution:
  llama.cpp built in (serves a local Lithos model on-device by default) plus optional
  routing to OpenAI/Anthropic/Google. Assay calls no provider directly — the layer does.

Optional **external** accelerators — genuinely optional, never bundled:

- **Lithos** — the model the inference layer serves (swap the served model, not the layer).
- **Chisel** — the **breadth engine**: populates candidate declarative templates +
  fixtures against the shared schema (§7.2); build-time only, never a runtime dependency.
- **Verity** — action-layer verification of any real tools Assay is wired to drive.
- **Petra** — interpretability research on the model the layer serves.

**The bar: a stranger `pip install`s Assay and it just works — Strata is inside, nothing
to deploy.** Self-contained *and* built on Strata: both, because Strata is embedded.

**Adoption is staged** (implementation plan E0.2 → E2.8): the store ships behind a seam
first — the StrataDB adapter as the optional `assay[stratadb]` extra, exercised in CI,
with an in-memory default — and StrataDB flips to the bundled, required default once its
embedded surface is production-ready. A-10 states the end state; the seam is how we get
there without gating the spine on StrataDB's maturity.

---

## 20. Open questions

1. **IR authoring format** — YAML for humans, JSON for machines, or one canonical form
   with both renderings? (Content hash must be stable across renderings.)
2. **Resolver data governance** — how curated tables are sourced, versioned, and
   cross-checked; the bar for admitting a value; license of each data source.
3. **Cross-method verification coverage** — which task classes get a genuine independent
   second method vs only dimensional/bounds checks.
4. **Solver/template selection** — when multiple templates could match, how the choice is
   made and disclosed (and how ambiguity is surfaced rather than guessed).
5. **Sandbox boundary** — how much execution is fixed template code (trusted) vs
   model-influenced (must be sandboxed), and where that line sits per phase.
6. **Served-model capability floor** — the minimum capability the inference layer's model
   needs for reliable NL→IR, and how gracefully Assay degrades below it (fall back to
   deterministic mode?).
7. **Embedded StrataDB maturity** — Assay ships and runs on embedded StrataDB (A-10), so
   release is gated on StrataDB being a stable embedded store — and Assay becomes a
   forcing function for that maturation. What minimum StrataDB surface does v0 need?
8. **Embedded inference footprint** — how the Strata inference layer + a local Lithos
   model are bundled or fetched (install size, first-run download) so "pip install and go"
   holds on modest hardware.
