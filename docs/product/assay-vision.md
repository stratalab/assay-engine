# Assay — Thesis, Strategy & Plan

*The north-star document: what Assay is, the one claim it rests on, the architecture
that follows from it, the bounded scope, and what it deliberately is not. Detailed
specs will live alongside this — the IR schema, the plugin SDK, the inference-layer
binding, and the execution plan. This doc is the synthesis they hang from.*

---

## 1. The thesis, in one paragraph

Ask a scientific or mathematical question in natural language and get back an
**executable, inspectable, reproducible** answer — computed by established open-source
libraries, not by a language model's memory. Wolfram Alpha proved the value of a
natural-language computational engine, but it bundles two things — language
understanding and a curated computational core — into one **closed, API-gated**
product. The open pieces of the computational half already exist (SymPy, SciPy, Pint,
mpmath, and their domain cousins), but they are **fragmented**: there is no unified,
curated, callable engine that ties them together with a self-consistent interface,
trusted constants, and a reproducible answer. **Assay is that engine** — an
orchestrator of trusted computational libraries whose language-model inference runs
through the **Strata inference layer, shipped embedded** (§9). It builds no new solver and
no new symbolic engine; its value is the
**curated layer** between language and computation, and one hard guarantee: **facts
come from trusted sources, never from the model.**

---

## 2. The one claim everything rests on

A trustworthy scientific answer engine must satisfy two properties, and every design
decision follows from them:

- **(a) No fabricated facts.** The model may *interpret* and *plan*, but it must never
  *supply a number*. Every constant, material property, and formula is resolved from a
  trusted library or dataset, or it becomes a declared missing input — never filled
  from parametric memory. Violate this and you have built a confident hallucination
  machine wearing a scientific-library logo.
- **(b) A structured intermediate representation.** Between the language and the
  computation there must be an inspectable, executable structured object — not an LLM
  emitting Python snippets and hoping they run. The IR is what makes an answer
  reproducible, portable across models, cacheable, and verifiable.

Everything below is the consequence of taking these two seriously.

---

## 3. The problem Assay solves

Two failure modes bracket the status quo, and Assay is defined by refusing both:

- **The chatbot.** An LLM asked a physics question will confidently produce an answer,
  a plausible formula, and an invented constant — with no execution, no units check,
  no provenance, and no way to reproduce it. Fluent and unfalsifiable.
- **The black box.** Wolfram Alpha computes correctly but is closed, unscriptable at
  the reasoning layer, and gives you an answer you cannot open, rerun, or port to your
  own model or data.

Between them sits the thing that does not exist in open form: a computational answer
engine that is **open, reproducible, model-agnostic, and honest about what it does and
does not know.**

---

## 4. What Assay is

```
question (NL, CLI, or hand-authored)  →  Intermediate Representation  →  execute  →  answer + artifact
                                              ▲
                                    (a model is one way to
                                     produce the IR — not the
                                     only way, not required)
```

An **orchestrator for trusted computational libraries**, organized around an
inspectable IR. The IR is the entry point and the spine; language is *one producer* of
it, not the mandatory front door.

**The IR is the product.** A question like *"the maximum deflection of a simply
supported steel beam under a 5 kN central load over 2 m"* becomes a structured object,
not a code snippet:

```yaml
domain: structural_mechanics
task: beam_deflection
model: { type: simply_supported_beam, loading: { type: point_load, position: center } }
inputs:
  load:   { value: 5000, unit: N }
  length: { value: 2,    unit: m }
missing_inputs: [ elastic_modulus, second_moment_of_area ]   # asked or resolved — never invented
assumptions: [ euler_bernoulli, small_deflection, linear_elastic ]
execution_plan: [ resolve_material_property, validate_units, evaluate_formula, generate_plot ]
```

That object gives inspectability, reproducibility, model-portability, deterministic
execution, caching, and verification — for free, structurally.

**The moat is not the schema — it is the task library behind it.** Designing the IR is
a day's work. The value is the curated catalog of *task templates* — `beam_deflection`,
`projectile_motion`, `rc_transient`, `ideal_gas_state`, and hundreds more — each one
hand-authored with its canonical formula, its required inputs, its assumption set, and
its solver binding, by someone who knows the domain. **The template owns the method and
the plan; the model only maps language → task + inputs + missing-flags.** That bounds
the model to the one job it is good at and makes the failure surface small. The task
library is the multi-year, compounding defensibility.

---

## 5. Deterministic self-verification

Assay checks its own work in ways a chatbot structurally cannot, and this — with
reproducibility — is the honest line between it and both a chatbot and Wolfram:

- **Dimensional consistency** — a wrong formula usually fails on units (Pint enforces it).
- **Sanity bounds** — 3 km of beam deflection is rejected before it is reported.
- **Cross-method agreement** — solve symbolically *and* numerically, and compare.

Verification is a first-class stage of the pipeline, not a formatting afterthought.

---

## 6. Every answer has four parts

1. **Result** — the value, with units. *(Maximum deflection: 0.50 mm)*
2. **Interpretation** — what was assumed and understood.
3. **Method** — the formula and the assumptions. *(δ = PL³/48EI; simply supported, central point load, linear-elastic, small deflection)*
4. **Executable artifact** — the IR + library versions, rerunnable: `assay run result.json`.

Reproducibility is the deliverable, not a nicety.

**Plots and diagrams are renderings of verified data — never answers.** When a question
asks to *plot* or *draw* (e.g. "plot and solve x² − 5x + 6 = 0"), Assay computes and
verifies the underlying quantities first — the roots, the sampled curve, the marked
points — and *then* renders a figure from that verified data. The model never draws; it
selects a view. Every mark on an Assay figure traces to a computed, verified quantity, so
a picture inherits its trust from the data behind it, not from the pixels. The figure is
an attached artifact, labelled a rendering — the *result* is still the verified value.

---

## 7. Inference runs through the embedded Strata inference layer

Assay does not call model providers directly. Turning a question into a candidate IR runs
through **Strata's inference layer**, which ships **embedded** inside Assay (§9) — a
complete solution: **llama.cpp built in** (serving a local Lithos model on-device by
default, so there is nothing external to deploy or key) **plus optional routing to
OpenAI/Anthropic/Google**. Assay's only inference dependency is that one layer; model
selection and serving are its concern, not Assay's.

And a **deterministic mode** bypasses language entirely, so the engine is a programmable
computational interface with no model in the path at all:

```
assay solve     "x^2 + 3x - 4 = 0"
assay integrate "sin(x)^2"
assay units     "30 psi to kPa"
assay optimize  problem.yaml
```

The model produces only a *candidate IR*, validated before anything runs; it never
supplies a fact or a method. Inference is an accelerator of the front-end — the
deterministic spine stands on its own.

---

## 8. Bounded, not inferior

"Lite" means **narrow and complete**, not shallow and broad. A few domains supported
end-to-end beat hundreds supported superficially — depth is the defensibility.

- **Nucleus (v1)** — a *small, reliable* set of libraries: SymPy (symbolic),
  NumPy/SciPy (numeric + CODATA constants), mpmath (high precision), Pint (units),
  Matplotlib (plots), statsmodels (statistics). Covers: arithmetic & algebra, equation
  solving, calculus, unit conversion, numerical methods, statistics, plotting, and the
  common physics/engineering formula families.
- **Later** — chemistry (RDKit), astronomy (Astropy), thermodynamics (CoolProp),
  control systems (`control`), optimization (CVXPY), graphs (NetworkX), materials data,
  differential equations, and deeper domain solvers.

Resist the catalog. A tight nucleus that always works is worth more than a vast one
that is brittle.

**Growing the scope — Chisel is the breadth engine.** Bounded is the *starting* posture,
not the ceiling. A new domain is a set of **declarative task templates** — a formula, its
dimensioned inputs and outputs, its assumptions, and worked-example fixtures — and
[Chisel](https://github.com/stratalab/chisel) can populate these at scale from books,
papers, Wikipedia, and question papers. Crucially this grows breadth *without* lowering
the correctness floor: the same question papers that widen a domain also supply the
**known-answer fixtures that prove it correct**. Chisel emits **candidates**; a template
becomes **trusted** only once its fixtures pass — so extraction never silently becomes
authority. Any subject Chisel can process can become an Assay domain, each gated by its
own proof.

---

## 9. Self-contained: Strata ships embedded

Assay is **standalone in the sense that matters** — self-contained, nothing external to
deploy. `pip install assay` and it works. But it is **built on the Strata stack**, which
ships **embedded** inside it:

```
Assay (self-contained — one install, no external services)
├── the IR + task-template library
├── the generic executor + resolver
├── a plugin SDK for new domains
└── Strata, embedded ─┬── StrataDB        — the store (embedded, like SQLite): artifacts, cache, lineage
                      └── inference layer — NL→IR: llama.cpp built in (local Lithos default) + OpenAI/Anthropic/Google
```

The user never installs, deploys, or manages Strata — it is an embedded implementation
detail, the way SQLite lives inside a desktop app. That is what makes Assay *both* a
hard-Strata product (every install runs StrataDB and the Strata inference layer, so the
stack is dogfooded and hardened) *and* a self-contained one (nothing to stand up).

Optional **external** accelerators — genuinely optional, never bundled:

```
├── Lithos   — the model the inference layer serves (swap the served model, not the layer)
├── Verity   — action-layer verification of any real tools Assay drives
├── Chisel   — the breadth engine: populates declarative templates + fixtures at scale
└── Petra    — interpretability research on the model the layer serves
```

**A stranger `pip install`s Assay and it just works — Strata is inside, nothing to
deploy.** That is the bar.

*The earned fit:* because the IR is a clean, verifiable target, "produce a correct IR" is
a far better training objective than "write Python and hope" — so Lithos is the natural
model for the embedded inference layer to serve, and Assay is an ideal environment to
train a tool-using reasoner against.

---

## 10. What Assay is *not*

- **Not a new CAS or symbolic engine** — it orchestrates SymPy et al., it does not
  replace them.
- **Not a generic coding agent** — it maps questions to *curated* computations, not to
  arbitrary code.
- **Not an LLM wrapper around Python** — the IR and the no-fabricated-facts rule are
  precisely the difference.
- **Not a giant scientific database** — it *resolves* facts from trusted sources; it is
  not trying to be the source.
- **Not a Wolfram Language clone.**
- **Not a service you deploy** — Strata (StrataDB + the inference layer) ships *embedded*,
  so there is nothing external to stand up; Lithos/Chisel/Verity are optional external
  accelerators.

---

## 11. The hard parts (the arithmetic is the easy part)

The real engineering is not computation; it is the judgment around it:

- Mapping ambiguous language to a *precise* task + inputs.
- Deciding when an input is missing and must be asked for vs resolved.
- **Preventing fabricated constants and material properties** (the §2a spine).
- Selecting the correct solver / template.
- Sandboxing any generated execution.
- Validating units and domain applicability.
- Explaining failures clearly instead of guessing.
- Producing stable results across different model providers.

---

## 12. Positioning

> **An open-source, self-contained computational answer engine — trusted scientific
> libraries and the Strata stack, embedded.**

Or, concretely: *Wolfram-lite for the open-source AI era — natural-language
computation, reproducible execution, inspectable results.* A self-contained top-level
project with its own repository, identity, plugin ecosystem, CLI, API, and web
interface; the Strata stack (StrataDB + inference) ships **embedded** — one install,
nothing external to deploy — and Lithos/Chisel/Verity are optional external accelerators.

The two ideas carved on the cornerstone: **the task-template library is the moat**, and
**the model never sources a fact.**

---

*Assay — from **assay** (a trial; the analytical test that determines the precise
proportion of metal in an ore): a trusted, reproducible determination of the answer.*
