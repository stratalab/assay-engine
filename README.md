# Assay

**An open-source, self-contained computational answer engine — trusted scientific libraries and the Strata stack, embedded.**

*Wolfram-lite for the open-source AI era — natural-language computation, reproducible execution, inspectable results.*

Ask a scientific or mathematical question and get back an **executable, inspectable,
reproducible** answer — computed by established open-source libraries (SymPy, SciPy,
Pint, …), not by a language model's memory. Assay builds no new solver; its value is
the curated layer between language and computation, and one hard guarantee: **facts
come from trusted sources, never from the model.**

## The two cornerstones

- **The model never sources a fact.** It may interpret and plan, but every constant,
  property, and formula is resolved from a trusted library — or declared a missing
  input. Never filled from parametric memory.
- **The intermediate representation is the product.** A question becomes a structured,
  inspectable, executable object — and the curated library of *task templates* behind
  it is the moat.

## Status

**M1 — the deterministic spine (v0) — complete**, gate passed
(`tests/test_m1_gate.py`, green on the Linux/macOS/Windows matrix): IR → resolve
(curated facts, never fabricated) → execute (safe parse, unit-bound) → verify
(withhold-with-reason) → four-part answer → reproducible artifact, with a working CLI
(`assay solve / integrate / plot / units / run / show`) and deterministic figures —
**no model anywhere in the deterministic loop**.

**M2 — the product — in progress.** E2.1 shipped: `assay ask` (natural language)
through the **inference seam** — propose → validate → execute, honest states
first-class (missing-input ask / fail-clear, ambiguity fork, out-of-scope refusal,
`--emit-ir` for correct-by-editing). Deterministic rule-based backend by default;
a local llama.cpp model via `pip install assay[llm]` + `assay ask --llm model.gguf`
(the interim binding the embedded Strata inference layer replaces behind the same
seam). E2.2 shipped: the **candidate → verified fixture gate**
(`assay.templates.promote`) — a template serves only after its own fixtures pass under
the generic executor; failing candidates stay quarantined. E2.4 shipped: the **plugin
SDK** — a package declares an `assay.templates` entry point and its templates are
discovered, validated, gated, and served with zero core edits (`assay domains` shows
everything covered). E2.5 shipped: **nucleus breadth** — 12 golden templates across 10
domains (algebra, calculus, kinematics, dynamics, oscillation, thermodynamics,
electricity, waves, materials, structural mechanics), each proven by a worked-problem
eval through the full NL pipeline at 100% pass rate (`tests/test_nucleus.py`). E2.6
shipped: the **HTTP API** (`pip install assay[api]`, `assay serve`) — the answer
object over the wire with the honest states as first-class outcomes, plus the full
artifact in every response so `POST /v1/rerun` reproduces statelessly. E2.7 shipped:
**IR-model attribution** — every artifact names the NL→IR producer
(`ir.produced_by`), excluded from the content hash (provenance, not content), with a
re-asking caveat for model-produced IRs. E3.1 smoke-sampled: chemistry, astronomy,
control, and optimization join the nucleus. E2.9 shipped: **relative fixture tol** +
**DAG-of-assignments** (multi-step methods — Mohr's circle ships as a golden); the
schema is pinnable for Chisel's emitter. E3.5 shipped: **`kind: solver`** — four
curated SciPy bindings (Brent root-finding, adaptive quadrature, bounded
minimization, RK45 ODE), each with a built-in independent check; the quintic with no
closed form now answers *verified by substitution*. **Content ships separately**:
this repository is the ENGINE — the schema, the gated executor, the verification
stage, the artifact machinery, and a 31-template **demo catalog** proving every
method shape (formulas, DAGs, cases, list reducers, multi-output, all symbolic
operations, all solver bindings) end to end. The full curated corpus —
hand-extracted, reviewed, fixture-gated template batches across the undergraduate
STEM canon — is developed in a private content repository and loads through this
same seam (`assay/templates/chisel/` + the plugin SDK) with zero engine changes:
the seam is public, the content is not. Every resolve hint in any catalog is
checked against the curated vocabulary (`assay.constants` at v0.5),
and the canonical routes
tripwired against catalog growth. E2.10 shipped: the **solve-for execution mode** —
formula template + a target input + the output's value → symbolic inversion through
the same gated machinery, roots filtered by declared dimension, both-roots honesty,
verified by independent forward substitution. E2.12 shipped: the **catalog taxonomy**
(subject → topic → domain, curated Assay-side data with a lockstep CI rule: an
unplaced domain is a red build). E2.11 shipped: **schema v2** — `many: true` list
inputs with whitelisted reducers (`sum`, `sum_inverse`, executed by expansion to
plain gated arithmetic — the arity-suffixed template families collapse to one row)
and the `cases` discriminator (the moment-of-inertia table is one object, the case
selected by `setup`); v1 stays frozen. E2.13 shipped: the **mathematics & statistics
extension** — statistical reducers (`count`/`mean`/`sum_sq`, order-statistic
`min`/`max`/`median`, paired `sum_product` for regression), multi-output steps
templates (slope AND intercept as one row), `limit` and `solve_inequality`
operations, definite/improper symbolic integration (divergence is stated, then
proven), and `erf` — every new result shape carrying its own independent check.
E2.15 shipped: the **execution trace** — step-by-step that IS the computation:
every multi-step answer carries the numbered record of what the executor actually
evaluated (each step's expression and dimension-checked intermediate value, from
the same run that produced the answer — never a post-hoc narration), rendered in
the CLI and the web glass box, embedded in the artifact, reproducible on rerun.
Still ahead: the automated Chisel emitter (E2.3 proper) and StrataDB adoption
(E2.8).

The north-star document is
[`docs/product/assay-vision.md`](docs/product/assay-vision.md): the thesis, the one
claim it rests on, the architecture, the bounded scope, and what it deliberately is
not. The build sequence is
[`docs/implementation-plan.md`](docs/implementation-plan.md).

## Footprint

Measured installed size (site-packages, Linux x86-64, Python 3.12; engineering §8):

| Install | Size | What arrives |
|---|---|---|
| `pip install assay` | ~304 MB | the full deterministic engine — SymPy, SciPy, NumPy, Pint, Matplotlib, pydantic. **No model; nothing else needed.** |
| `assay[api]` | +3 MB | FastAPI + uvicorn (`assay serve`, the web glass box) |
| `assay[stratadb]` | +11 MB | the embedded StrataDB backend |
| `assay[llm]` | +55 MB | llama.cpp via llama-cpp-python |
| a GGUF model | model-sized (≈0.7 GB for a 1B Q4) | `assay model fetch <url> --sha256 <digest>` — explicit, checksum-gated, cached under the user data dir; never in the wheel, never fetched by the compute path |

The wheel itself is pure Python (~240 KB) and is installed-and-smoked on
Linux/macOS/Windows in CI on every push.

## Not

A new CAS · a coding agent · an LLM wrapper around Python · a scientific database · a
Wolfram Language clone · a service you deploy. It is an orchestrator for trusted
computational libraries.

## License

[MIT](LICENSE).

---

Built on the [Strata](https://github.com/stratalab) stack, which ships **embedded**
(StrataDB + the inference layer) so Assay stays self-contained: `pip install` and go,
nothing else to deploy. Adoption is staged: today the StrataDB backend is the optional
`assay[stratadb]` extra while its embedded surface matures, and the in-memory store is
the interim default (see the implementation plan, E2.8).
