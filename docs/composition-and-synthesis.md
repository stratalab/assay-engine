# Composition & leaf synthesis — how Assay grows into multi-step problems

*Design note (2026-07-12). **Not scheduled for implementation** — the simple pieces ship
first (E2.9: DAG-of-assignments + relative tol; E3.5: `kind: solver`). This document
records the design so the pieces are built pointing at it. Plan entry: E3.7.*

---

## 1. The problem

Assay answers what a curated template covers and refuses everything else. That is the
brand — but it leaves the question of **multi-step problems**: the stress check that
feeds a comparison that feeds a sizing decision; the question whose method is standard
but whose *combination* isn't a single template.

Two tempting answers are rejected up front:

- **"Let the model write Python."** Infinite flexibility, no floor. The failure mode is
  a silent, plausible, wrong number with facts drawn from parametric memory — the one
  output that would discredit the engine (vision: *not an LLM wrapper around Python*).
- **"Let the model write any template at runtime, including replacements."** Better —
  the schema is a constrained target — but it would let a model's version of a
  standard formula compete with the curated one. Standard formulas are not overridable.
  Ever.

## 2. The invariant: curated is authoritative and closed

**No synthesized artifact can override, shadow, or displace a curated template.** If a
curated template covers the question, it answers; synthesis is never consulted. The
registry already enforces the same shape at the plugin seam (E2.4: a plugin cannot
shadow a shipped id — collisions are ignored and reported); synthesis inherits that
rule verbatim. The only path by which curated content ever changes remains human
review + the fixture gate (E2.2). A model may *add* — at the bottom of the ladder,
flagged — never *replace*.

## 3. The ladder

Answer resolution walks a strict precedence, each rung reachable only when the rung
above cannot answer:

| Rung | What answers | Who wrote the math | Trust posture |
|---|---|---|---|
| 1 | A **curated template** (incl. its E2.9 multi-step DAG) | domain experts / Chisel, fixture-gated | `verified` — the default, the brand |
| 2 | A **composition of curated templates** (`kind: composition`) | the math: curated. The *wiring*: model-proposed | flagged model-planned; every node verified; edges mechanically checked |
| 3 | Rung 2 with a **synthesized leaf** — one missing formula, minimally | the leaf: the model | ⚠ candidate, dimensional+safety floor, promotable only via independent fixtures |
| 4 | Nothing | — | refuse: "I won't guess." |

## 4. Rung 2 — `kind: composition` (the strong tier)

An orchestration is a DAG whose **nodes are invocations of existing templates by id**
and whose edges wire one node's output into another's input. The model never writes an
expression; it only proposes the graph.

Why the wiring is trustworthy in a way prose reasoning is not:

- **Every edge is dimension-checked.** An output feeds an input only if Pint agrees on
  the dimension — a mis-wired plan mostly *cannot execute*.
- **Every node runs its own verification** (bounds, cross-method, dimensional), exactly
  as if invoked alone; facts still resolve from the curated tables with provenance.
- **The composite answer carries the full per-node audit trail** — the four-part answer
  becomes a sequence of verified steps, each citable, the whole reproducible from one
  artifact.

Distinct from E2.9's DAG-of-assignments, which is multi-step *within* one curated
template (chained expressions, one method, one fixture set). Composition is multi-step
*across* templates; E2.9 is its prerequisite (stable step semantics first).

## 5. Rung 3 — leaf synthesis (the weak tier, minimally invoked)

Only when a composition needs a formula family Assay does not have may the model emit
a new template — **that leaf only**, never more, and never one whose id or coverage
collides with a curated template. The leaf is a full schema citizen and the gate treats
it accordingly:

- **Safe grammar** — the same parse gate as every formula; no code, whitelisted
  functions only.
- **Dimensional discipline does structural work**: literals are dimensionless, so a
  baked-in physical constant (`m * 9.8 * h`) fails the output dimension check —
  constants are *forced* into declared inputs with resolve hints (A-2, enforced by
  arithmetic rather than policy). Dimensionless coefficients (`1/48`, `pi`) pass, as
  they should.
- **What the gate cannot catch**: a dimensionally-clean wrong formula (the `/3` vs
  `/48` coefficient). Only *independent* fixtures distinguish those — and a model
  generating its own fixture grades its own homework (correlated errors). Therefore:
- **Synthesized leaves are ⚠ candidates, permanently, until independent fixtures
  arrive** (a confirmed worked answer, a Chisel answer key, human review). Provenance
  records `synthesized:<provider>/<model>`; serving requires the explicit
  candidate opt-in (UX §5.7) and renders the flag; promotion runs the ordinary E2.2
  gate at the pipeline fixture floor. Synthesis is thereby the **intake funnel for the
  curated library**, never a bypass of it.

## 6. What already exists for this (built, tested, shipped)

- The **candidate/verified machinery** (A-14, E2.2) — including "a claimed `verified`
  label is re-earned, never imported."
- The **no-shadow rule** at the plugin seam (E2.4) — the same invariant, same tests.
- The **inference seam** (E2.1) — synthesis is just another proposal shape behind
  `InferenceBackend`, validated before anything runs.
- **Attribution** (E2.7) — `produced_by` names the planner/synthesizer in the artifact.
- The **fixture floor** (round 2) — the promotion bar synthesized leaves must clear.
- The **UX** for provisional answers (§5.7) — the ⚠ flag predates this design.

## 7. What must be designed (the E3.7 work, when scheduled)

1. **Schema**: `kind: composition` — nodes (template ids + input bindings), edges,
   which node's output is *the* answer, per-composition assumptions; validation rules
   (all ids exist and are servable, the graph is acyclic, every edge
   dimension-compatible, no orphan inputs).
2. **Executor**: run nodes in topological order via the existing `execute_template`,
   carry Pint quantities across edges, aggregate per-node verifications into one
   verdict (any node withheld ⇒ composite withheld).
3. **Answer/artifact rendering**: the four bands per step + a composite headline; one
   artifact embedding the plan and every node's template (offline-forever, as today).
4. **The synthesis backend**: prompt/contract for proposing compositions (and leaves,
   rung 3) against the catalog + key vocabulary; the precedence walk; refusal when the
   ladder bottoms out.
5. **Rulings to make at build time**: how compositions are cached (content-hash of the
   plan?); whether a composition can itself be saved as a candidate template
   ("promoted orchestration"); rate/size limits on synthesized leaves per answer.

## 8. Positioning (why this and not codegen)

Wolfram Alpha's ceiling is the same architecture-shaped ceiling: curated methods, no
query-time improviser — and their answer (pair with an LLM) concedes the reasoning
layer lives *above* the computation engine. Assay's version keeps that division of
labor but gives the reasoning layer something codegen can never be: steps that are
individually verified, citable, and reproducible, on a ladder where the model's
freedom is largest exactly where the checks are strongest (wiring), and smallest where
they are weakest (new formulas). An SLM emitting constrained templates/plans against a
hard validator (the Lithos thesis) beats a large model emitting unchecked code — for
this job.

**Sequencing**: E2.9 (DAG steps, relative tol) → E3.5 (`kind: solver`) → E3.7 (this).
Build the simple one first.
