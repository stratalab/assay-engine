# Assay — UX Document

*How Assay feels to use, across every surface. Companion to
[`assay-vision.md`](assay-vision.md) (why) and [`assay-prd.md`](assay-prd.md) (what).
This document defines the interaction model, the answer object, the honest states, the
core flows, and the voice — the experience that makes Assay a glass box rather than a
chatbot or a black box.*

---

## 1. Design principles

Six principles, each a direct consequence of the product's cornerstones:

1. **Answer, not chat.** Assay returns a *structured answer object*, not conversational
   prose. Everything — terminal, JSON, web — is a rendering of that one object.
2. **Glass box, not black box.** Every answer is *openable*: drill from the headline
   result down to the method, the assumptions, each sourced fact, and the raw IR.
   Progressive disclosure — concise by default, complete on demand.
3. **Honest by construction.** Verification status is always shown. "Couldn't verify,"
   "missing input," "ambiguous," and "out of scope" are *first-class states*, never
   hidden behind a confident guess.
4. **Correct by editing, not re-prompting.** When Assay misreads you, you edit the
   structured *interpretation* (the IR) — you don't rephrase and pray. This is the
   anti-chatbot loop.
5. **Reproducible by default.** Every answer yields an artifact you can rerun and share.
   Reproduction is a first-class action, not an export afterthought.
6. **One object for human and machine.** A person reading a terminal and an agent
   consuming JSON get the *same* structured answer — same fields, same guarantees.

---

## 2. The answer object (the core UX primitive)

Every Assay result is one object with a fixed shape (the PRD's four-part output plus
status). Its **default terminal rendering** is concise:

```
$ assay ask "max deflection of a simply supported steel beam, 5 kN center load, 2 m span, I = 8.33e-6 m^4"

  Maximum deflection: 0.50 mm

  Interpretation  simply supported beam · central point load · Euler–Bernoulli
  Method          δ = P·L³ / (48·E·I)
  Facts           E = 200 GPa · steel (structural) · assay.materials v0.3   [resolved, not assumed]
  Verified        ✓ units balance (length)   ✓ within plausible range
  Artifact        ./beam.result.json   ·   rerun: assay run beam.result.json
```

The six always-present bands and their job:

| Band | Shows | Why it's always there |
|---|---|---|
| **Result** | the value + unit | the answer |
| **Interpretation** | the task + key modeling choices, in one line | so you can catch a misread *before* trusting the number |
| **Method** | the formula (one line; expandable to derivation) | the answer is never a bare number |
| **Facts** | every resolved constant/property + its source, tagged `[resolved, not assumed]` | the no-fabrication guarantee, made visible |
| **Verified** | each verification check + its verdict | honest by construction |
| **Artifact** | the saved path + the rerun command | reproducible by default |

**Progressive disclosure.** The default is concise; each band expands on request:

```
assay show beam.result.json                # the four-part answer (above)
assay show beam.result.json --method       # formula + step-by-step derivation
assay show beam.result.json --provenance   # every fact's exact source + version
assay show beam.result.json --ir           # the raw intermediate representation
assay show beam.result.json --plot         # any figure the answer produced
```

When a query asks to *plot* or *draw*, the answer carries a **figure** — rendered from
verified data and labelled a *rendering*, never a result. The verified value is still the
answer; the picture is a faithful view of it (see §5.9, and PRD §10.1).

---

## 3. The states of an answer (honest by construction)

An Assay response is always in exactly one state. There is **no state in which it
returns a confident number it cannot stand behind** — that is the whole point.

| State | Rendering | User's move |
|---|---|---|
| **Answered · verified** | the full object with `Verified ✓` | trust it, rerun it, cite it |
| **Answered · unverified** | the object, `Verified ⚠`, verification withheld or the answer withheld (per `--strict`) | inspect why; treat with caution |
| **Needs input** | lists the missing required input(s) + what was resolved/provided | supply it, or `--assume-*` |
| **Ambiguous** | shows the interpretation(s) it's torn between | confirm or disambiguate |
| **Out of scope** | says so plainly + lists nearby covered domains | narrow the question, or accept it's not covered |
| **Candidate-only** | answers using an unverified (Chisel-candidate) template, clearly flagged | treat as provisional |
| **Error** | states exactly what failed and where | fix input / report the bug |

---

## 4. Surfaces

| Surface | Model in the loop? | Primary user | Phase |
|---|---|---|---|
| **Python API** — `assay.solve(...)`, `assay.run(...)` (`assay.ask(...)` arrives in v1) | optional | developer / scripts | v0 |
| **CLI — deterministic** — `assay solve / integrate / units / run` | **no** | developer, CI | v0 |
| **CLI — natural language** — `assay ask "..."` | optional — deterministic rules by default; a local llama.cpp model via `--llm` (E2.1 interim); the embedded Strata inference layer when its SDK ships | scientist / engineer | v1 |
| **HTTP API** — the answer object over the wire | optional | services, agents | v1 |
| **Web UI** — the glass box in a browser | optional | anyone | minimal ships with `assay[api]` (`assay serve` → `/`); richer later |

The deterministic surfaces come *first* and never require inference — the spine is usable
by itself. When inference *is* used it stays local and contained behind the inference
seam — today a local GGUF via llama.cpp, ultimately the embedded Strata inference layer
(vision §7); nothing external is deployed either way.

---

## 5. Core flows

### 5.1 Deterministic solve (no model)

```
$ assay solve "x^2 + 3x - 4 = 0"

  x = 1 ,  x = -4

  Method    factor over ℝ, solve  (SymPy)
  Verified  ✓ both roots substitute to 0
  Artifact  ./x2_3x_4.result.json
```

Fast, deterministic, no LLM. `assay units "30 psi to kPa"` → `206.84 kPa`, same shape.

### 5.2 Missing input — ask (interactive) vs fail-clear (batch)

Interactive: Assay resolves what it can, then asks for exactly what's missing — it never
fabricates:

```
$ assay ask "max deflection of a simply supported steel beam, 5 kN center load, 2 m span"

  I need one input to answer this:
    • second_moment_of_area (I) — dimension length⁴

  Resolved   E = 200 GPa (steel, structural — assay.materials v0.3)
  Have       P = 5 kN,  L = 2 m

  Enter I (e.g. "8.33e-6 m^4"), or --assume-section to pick a standard section:
  ▸ _
```

Non-interactive fails closed with a clear reason and a nonzero exit — never a guess:

```
$ assay ask "...same query..." --batch
  error: missing required input 'second_moment_of_area' (length⁴)
    resolved: E = 200 GPa (steel)   provided: P = 5 kN, L = 2 m
    supply it in the query or pass --assume-section. nothing was fabricated.
  (exit 2)
```

### 5.3 Ambiguity — confirm the interpretation before computing

When more than one template fits, Assay surfaces the fork instead of silently choosing:

```
$ assay ask "deflection of a steel beam, 5 kN load, 2 m"

  This is ambiguous — I can read it two ways:
    1) simply supported · central point load        (beam_deflection.simply_supported.center_point)
    2) cantilever · end point load                  (beam_deflection.cantilever.end_point)

  Re-run with --pick 1|2, or add the support/loading to your question.
```

### 5.4 Misinterpretation — correct by editing the IR

The anti-chatbot loop. If the interpretation is wrong, you don't rephrase — you open the
structured interpretation, fix it, and rerun:

```
$ assay ask "..." --emit-ir beam.ir.yaml     # writes the IR instead of executing
$ $EDITOR beam.ir.yaml                        # fix the task/inputs directly
$ assay run beam.ir.yaml                       # execute the corrected interpretation
```

You are editing *what Assay understood*, not fighting a parser. The IR is the same
object whether Assay wrote it or you did.

### 5.5 Out of scope — refuse, don't guess

```
$ assay ask "simulate turbulent flow over an airfoil at Mach 0.8"

  I can't answer this — it's outside what I cover.
    no template matches: compressible CFD / turbulence modeling
    nearest covered:     dimensional analysis, incompressible-flow basics
  I won't guess.  (assay domains — to see everything I cover.)
```

### 5.6 Verification failed — withhold with a reason

If a computed answer fails verification, Assay **withholds it** and blames the right
party — the template, not the user:

```
  I computed a candidate answer but could not verify it, so I'm not returning it:
    ✗ cross-method: symbolic 0.50 mm vs numeric 0.46 mm  (Δ > tol)
  This is a template bug, not your input. Filed: assay-tmpl beam_deflection.* @v0.3
  Re-run with --unsafe to see the unverified value anyway.
```

### 5.7 Candidate template — provisional, clearly flagged

A query answered via a Chisel-candidate (not-yet-`verified`) template is labelled, and
by default requires opt-in:

```
$ assay ask "..." --allow-candidate

  Torsional stress: 42.6 MPa      ⚠ CANDIDATE TEMPLATE — unverified
    template thin_walled_torsion @candidate  ·  source: <question-paper id>
    fixtures not yet passing; treat as provisional. (default is to refuse candidates.)
```

### 5.8 Reproduce

```
$ assay run beam.result.json

  Maximum deflection: 0.50 mm      reproduced ✓  (identical; SymPy 1.x, Pint 0.x)
```

Same artifact, anywhere, offline, forever — independent of Assay's servers or version
drift, because the pinned versions travel with it.

### 5.9 Plot + solve (a figure alongside a verified result)

A compound query returns a verified value *and* a figure rendered from verified data:

```
$ assay ask "plot and solve x^2 - 5x + 6 = 0"

  x = 2 ,  x = 3

  Interpretation  solved the quadratic; plotted y = x²−5x+6 on [1, 4], roots + vertex marked
  Method          factor + solve (SymPy); curve sampled from the verified formula
  Verified        ✓ both roots substitute to 0
  Figure          ./quadratic.svg   (rendering of verified data — not a result)
  Artifact        ./quadratic.result.json   ·   rerun: assay run quadratic.result.json
```

The **result** is the verified roots; the **figure** is a faithful view of computed data —
the curve is the real formula, the marked points are the real roots. `assay plot
"x^2 - 5x + 6"` renders without solving; `--solve` adds the roots. Nothing on the figure is
un-computed (PRD §10.1).

---

## 6. The agent-facing UX (the API is a surface)

For a machine consumer the "experience" is a predictable, structured, *citable* result.
`--json` (and the HTTP API) return the same object:

```jsonc
{
  "result":         [ { "label": "max_deflection", "value": 5.0e-4, "unit": "m" } ],
  "interpretation": "simply supported beam, central point load (beam_deflection.simply_supported.center_point)",
  "method":         "P*L**3/(48*E*I)",
  "facts":          [ { "name": "E", "value": 2.0e11, "unit": "Pa",
                        "source": { "library": "assay.materials", "key": "steel.structural.E", "version": "0.3" } } ],
  "verified":       { "ok": true, "checks": [ { "name": "dimension:length", "ok": true, "detail": "" },
                                              { "name": "bounds",           "ok": true, "detail": "" } ] },
  "figure":         null,                      // a rendering of verified data, when asked for (§5.9)
  "ir_hash":        "9f2c…",                   // keys the full IR in the artifact/store
  "assay_version":  "0.0.1",
  "versions":       { "sympy": "1.x", "pint": "0.x" }   // pinned — the reproducibility record
}
```

This is the `Answer` type (`assay/answer.py`) verbatim — the JSON is the object, not a
report about it. Why this is good agent UX: `verified` (per-check verdicts, not a bare
boolean) lets the agent decide whether to trust the value; `facts[].source` lets it
**cite** the answer; `ir_hash` + `versions` key the exact IR and the pinned libraries
(both travel in the artifact file), so it can **reproduce** via `assay run`; and the
schema is *stable*, so the agent parses one shape forever. An agent gets a computation
it can stand behind — the thing a chatbot can never hand it.

---

## 7. Failure & refusal UX

Failures are answers too, and they obey the same honesty rule (PRD A-12): **say exactly
what went wrong, where, and what to do — never guess, never hedge-theatre.**

- **Can't map** → "no template matches X; nearest covered: …".
- **Can't resolve a fact** → "couldn't resolve `E` for material `inconel-718`; not in
  `assay.materials`. Supply it, or add a source." (Never invents it.)
- **Can't verify** → withhold + name the failing check + blame the template.
- **Out of scope** → refuse + list coverage.

Every failure carries a stable machine code (for scripts/agents) *and* a human sentence.

---

## 8. Voice

Assay speaks like an **instrument**, not an assistant. Precise, plain, and honest to a
fault. It states uncertainty rather than performing confidence, and it refuses rather
than guesses.

- **Do:** "I need `I` to answer this." · "I won't guess." · "Verified: units balance." ·
  "E = 200 GPa [resolved, not assumed]."
- **Don't:** "Great question! Let me help you with that…" · a number with no method ·
  a confident answer it can't verify · apologies in place of a reason.

Facts are always tagged with their epistemic status (`resolved` vs a value you supplied).
The signature line of the whole product is **"I won't guess."**

---

## 9. Open questions

1. **Interactive depth** — how many missing inputs before Assay stops asking one-by-one
   and shows the whole form to fill at once.
2. **`--strict` default** — should an *unverified* answer be withheld by default (safe)
   or shown-with-caveat (permissive)? Leaning withheld.
3. **Confirmation threshold** — at what interpretation-confidence does Assay auto-confirm
   (§5.3) vs just compute; is that user-configurable.
4. **Terminal richness** — plain text vs a TUI with foldable sections; how the four
   bands render in a minimal 80-column terminal vs a rich one.
5. **Web glass box** — how the IR/method/provenance drill-down is presented visually
   without becoming an overwhelming form.
6. **Units-out control** — how the user pins the output unit (`--in mm`) and how a
   dimension mismatch there is surfaced.
