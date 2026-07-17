# Assay — Engineering Requirements & Tech Stack

*The "how we build it" to the PRD's "what." Companion to
[`assay-prd.md`](assay-prd.md). Pins the language/runtime, the dependency &
licensing policy (the load-bearing constraint), the module architecture, the
non-functional requirements, packaging, testing, and security.*

---

## 1. Engineering thesis

Four properties the engineering must hold, in priority order — each is also a product
promise, so they are requirements, not preferences:

1. **Permissively licensed, all the way down.** Assay is MIT and ships embedded and
   redistributed; every runtime dependency must be MIT-redistributable. This gates tool
   selection before anything else (§2).
2. **Deterministic & reproducible.** Same IR + pinned versions → the same answer; the
   artifact reruns. With one honest caveat about floating point (§6).
3. **Self-contained.** One install, no external services — Strata (StrataDB + inference)
   ships embedded (§3, §8).
4. **Correct by construction, provable by test.** The fixture/golden discipline and safe
   evaluation are engineering requirements, not QA afterthoughts (§7, §9).

---

## 2. Dependency & licensing policy (the load-bearing constraint)

Assay is an **orchestrator of other people's libraries**, so its dependency graph *is*
its risk surface. Two hard rules and a process.

### 2.1 The runtime-license gate (hard)

**Every runtime/distributed dependency must be under a permissive license** —
MIT, BSD (2/3-clause), Apache-2.0, PSF, ISC, or equivalent. **No copyleft in the runtime:**
no GPL, no AGPL, and no LGPL-by-default (LGPL is only admissible dynamically-linked and
un-bundled, which our embedded/wheel distribution usually violates — treat as excluded
unless a specific case is cleared). A single GPL dependency would relicense Assay.

This is not a stance we can revisit per-library under deadline pressure — it is checked
**before** a dependency is adopted, mechanically, in CI (§2.4).

### 2.2 The nucleus is already clean — verify it stays that way

The v1 nucleus was chosen for capability, and it happens to be **uniformly permissive**
(not an accident — the scientific-Python core standardised on BSD):

| Library | Role | License |
|---|---|---|
| SymPy | symbolic | BSD-3 |
| NumPy / SciPy | numeric + constants | BSD-3 |
| mpmath | high precision | BSD-3 |
| Pint | units / dimensions | BSD-3 |
| Matplotlib | plots | Matplotlib (BSD-style, PSF-derived) |
| statsmodels | statistics | BSD-3 |
| pydantic | schema / validation | MIT |
| StrataDB | embedded store | Apache-2.0 |

Later-domain candidates that are **also clean** (pre-cleared): CoolProp (MIT), RDKit
(BSD-3), Astropy (BSD-3), NetworkX (BSD-3), python-control (BSD-3), CVXPY (Apache-2.0 —
**but** its solver backends vary; pin permissive backends: OSQP/ECOS/SCS, **not** GLPK).

**Known traps to avoid** (name them so nobody reaches for them): GLPK and many MILP/LP
solvers (GPL), SCIP (non-free/academic), some geometry/CAD and a few domain libraries that
are GPL. When a domain needs a copyleft-only tool, the answer is *not* to bundle it — it is
to find a permissive equivalent, wrap the tool as an **optional out-of-process** plugin the
user installs themselves, or skip the domain.

### 2.3 Runtime vs dev/build deps

- **Runtime (shipped in the wheel):** permissive only. Hard gate.
- **Dev / build / test (not distributed):** permissive preferred; weak-copyleft is
  acceptable because it is not redistributed — e.g. `hypothesis` (MPL-2.0) and `mypy`
  are fine as dev-only tools. The distinction is *distribution*, and it is explicit in
  the dependency groups (`[project.dependencies]` vs `[dependency-groups]`).

### 2.4 Process (mechanical, in CI)

- A **license-scan gate** (`scripts/check_licenses.py`, run over a runtime-only
  `uv sync --no-dev` environment) fails CI on any runtime dep outside the permissive
  allowlist — including transitive deps — and **fails closed** on anything it cannot
  classify as permissive.
- A **committed lockfile** (`uv.lock`) pins exact versions + hashes; the pinned versions
  travel into every answer's artifact (reproducibility, §6).
- **Dependency minimalism** — the "nucleus, not catalog" discipline is a licensing and a
  supply-chain control, not just product scope: every added dep is new license risk, new
  CVE surface, and new install weight. Adding one is a reviewed decision.

### 2.5 Data has licenses too

The resolver's constants/material tables are **content**, and content has licenses.
Constants (CODATA via `scipy.constants`) are effectively public data; material-property
databases are **not** uniformly open (MatWeb/ASM are restricted; Materials Project is
CC-BY but API-gated). Resolver data follows the same discipline as Chisel's corpus tiers:
**prefer open/PD sources, record the source + license per value, never ship a restricted
table.** Because Assay records provenance per fact, a bad-license value is *discoverable* —
provenance is a liability for sloppy sourcing and an asset for clean sourcing (the same
argument as the Lithos tier gate).

---

## 3. Language & runtime

- **Python 3.12** is the primary language (the scientific stack is Python; matches the
  ecosystem — Lithos/Chisel are 3.12). Fully type-hinted; `mypy` clean.
- **Rust**, via **PyO3**, is the **embedded StrataDB** surface — consumed as the
  published `stratadb` SDK (its own maturin-built native wheels per platform), not
  compiled in Assay's CI. Assay itself stays a pure-Python wheel; StrataDB is the one
  native-code dependency and the main packaging consideration (§8).
- **The embedded inference layer is Strata's inference layer** — a *complete solution*
  shipped embedded: **llama.cpp built in** (serves a local quantized-GGUF Lithos model
  on-device by default — edge-right, offline, no torch) **plus optional routing to
  OpenAI / Anthropic / Google**. Assay depends on that one layer and **never calls a
  provider directly**; the provider flexibility lives in the layer, not in Assay.
  *Interim (E2.1, until that SDK ships):* Assay binds llama.cpp directly via
  `llama-cpp-python` (MIT — the optional `assay[llm]` extra, a local GGUF, still no
  provider and no network) behind the same `InferenceBackend` seam; the default
  `assay ask` backend is deterministic (rule-based, no model). Swapping in Strata's
  layer replaces the binding module, not the seam.
- **uv** for dependency management and the lockfile (ecosystem standard).

---

## 4. Architecture & module layout

```
assay/
├── ir/           # the IR schema + validation + content_hash()
├── hashing.py    # the content-hash primitive: canonical JSON → sha256 (shared by ir/, store/)
├── templates/    # template schema, validate_template(), registry, trust status (candidate/verified)
├── execute/      # the generic executor — SAFE symbolic parse + evaluate + dimension check
├── resolver/     # the resolver + curated constant/material tables (+ per-value provenance)
├── verify/       # verification stage: dimensional, bounds, cross-method
├── render/       # figures from VERIFIED data (Matplotlib) — plots + geometry diagrams; deterministic
├── answer.py     # the four-part answer object (one shape for terminal + JSON)
├── artifact.py   # answer assembly + the reproducible artifact: save/load/rerun (NFR-2)
├── inference/    # binding to the embedded Strata inference layer (NL → candidate IR)
├── store/        # binding to embedded StrataDB (artifacts, cache, lineage)
├── cli/          # argparse-based CLI (deterministic + `ask`)
└── api/          # HTTP API (v1) — FastAPI/Starlette (MIT/BSD)
```

Boundaries that matter: `execute/`, `resolver/`, and `render/` are **pure and
deterministic** (`render/` consumes only verified data — the compute/render split, PRD
§10.1); `inference/` is the *only* place a model is touched; `store/` and `inference/` are the
embedded-Strata seams; nothing imports across those seams except through their interfaces.
`validate_template()` lives in `templates/` and is the **standalone function Chisel
imports** (dependency-light — pydantic + stdlib only, no heavy imports).

---

## 5. The stack (concrete choices)

| Concern | Choice | License | Why |
|---|---|---|---|
| Language | Python 3.12 | PSF | scientific stack; ecosystem match |
| Native binding | Rust + PyO3 (maturin) | Apache/MIT | embedded StrataDB |
| Deps / lockfile | uv | Apache/MIT | ecosystem standard, hashes |
| Schema / validation | pydantic v2 | MIT | IR + template schema; Chisel already uses it |
| Symbolic / numeric / units / precision | SymPy · SciPy/NumPy · Pint · mpmath | BSD | the compute nucleus |
| Plots / stats | Matplotlib · statsmodels | BSD-style | v1 nucleus |
| CLI | **argparse (stdlib)** | PSF | zero new dep; matches Lithos `cli.py` |
| HTTP API (v1) | FastAPI + Starlette | MIT/BSD | permissive, minimal |
| Test | pytest (+ hypothesis, dev-only) | MIT (MPL dev-only) | golden fixtures + property tests |
| Lint / type | ruff · mypy | MIT | ecosystem standard |

Deliberate non-choices: **no Click/Typer** (argparse suffices — dep minimalism); **no ORM**
(StrataDB is the store); **no provider SDKs in Assay** — the embedded Strata inference
layer (llama.cpp built in + OpenAI/Anthropic/Google) owns all serving and routing.

---

## 6. Non-functional requirements

- **NFR-1 Determinism.** No wall-clock, no RNG, no network in the compute path.
  `PYTHONHASHSEED=0`; **single-threaded BLAS** (`OPENBLAS_NUM_THREADS=1`) in the compute
  path for reproducibility (the Lithos-sandbox discipline). Pinned versions.
- **NFR-2 Reproducibility — stated precisely (the honest floating-point caveat).**
  **Symbolic** results (SymPy) are *exactly* reproducible everywhere. **Numeric** results
  are *bitwise* reproducible on the **same platform + pinned versions**, and reproducible
  **within the fixture tolerance** cross-platform (BLAS/FMA/CPU differences make bitwise
  cross-platform reproduction false to promise). The artifact records platform + versions;
  `assay run` reports *exact* vs *within-tol* accordingly. Do not overclaim "bit-for-bit"
  unconditionally. **Figures are deterministic too** — pin the Matplotlib backend (Agg),
  strip timestamps/metadata, prefer SVG — so the same verified data + view spec render
  identically and regressions are diffable (PRD §10.1).
- **NFR-3 Performance.** The deterministic path is interactive (target sub-second for the
  common nucleus tasks). SymPy can be slow on some operations — cache results in StrataDB
  keyed by **IR content-hash + pinned execution versions** (the hash identifies the
  computation, not the software that ran it — PRD §6); prefer numeric evaluation once a
  symbolic form is fixed.
- **NFR-4 Install footprint.** SciPy/NumPy + the native StrataDB wheel are the base;
  the **model is fetched on first run**, not bundled in the wheel, so `pip install assay`
  stays a reasonable download. State the with-model footprint honestly.
- **NFR-5 Offline.** The compute path makes **no network calls** (facts from local curated
  tables). The only network touch is the optional first-run model fetch.
- **NFR-6 Cross-platform.** Linux + macOS + Windows; native wheels per platform (SciPy has
  them; StrataDB must build them). CI matrix covers all three.
- **NFR-7 Observability.** Structured logs; every answer carries its IR content-hash,
  resolver sources, and pinned versions (the artifact is the audit record).

---

## 7. Security

- **Safe symbolic parsing (real risk, real requirement).** SymPy's `sympify`/`S` will
  **execute arbitrary Python** on a crafted string. Templates come from Chisel and from
  plugins (semi-trusted), so the executor **must** parse formulas with `parse_expr` over a
  **restricted transformation set and a locked-down namespace (no `__builtins__`)** — never
  `sympify`/`eval` on a raw template string. (Lithos's `check_symbolic` already does this;
  same discipline.)
- **Execution sandbox boundary.** Fixed template code (a parsed symbolic expression) is
  trusted; any **model-influenced or `kind: solver` execution** runs sandboxed
  (subprocess + resource limits, the Lithos-sandbox pattern) with no network/fs. Where that
  line sits per phase is PRD §20.
- **No network in the compute path** (NFR-5) is also a security property.
- **Supply chain.** Locked hashes (`uv.lock`), the license-scan gate (§2.4), and dependency
  minimalism are the supply-chain controls.

---

## 8. Packaging & distribution

- **`pip install assay`** → a pure-Python wheel; the native code arrives with the
  `stratadb` dependency's own per-platform wheels (PyO3/maturin, built and published by
  the StrataDB repo). Until StrataDB adoption (implementation plan E2.8) that dependency
  is the optional `assay[stratadb]` extra; CI exercises it on the full platform matrix.
- **The model is not in the wheel.** First-run fetch of the quantised GGUF Lithos model
  (with a checksum), cached under a user data dir; deterministic mode needs no model, so a
  no-model install is valid and smaller.
- **Footprint budget** stated in the README: base install (engine + StrataDB + SciPy) vs
  with-model. Keep the base modest so "pip install and go" is honest on modest hardware.

---

## 9. Testing & CI

- **Golden templates + fixtures** — every template (incl. the goldens shared with Chisel)
  runs through the generic executor; fixtures are the correctness proof (PRD §7).
- **Property-based tests** (hypothesis, dev-only) for the executor and unit algebra —
  dimensional consistency and round-trips over generated inputs.
- **Determinism tests** — same IR twice → identical artifact; reproduction tests exercise
  NFR-2 (exact same-platform, within-tol cross-platform).
- **License-scan gate** (§2.4), **ruff + mypy** clean, full `pytest`, on the Linux/macOS/
  Windows matrix. Same green-bar discipline as Lithos/Chisel.

---

## 10. Versioning & compatibility

- **SemVer** for the package. **The IR schema and the template schema are independently
  versioned** (`ir_version`, template `schema_version`); an artifact records both, and
  `assay run` refuses an artifact whose schema it can't honor (rather than mis-execute).
- Answers are comparable/reproducible only within pinned dependency versions — which is why
  the versions travel in the artifact (§6).

---

## 11. Open questions

1. **IR-model attribution** — *resolved (E2.7)*: `IR.produced_by` records the
   provider + model behind every backend-proposed IR (hand-built IRs carry none); it is
   provenance, not content — the content hash and cache key ignore it — and a
   re-asking caveat surfaces for any non-deterministic producer. When the Strata layer
   routes to a hosted provider, it fills the same field.
2. **StrataDB embedded surface & maturity** — the minimum StrataDB API v0 needs (KV +
   content-hash cache + a small lineage table), and whether its native wheel builds cleanly
   on the CI matrix (gates release; PRD §20).
3. **LGPL case-by-case** — is any high-value LGPL library worth the dynamic-link/un-bundle
   handling, or is the policy "permissive-only, no exceptions" for simplicity? (Leaning
   no-exceptions.)
4. **Windows sandbox** — *positioned (E3.3, [`sandboxing.md`](../sandboxing.md))*: a
   process boundary becomes load-bearing only for model-influenced execution (E3.7
   leaf synthesis); today's solvers are Assay-authored code behind a name whitelist,
   and the grammar gate is the sandbox. When it activates: POSIX subprocess+rlimits
   (the Lithos pattern) and a Windows Job Object backend (stdlib ctypes), one worker
   entrypoint, proven on the CI matrix.
5. **Model-fetch UX** — *resolved (E3.3)*: never bundle, always explicit —
   `assay model fetch <url> --sha256 <digest>` (checksum-gated, fails closed, cached
   under the user data dir; `ASSAY_HOME` overrides); `--llm` resolves explicit path →
   `$ASSAY_LLM_MODEL` → the single cached model, and asks otherwise. The compute path
   never touches the network (the fetcher is the one gate-scoped exception, and the
   engine provably never imports it). A deterministic no-model install stays complete.
