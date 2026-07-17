# Sandboxing `kind: solver` and model-influenced execution — position (E3.3)

*Answers engineering §11 Q4 (the Windows sandbox). A design position, not an
implementation — because of **when** a process boundary actually becomes load-bearing.*

## When a sandbox matters, honestly

Today, nothing Assay executes is model-influenced code:

- Formula/symbolic methods are **data** through a parse-only grammar gate (no eval,
  no attribute access, whitelisted calls only) — the gate *is* the sandbox, and it is
  platform-independent.
- Solver bindings (E3.5) are **Assay-authored trusted code** behind a name whitelist —
  a model can choose *which* curated binding runs, never what code runs.

A process boundary becomes load-bearing at exactly one future point: **executing
candidate-tier artifacts a model shaped** beyond the grammar's reach — E3.7 leaf
synthesis if it ever grows past the safe grammar, or any future codegen (which the
vision rejects). Until then an OS sandbox would add platform risk while defending
nothing the grammar doesn't already defend.

## The design, when it activates

One shape, two platform backends, wrapped around a single worker entrypoint
(`execute a validated IR against an embedded template, print the JSON result`):

- **POSIX** (the Lithos pattern): `subprocess` + `resource.setrlimit` — CPU seconds
  (`RLIMIT_CPU`), address space (`RLIMIT_AS`), no new files beyond stdio
  (`RLIMIT_NOFILE`), `RLIMIT_NPROC` against forking — plus a hard wall-clock kill from
  the parent. No network by construction (the gate already proves the engine imports
  none; the rlimits make it a process guarantee).
- **Windows**: a **Job Object** (`JOBOBJECT_EXTENDED_LIMIT_INFORMATION`: process
  memory cap, active-process count = 1, job time limit; `KILL_ON_JOB_CLOSE`) around a
  `CREATE_SUSPENDED` child assigned to the job before it runs — stdlib `ctypes`, no
  new dependency. Same wall-clock kill, same worker entrypoint.
- Both report the same three outcomes upward: completed / resource-limit exceeded
  (named) / killed — mapped onto the ordinary fail-clear error shape (A-12).

CI's existing three-platform matrix is the proof harness when this lands: the sandbox
tests run everywhere the wheel smokes.

## The rule until then

The grammar gate stays the only execution boundary, **and therefore the grammar gate
stays load-bearing**: nothing may widen `expr_symbols` / the locked parse / the solver
whitelist without treating it as a security change (engineering §7).

## The widening ledger

Every deliberate widening of the safe namespace, with its reason:

- `asin`, `acos`, `atan`, `log10`, `abs` — chisel round 4 §3/§4 (the RLC phase-angle
  family; decibels/beats). Dimensionless-in (except `abs`), one argument, total on
  their domains or fail-clear.
- `erf` (E2.13) — the normal CDF for the statistics corpus:
  `(1 + erf(z/sqrt(2)))/2`. `math.erf` at evaluation, `sympy.erf` at parse;
  dimensionless-in/dimensionless-out, one argument, total. Deliberately NOT added at
  the same time: `factorial`/`comb` (integer-domain semantics the float pipeline
  cannot honor honestly — binomial coefficients wait on `kind: table` or a typed
  integer input) and `floor`/`ceil` (discontinuities break the difference-quotient
  and tolerance machinery).
- Reducers (`sum`, `sum_inverse` E2.11; `count`, `mean`, `sum_sq`, `min`, `max`,
  `median`, paired `sum_product` E2.13) are NOT evaluation-namespace widenings: they
  expand to plain whitelisted arithmetic before parse (sum-like) or bind as
  precomputed scalars at input-binding time (order statistics) — the evaluation walk
  and the locked parse are untouched.
