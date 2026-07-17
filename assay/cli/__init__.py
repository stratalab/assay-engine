"""The Assay CLI (E1.7 + E2.1, PRD §12, UX §2/§5).

Deterministic verbs (the no-model surface, A-4): ``solve`` / ``integrate`` (the
symbolic goldens), ``plot`` (a figure from verified data; ``--solve`` marks the
verified roots — UX §5.9), ``units`` (Pint conversion, same answer shape), ``run``
(rerun an artifact — the honest reproduction report, UX §5.8 — or execute an edited IR
file, UX §5.4), ``show`` (the concise four-part rendering with progressive
disclosure — ``--method`` / ``--provenance`` / ``--ir``, UX §2), and ``domains``
(everything covered: shipped templates + discovered plugins, E2.4). ``--json`` prints
the answer object itself — the agent contract (UX §6).

``ask`` (E2.1) is natural language through the inference seam: propose → validate →
execute (PRD §11), with the honest states first-class — missing-input ask (interactive)
or fail-clear (``--batch``, UX §5.2), ambiguity fork + ``--pick`` (UX §5.3),
out-of-scope refusal (UX §5.5), and ``--emit-ir`` for the correct-by-editing loop
(UX §5.4). Today it runs on the shipped deterministic backend (no model); the embedded
Strata inference layer lands behind the same seam.

Shipped goldens are ``candidate`` templates; the CLI serves one only after the fixture
gate (``assay.templates.promote``, E2.2) promotes it in-memory — the A-14 gate, honored
at the point of use.

Exit codes: 0 answered (and verified) / reproduced; 1 withheld or reproduction failed;
2 the input or artifact was unusable (fail-clear, with the reason on stderr).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections.abc import Sequence
from functools import cache, lru_cache
from pathlib import Path

import pint
from pydantic import ValidationError

from assay import __version__
from assay.answer import Answer, Figure, ResultValue, Verification, VerificationCheck
from assay.artifact import (
    ArtifactError,
    capture_environment,
    create_artifact,
    load_artifact,
    rerun,
    save_artifact,
)
from assay.execute import ExecutionError, normalize_expression
from assay.inference import (
    Ambiguity,
    CandidateIRError,
    DeterministicBackend,
    InferenceBackend,
    OutOfScope,
    ProposalError,
    ProposedIR,
    validate_candidate,
)
from assay.ir import IR, Producer, Quantity, RenderDirective
from assay.resolver import Resolution, Resolver, key_vocabulary
from assay.templates import Template, chisel_templates, golden_templates
from assay.templates.plugins import discover_plugins, full_catalog
from assay.templates.promote import PromotionError, promote


class UsageError(Exception):
    """The command's input couldn't be used — the message says what to fix (A-12)."""


@lru_cache(maxsize=1)
def _registry() -> pint.UnitRegistry[float]:
    return pint.UnitRegistry()


def _catalog() -> tuple[Template, ...]:
    """Everything installed (E2.4, A-9): shipped goldens + plugin templates."""
    return full_catalog()


@cache
def _promoted(task_id: str) -> Template:
    """Serve an installed template only after the fixture gate promotes it —
    candidate → verified at the point of use (E2.2, A-14). One that fails its own
    fixtures is refused with the reason (a template bug, not the user's input —
    UX §5.6)."""
    catalog = {template.id: template for template in _catalog()}
    if task_id not in catalog:
        raise UsageError(f"no template {task_id!r} is installed")
    return promote(catalog[task_id])


def _slug(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", text.split("=")[0]).strip("_")[:40]
    return stem or "answer"


def _format_value(value: float | str) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _headline(answer: Answer) -> str:
    if not answer.result:
        if answer.verified.ok and answer.figure is not None:
            return f"rendered: {answer.figure.path}"
        return "answer withheld (verification failed)"
    if len(answer.result) > 1:
        return " ,  ".join(f"{v.label} = {_format_value(v.value)}" for v in answer.result)
    value = answer.result[0]
    if not value.unit:  # dimensionless (roots, antiderivatives): "x = -1"
        return f"{value.label} = {_format_value(value.value)}"
    return f"{value.label}: {_format_value(value.value)} {value.unit}"


def render_answer(answer: Answer, artifact_path: str | None = None) -> str:
    """The concise terminal rendering of the one answer object (UX §2)."""
    lines: list[str] = []
    if answer.verified.ok:
        lines += [f"  {_headline(answer)}", ""]
    else:
        lines += [
            "  I computed a candidate answer but could not verify it,"
            " so I'm not returning it:",
            "",
        ]
    bands: list[tuple[str, str]] = []
    if answer.interpretation:
        bands.append(("Interpretation", answer.interpretation))
    if answer.method:
        bands.append(("Method", answer.method))
    # the execution trace (E2.15): shown when there is real step-by-step work — a
    # multi-step DAG, a selected case, or a solve-for recovery (the note); a single
    # bare expression is already the Method line
    if len(answer.steps) > 1 or any(step.note for step in answer.steps):
        for index, step in enumerate(answer.steps, start=1):
            expr = f"{step.expr}  =  " if step.expr else ""
            note = f"   ({step.note})" if step.note else ""
            bands.append(
                (
                    "Steps" if index == 1 else "",
                    f"{index}. {step.label} = {expr}{_format_value(step.value)}"
                    f" {step.unit}{note}".rstrip(),
                )
            )
    for fact in answer.facts:
        source = fact.source
        bands.append(
            (
                "Facts",
                f"{fact.name} = {_format_value(fact.value)} {fact.unit} ·"
                f" {source.library} {source.key} v{source.version}   [resolved, not assumed]",
            )
        )
    verdicts = "   ".join(
        f"✓ {check.name}" if check.ok else f"✗ {check.name}: {check.detail}"
        for check in answer.verified.checks
    )
    bands.append(("Verified", verdicts or ("✓" if answer.verified.ok else "✗")))
    if answer.figure is not None:
        bands.append(
            ("Figure", f"{answer.figure.path}   (rendering of verified data — not a result)")
        )
    if artifact_path is not None:
        bands.append(("Artifact", f"{artifact_path}   ·   rerun: assay run {artifact_path}"))
    lines += [f"  {label:<14} {text}" for label, text in bands]
    return "\n".join(lines)


def _emit_answer(answer: Answer, artifact_path: str | None, as_json: bool) -> int:
    if as_json:
        print(answer.model_dump_json(indent=2))
    else:
        print(render_answer(answer, artifact_path))
    return 0 if answer.verified.ok else 1


def _symbolic_command(task_id: str, args: argparse.Namespace) -> int:
    template = _promoted(task_id)
    setup: dict[str, str] = {"expression": normalize_expression(args.expression)}
    if args.variable:
        setup["variable"] = args.variable
    ir = IR(
        assay_version=__version__,
        query=args.expression,
        domain=template.domain,
        task=template.id,
        setup=setup,
    )
    artifact = create_artifact(ir, template)
    path = save_artifact(artifact, args.out or f"{_slug(args.expression)}.result.json")
    return _emit_answer(artifact.answer, str(path), args.json)


def _cmd_solve(args: argparse.Namespace) -> int:
    return _symbolic_command("solve_equation.univariate", args)


def _cmd_integrate(args: argparse.Namespace) -> int:
    return _symbolic_command("integrate.univariate", args)


def _cmd_differentiate(args: argparse.Namespace) -> int:
    return _symbolic_command("differentiate.univariate", args)


def _cmd_plot(args: argparse.Namespace) -> int:
    """UX §5.9: plot renders verified data — with ``--solve``, the marked roots come
    from the verified solve; without it, just the sampled curve. Matplotlib is imported
    lazily so the other verbs stay fast."""
    from assay.render import directive_data, function_plot_data, render_svg

    expression = normalize_expression(args.expression)
    figure_path = args.fig or f"{_slug(args.expression)}.svg"
    if not args.solve:
        data = function_plot_data(expression, args.variable)
        render_svg(data, figure_path)
        environment = capture_environment()
        curve = data.series[0]
        answer = Answer(
            interpretation=(
                f"plotted {data.title} on [{curve.x[0]:g}, {curve.x[-1]:g}]"
                " (no solve requested)"
            ),
            method="curve sampled from the gated expression (SymPy-parsed; nothing freehand)",
            verified=Verification(
                ok=True,
                checks=[
                    VerificationCheck(
                        name="figure:data",
                        ok=True,
                        detail=f"{len(curve.x)} points computed from the expression;"
                        " every mark traces to a computed quantity",
                    )
                ],
            ),
            figure=Figure(path=str(figure_path), kind="function_plot"),
            assay_version=environment.assay_version,
            versions=environment.versions,
        )
        return _emit_answer(answer, None, args.json)
    template = _promoted("solve_equation.univariate")
    setup: dict[str, str] = {"expression": expression}
    if args.variable:
        setup["variable"] = args.variable
    ir = IR(
        assay_version=__version__,
        query=args.expression,
        domain=template.domain,
        task=template.id,
        setup=setup,
        render=RenderDirective(kind="function_plot", spec={"mark_extrema": True}),
    )
    artifact = create_artifact(ir, template)
    out_path = args.out or f"{_slug(args.expression)}.result.json"
    if not artifact.answer.verified.ok:  # render only verified data: no figure
        path = save_artifact(artifact, out_path)
        return _emit_answer(artifact.answer, str(path), args.json)
    marks = [
        (f"{value.label} = {_format_value(value.value)}", float(value.value), 0.0)
        for value in artifact.answer.result
        if isinstance(value.value, float)  # exact-irrational roots aren't markable (v0)
    ]
    assert ir.render is not None
    data = directive_data(ir.render, expression=expression, variable=args.variable, marks=marks)
    render_svg(data, figure_path)
    answer = artifact.answer.model_copy(
        update={"figure": Figure(path=str(figure_path), kind="function_plot")}
    )
    path = save_artifact(artifact.model_copy(update={"answer": answer}), out_path)
    return _emit_answer(answer, str(path), args.json)


def _cmd_units(args: argparse.Namespace) -> int:
    match = re.fullmatch(
        r"\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*(.+?)\s+(?:to|in)\s+(\S.*?)\s*", args.query
    )
    if match is None:
        raise UsageError("expected '<value> <unit> to <unit>' (e.g. '30 psi to kPa')")
    value, source_unit, target_unit = match.groups()
    ureg = _registry()
    try:
        converted = ureg.Quantity(float(value), source_unit).to(target_unit)
    except pint.UndefinedUnitError as exc:
        raise UsageError(f"unknown unit: {exc}") from exc
    except pint.DimensionalityError as exc:
        raise UsageError(f"incompatible dimensions: {exc}") from exc
    environment = capture_environment()
    answer = Answer(
        result=[
            ResultValue(
                label="value", value=float(converted.magnitude), unit=f"{converted.units:~}"
            )
        ],
        interpretation=f"convert {value} {source_unit} to {target_unit}",
        method="unit conversion (Pint)",
        verified=Verification(
            ok=True,
            checks=[
                VerificationCheck(
                    name="dimension",
                    ok=True,
                    detail=f"{source_unit} → {target_unit} is dimensionally consistent",
                )
            ],
        ),
        assay_version=environment.assay_version,
        versions=environment.versions,
    )
    return _emit_answer(answer, None, args.json)


def _reask_caveat(producer: Producer) -> str:
    """The E2.7 honesty note: a model-produced IR reruns exactly from the artifact,
    but re-*asking* the same question may be read differently. The deterministic
    rules (provider "assay") carry no such caveat — same question, same IR."""
    if producer.provider == "assay":
        return ""
    return "   (re-asking may read differently; the artifact reruns this exact IR)"


def _parse_quantity(raw: str) -> Quantity:
    """Parse an interactively supplied '<value> <unit>' (unit optional)."""
    match = re.fullmatch(r"\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*(.*?)\s*", raw)
    if match is None:
        raise UsageError(f"expected '<value> <unit>' (e.g. '8.33e-6 m^4'); got {raw!r}")
    value, unit = match.groups()
    try:
        magnitude = float(value)
    except ValueError as exc:
        raise UsageError(f"{value!r} is not a number") from exc
    return Quantity(value=magnitude, unit=unit.replace("^", "**"))


def _resolved_line(ir: IR) -> str:
    return ",  ".join(
        f"{name} = {_format_value(fact.value)} {fact.unit}"
        f" ({fact.source.library} {fact.source.key} v{fact.source.version})"
        for name, fact in sorted(ir.resolved.items())
    )


def _have_line(ir: IR) -> str:
    parts = []
    for name, supplied in sorted(ir.inputs.items()):
        if isinstance(supplied, list):  # a list input (schema v2)
            rendered = ", ".join(
                f"{_format_value(q.value)} {q.unit}".rstrip() for q in supplied
            )
            parts.append(f"{name} = [{rendered}]")
        else:
            parts.append(f"{name} = {_format_value(supplied.value)} {supplied.unit}".rstrip())
    return ",  ".join(parts)


def _supply_missing(ir: IR, template: Template, resolution: Resolution, batch: bool) -> IR | None:
    """The missing-input honest state (UX §5.2, A-8): ask interactively for exactly
    what's missing, or fail clear — nothing is ever fabricated or defaulted.
    Returns the completed IR, or ``None`` after printing the fail-clear report."""
    declared = {inp.name: inp for inp in template.inputs}
    needed = [(name, declared[name].dimension) for name in ir.missing_inputs]
    if batch or not sys.stdin.isatty():
        wanted = ", ".join(f"{name!r} ({dimension})" for name, dimension in needed)
        print(f"error: missing required input {wanted}", file=sys.stderr)
        if line := _resolved_line(ir):
            print(f"  resolved: {line}", file=sys.stderr)
        if line := _have_line(ir):
            print(f"  provided: {line}", file=sys.stderr)
        for name, _ in needed:
            if reason := resolution.unresolved.get(name):
                print(f"  {reason}", file=sys.stderr)
        print("  supply it in the question. nothing was fabricated.", file=sys.stderr)
        return None
    count = "one input" if len(needed) == 1 else f"{len(needed)} inputs"
    print(f"  I need {count} to answer this:")
    for name, dimension in needed:
        print(f"    • {name} — dimension {dimension}")
    print()
    if line := _resolved_line(ir):
        print(f"  Resolved   {line}")
    if line := _have_line(ir):
        print(f"  Have       {line}")
    for name, dimension in needed:
        raw = input(f"  Enter {name} ({dimension}, e.g. '<value> <unit>'): ")
        if not raw.strip():
            print(f"error: no value for {name!r} — nothing was fabricated.", file=sys.stderr)
            return None
        ir = ir.model_copy(
            update={
                "inputs": {**ir.inputs, name: _parse_quantity(raw)},
                "missing_inputs": [m for m in ir.missing_inputs if m != name],
            }
        )
    return ir


def _backend(args: argparse.Namespace) -> InferenceBackend:
    """Pick the NL→IR backend: rule-based by default (deterministic, no model); a
    local GGUF model via llama.cpp with ``--llm`` (interim binding — the embedded
    Strata inference layer replaces it behind the same seam). A bare ``--llm``
    resolves the model: $ASSAY_LLM_MODEL, else the single fetched model (E3.3)."""
    if args.llm is not None:
        from assay.inference.llama import LlamaBackend  # lazy: the only model import
        from assay.models import resolve_model

        model = resolve_model(args.llm or None)
        if model is None:
            raise UsageError(
                "no model to serve: pass --llm PATH, set ASSAY_LLM_MODEL, or fetch one"
                " (assay model fetch <url> --sha256 <digest>)"
            )
        return LlamaBackend(model)
    return DeterministicBackend()


def _cmd_model(args: argparse.Namespace) -> int:
    """Local model management (E3.3): explicit, checksum-gated fetch — the compute
    path never touches the network; this verb is the one deliberate exception."""
    from assay.models import ModelFetchError, cached_models, fetch_model, models_dir

    if args.model_command == "fetch":
        try:
            path = fetch_model(args.url, args.sha256, name=args.name)
        except ModelFetchError as exc:
            raise UsageError(str(exc)) from exc
        print(f"  fetched and verified: {path}")
        return 0
    models = cached_models()
    if not models:
        print(f"  no models fetched (they land in {models_dir()})")
        return 0
    for path in models:
        size = path.stat().st_size / (1 << 30)
        print(f"  {path.name}   {size:.2f} GiB   {path}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Natural language through the inference seam (E2.1): propose → validate →
    execute, with every honest state surfaced instead of guessed through."""
    backend = _backend(args)
    catalog = _catalog()
    proposal = backend.propose(args.question, catalog)
    if isinstance(proposal, OutOfScope):
        print("  I can't answer this — it's outside what I cover.")
        print(f"    {proposal.reason}")
        print(f"    covered domains: {', '.join(proposal.covered)}")
        print("  I won't guess.  (assay domains — to see everything I cover.)")
        return 2
    if isinstance(proposal, Ambiguity):
        if args.pick is None:
            ways = len(proposal.options)
            print(f"  This is ambiguous — I can read it {ways} ways:")
            for index, option in enumerate(proposal.options, start=1):
                print(f"    {index}) {option.reading}   ({option.ir.task})")
            picks = "|".join(str(i) for i in range(1, ways + 1))
            print(f"  Re-run with --pick {picks}, or add detail to your question.")
            return 2
        if not 1 <= args.pick <= len(proposal.options):
            raise UsageError(f"--pick must be 1..{len(proposal.options)}")
        proposal = proposal.options[args.pick - 1]
    assert isinstance(proposal, ProposedIR)
    validate_candidate(proposal.ir, catalog)  # the pre-execution gate (A-5)
    template = _promoted(proposal.ir.task)  # serve only fixture-proven templates (A-14)
    resolution = Resolver().resolve_missing(proposal.ir, template)
    ir = resolution.ir
    if ir.missing_inputs:
        completed = _supply_missing(ir, template, resolution, args.batch)
        if completed is None:
            return 2
        ir = completed
    if args.emit_ir:
        Path(args.emit_ir).write_text(ir.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.emit_ir} — edit it, then: assay run {args.emit_ir}")
        return 0
    artifact = create_artifact(ir, template)
    path = save_artifact(artifact, args.out or f"{_slug(args.question)}.result.json")
    code = _emit_answer(artifact.answer, str(path), args.json)
    producer = ir.produced_by
    if not args.json and producer is not None and _reask_caveat(producer):
        print(f"  NL→IR: {producer.provider} {producer.model}{_reask_caveat(producer)}")
    return code


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"{args.artifact!r} is neither an artifact nor an IR (bad JSON: {exc})"
        ) from exc
    if isinstance(payload, dict) and "artifact_version" not in payload:
        # UX §5.4, correct-by-editing: an (edited) IR file executes directly. Missing
        # inputs are resolved where a trusted source is declared; the rest fail clear.
        ir = IR.model_validate(payload)
        template = _promoted(ir.task)
        if ir.missing_inputs:
            ir = Resolver().resolve_missing(ir, template).ir
        artifact = create_artifact(ir, template)
        out = args.out or f"{_slug(Path(args.artifact).stem)}.result.json"
        path = save_artifact(artifact, out)
        return _emit_answer(artifact.answer, str(path), args.json)
    reproduction = rerun(load_artifact(args.artifact))
    mark = {
        "exact": "reproduced ✓",
        "within-tolerance": "reproduced (within tolerance) ~",
        "failed": "reproduction FAILED ✗",
    }[reproduction.status]
    print(f"  {_headline(reproduction.answer)}      {mark}  ({reproduction.detail})")
    return 0 if reproduction.status != "failed" else 1


def _cmd_show(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact)
    answer = artifact.answer
    if args.ir:
        print(artifact.ir.model_dump_json(indent=2))
        return 0
    if args.provenance:
        for fact in answer.facts:
            source = fact.source
            print(
                f"  {fact.name} = {_format_value(fact.value)} {fact.unit} ·"
                f" {source.library} {source.key} v{source.version}"
            )
        if not answer.facts:
            print("  (no resolved facts)")
        pinned = ", ".join(f"{k} {v}" for k, v in sorted(answer.versions.items()))
        print(f"  pinned: {pinned} · assay {answer.assay_version}")
        producer = artifact.ir.produced_by
        if producer is not None:
            print(f"  NL→IR: {producer.provider} {producer.model}{_reask_caveat(producer)}")
        return 0
    if args.method:
        print(f"  Method       {answer.method}")
        for assumption in artifact.template.assumptions:
            print(f"  Assumption   {assumption}")
        return 0
    return _emit_answer(answer, args.artifact, args.json)


def _cmd_facts(args: argparse.Namespace) -> int:
    """The curated fact vocabulary (round 2): what the resolver can resolve — every
    library, version, key, unit, and source. ``--json`` is the emit-time contract
    artifact for template producers (Chisel's extraction references these keys)."""
    vocabulary = key_vocabulary()
    if args.json:
        print(json.dumps(vocabulary, indent=2, sort_keys=True))
        return 0
    for library, table in sorted(vocabulary.items()):
        print(f"  {library} v{table['version']}")
        for key, record in table["keys"].items():
            unit = f" [{record['unit']}]" if record["unit"] else ""
            print(f"    {key}{unit} — {record['source']}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """The HTTP API (E2.6): the answer object over the wire. Lazy imports — the
    deterministic CLI never needs the api extra installed."""
    try:
        import uvicorn

        from assay.api import create_app
    except ImportError as exc:
        raise UsageError(
            "the HTTP API needs the api extra — pip install 'assay[api]'"
        ) from exc
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    """The coverage map (E2.14): where Assay stands, subject → field → topic, with
    the intended source (and gate, where blocked) for everything not yet complete —
    growth is targeted, never bulk ingestion."""
    from assay.templates import coverage
    from assay.templates.plugins import full_catalog

    counts: dict[str, int] = {}
    for template in full_catalog():
        counts[template.domain] = counts.get(template.domain, 0) + 1
    glyph = {"complete": "✓", "in-progress": "~", "pending": "…"}
    totals = {"complete": 0, "in-progress": 0, "pending": 0}
    for _subject_key, subject in sorted(coverage()["subjects"].items()):
        print(f"  {subject['title']}   [CIP {subject['cip']}]")
        for _field_key, field in sorted(subject["fields"].items()):
            cip = ", ".join(field.get("cip", []))
            print(f"    {field['title']}   [CIP {cip}]" if cip else f"    {field['title']}")
            for _topic_key, topic in sorted(field["topics"].items()):
                status = topic["status"]
                totals[status] += 1
                shipped = sum(counts.get(domain, 0) for domain in topic["domains"])
                count = f"  ({shipped} templates)" if shipped else ""
                title = _topic_key.replace("_", " ")
                print(f"      {glyph[status]} {title}{count}   [{status}]")
                if status != "complete":
                    if gate := topic.get("gate"):
                        print(f"          gate: {gate}")
                    if source := topic.get("source"):
                        print(f"          source: {source}")
    print(
        f"  {totals['complete']} complete · {totals['in-progress']} in progress ·"
        f" {totals['pending']} pending"
    )
    return 0


def _cmd_domains(args: argparse.Namespace) -> int:
    """Everything covered, by domain (UX §5.5's pointer): shipped templates plus every
    installed plugin's — with the providing distribution named — and every plugin
    problem stated rather than swallowed."""
    shipped = {template.id for template in golden_templates()}
    rows: dict[str, list[tuple[str, str, str]]] = {}
    for template in golden_templates():
        rows.setdefault(template.domain, []).append(
            (template.id, template.description, "shipped")
        )
    errors: list[str] = []
    seen = set(shipped)
    for template in chisel_templates():
        if template.id in seen:
            errors.append(
                f"chisel: template {template.id!r} collides with an installed template"
                " — ignored"
            )
            continue
        seen.add(template.id)
        rows.setdefault(template.domain, []).append(
            (template.id, template.description, "chisel")
        )
    for plugin in discover_plugins():
        errors.extend(plugin.errors)
        for template in plugin.templates:
            if template.id in seen:
                errors.append(
                    f"{plugin.distribution}: template {template.id!r} collides with an"
                    " installed template — ignored"
                )
                continue
            seen.add(template.id)
            rows.setdefault(template.domain, []).append(
                (template.id, template.description, plugin.distribution)
            )
    from assay.templates import chisel_fixture_attachments, domain_placement, taxonomy

    for target in sorted(set(chisel_fixture_attachments()) - seen):
        errors.append(
            f"chisel: fixture attachment retained for unknown template {target!r} —"
            " it binds automatically when the target lands (never dropped)"
        )
    placement = domain_placement()
    subjects = taxonomy()["subjects"]
    for domain in sorted(set(rows) - set(placement)):
        errors.append(
            f"domain {domain!r} is not placed in the taxonomy — add it to"
            " templates/taxonomy.json (every domain has exactly one home)"
        )
    for subject_key in sorted(subjects):
        subject = subjects[subject_key]
        topics = subject["topics"]
        subject_printed = False
        for topic_key in sorted(topics):
            topic = topics[topic_key]
            local = [domain for domain in topic["domains"] if domain in rows]
            if not local:
                continue
            if not subject_printed:
                print(f"  {subject['title']}")
                subject_printed = True
            count = sum(len(rows[domain]) for domain in local)
            print(f"    {topic['title']}  ({count})")
            for domain in sorted(local):
                print(f"      {domain}")
                for template_id, description, source in sorted(rows[domain]):
                    summary = f" — {description}" if description else ""
                    print(f"        {template_id}{summary}   [{source}]")
    if errors:
        print()
        for error in errors:
            print(f"  ! {error}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assay",
        description="A self-contained computational answer engine (deterministic surface).",
    )
    parser.add_argument("--version", action="version", version=f"assay {__version__}")
    commands = parser.add_subparsers(dest="command")

    for name, handler, help_text in (
        ("solve", _cmd_solve, "solve a univariate equation exactly"),
        ("integrate", _cmd_integrate, "symbolic antiderivative"),
        ("differentiate", _cmd_differentiate, "symbolic derivative"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("expression")
        sub.add_argument("--variable", help="the variable (inferred when unambiguous)")
        sub.add_argument("--out", help="artifact path (default: <slug>.result.json)")
        sub.add_argument("--json", action="store_true", help="print the answer object")
        sub.set_defaults(handler=handler)

    plot = commands.add_parser("plot", help="render a figure from verified data")
    plot.add_argument("expression")
    plot.add_argument("--solve", action="store_true", help="solve too; mark the roots")
    plot.add_argument("--variable", help="the variable (inferred when unambiguous)")
    plot.add_argument("--fig", help="figure path (default: <slug>.svg)")
    plot.add_argument("--out", help="artifact path (with --solve)")
    plot.add_argument("--json", action="store_true", help="print the answer object")
    plot.set_defaults(handler=_cmd_plot)

    units = commands.add_parser("units", help="convert units, e.g. '30 psi to kPa'")
    units.add_argument("query")
    units.add_argument("--json", action="store_true", help="print the answer object")
    units.set_defaults(handler=_cmd_units)

    ask = commands.add_parser("ask", help="ask in natural language (the inference seam)")
    ask.add_argument("question")
    ask.add_argument(
        "--llm",
        nargs="?",
        const="",
        metavar="MODEL.gguf",
        help="propose via a local llama.cpp model; bare --llm resolves"
        " $ASSAY_LLM_MODEL or the single fetched model",
    )
    ask.add_argument(
        "--batch", action="store_true", help="never prompt; fail clear on missing input"
    )
    ask.add_argument("--pick", type=int, help="choose reading N when the question is ambiguous")
    ask.add_argument(
        "--emit-ir",
        dest="emit_ir",
        metavar="PATH",
        help="write the validated IR instead of executing (edit it, then: assay run PATH)",
    )
    ask.add_argument("--out", help="artifact path (default: <slug>.result.json)")
    ask.add_argument("--json", action="store_true", help="print the answer object")
    ask.set_defaults(handler=_cmd_ask)

    run_cmd = commands.add_parser(
        "run", help="rerun an artifact (reproduction report) or execute an IR file"
    )
    run_cmd.add_argument("artifact")
    run_cmd.add_argument("--out", help="artifact path when executing an IR file")
    run_cmd.add_argument("--json", action="store_true", help="print the answer object")
    run_cmd.set_defaults(handler=_cmd_run)

    domains = commands.add_parser("domains", help="everything covered: shipped + plugins")
    domains.set_defaults(handler=_cmd_domains)

    coverage_parser = commands.add_parser(
        "coverage", help="the coverage map: pending / in-progress / complete by topic"
    )
    coverage_parser.set_defaults(handler=_cmd_coverage)

    facts = commands.add_parser("facts", help="the curated fact vocabulary (resolver keys)")
    facts.add_argument("--json", action="store_true", help="machine-readable (the emit contract)")
    facts.set_defaults(handler=_cmd_facts)

    model = commands.add_parser("model", help="manage local GGUF models (checksum-gated)")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    fetch = model_commands.add_parser("fetch", help="download + verify a model (sha256 required)")
    fetch.add_argument("url")
    fetch.add_argument("--sha256", required=True, help="expected digest — fails closed on mismatch")
    fetch.add_argument("--name", help="filename to cache as (default: from the URL)")
    fetch.set_defaults(handler=_cmd_model)
    model_list = model_commands.add_parser("list", help="the fetched models")
    model_list.set_defaults(handler=_cmd_model)

    serve = commands.add_parser("serve", help="serve the HTTP API (needs assay[api])")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_cmd_serve)

    show = commands.add_parser("show", help="inspect an artifact's answer")
    show.add_argument("artifact")
    disclosure = show.add_mutually_exclusive_group()
    disclosure.add_argument("--ir", action="store_true", help="the raw IR")
    disclosure.add_argument("--provenance", action="store_true", help="sources + versions")
    disclosure.add_argument("--method", action="store_true", help="method + assumptions")
    show.add_argument("--json", action="store_true", help="print the answer object")
    show.set_defaults(handler=_cmd_show)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``assay`` command. Returns a process exit code."""
    for stream in (sys.stdout, sys.stderr):
        # A narrow console encoding (Windows cp1252) must degrade the ✓/·/→ glyphs
        # to '?', never crash the answer (caught by the CI wheel smoke, E3.3).
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is None:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except (
        ExecutionError,
        ArtifactError,
        UsageError,
        CandidateIRError,
        ProposalError,
        PromotionError,
        pint.PintError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        target = getattr(args, "artifact", "input")
        print(f"error: {target!r} is not a valid artifact: {exc}", file=sys.stderr)
        return 2
