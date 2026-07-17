"""HTTP API (E2.6, PRD §12, UX §6): the answer object over the wire — the machine surface.

The agent contract: every computed response carries the **same ``Answer`` object** the
CLI's ``--json`` prints — per-check ``verified`` verdicts (trust is decidable), per-fact
``facts[].source`` (the answer is citable), ``ir_hash`` + pinned ``versions`` (it is
reproducible) — plus the full **artifact**, so the caller can rerun the computation
bit-for-bit (``POST /v1/rerun``) without this server keeping any state.

The honest states are first-class response shapes, discriminated by ``outcome``:
``answer`` / ``missing_inputs`` (fail-clear — the API never prompts and never
fabricates) / ``ambiguous`` (the fork + ``pick`` to choose) / ``out_of_scope`` (refusal
with what *is* covered). Engine-level refusals (a hostile expression, a template that
fails its own fixtures, an unusable request) are HTTP 400 with the reason (A-12).

FastAPI/Starlette (MIT/BSD — engineering §5), shipped as the optional ``assay[api]``
extra; ``assay serve`` runs it. Inference here is the deterministic backend — the
embedded Strata layer lands behind the seam, not behind this surface.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from assay import __version__
from assay.answer import Answer
from assay.artifact import (
    Artifact,
    ArtifactError,
    Reproduction,
    capture_environment,
    create_artifact,
    rerun,
)
from assay.execute import ExecutionError
from assay.inference import (
    Ambiguity,
    CandidateIRError,
    DeterministicBackend,
    OutOfScope,
    ProposalError,
    ProposedIR,
    validate_candidate,
)
from assay.ir import IR
from assay.resolver import Resolver
from assay.templates import Template, TemplateValidationError
from assay.templates.plugins import full_catalog
from assay.templates.promote import PromotionError, promote

__all__ = ["create_app"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    pick: int | None = None  # choose reading N after an "ambiguous" outcome


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ir: IR


class RerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact: Artifact


class AnswerEnvelope(BaseModel):
    """A computed (possibly withheld — see ``answer.verified``) result: the answer
    object verbatim, plus the artifact that reproduces it."""

    model_config = ConfigDict(extra="forbid")
    outcome: Literal["answer"] = "answer"
    answer: Answer
    artifact: Artifact


class MissingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dimension: str
    reason: str = ""


class MissingInputsEnvelope(BaseModel):
    """Fail-clear (A-8): exactly what's missing, what resolved, what was provided —
    and nothing fabricated. Complete the returned IR and POST it to /v1/run."""

    model_config = ConfigDict(extra="forbid")
    outcome: Literal["missing_inputs"] = "missing_inputs"
    task: str
    needed: list[MissingInput]
    ir: IR


class AmbiguousOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pick: int
    task: str
    reading: str


class AmbiguousEnvelope(BaseModel):
    """The fork, surfaced (UX §5.3): repeat the request with ``pick`` set."""

    model_config = ConfigDict(extra="forbid")
    outcome: Literal["ambiguous"] = "ambiguous"
    options: list[AmbiguousOption]


class OutOfScopeEnvelope(BaseModel):
    """The refusal (UX §5.5): no guess, and what *is* covered."""

    model_config = ConfigDict(extra="forbid")
    outcome: Literal["out_of_scope"] = "out_of_scope"
    reason: str
    covered: list[str]


AskResponse = AnswerEnvelope | MissingInputsEnvelope | AmbiguousEnvelope | OutOfScopeEnvelope


class TemplateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    domain: str
    subject: str  # taxonomy placement: subject → topic → domain (curated data)
    topic: str
    description: str
    status: str


def _served(template_id: str, catalog: dict[str, Template]) -> Template:
    """The serving gate, same as every surface (E2.2, A-14): promote at the point of
    use or refuse with the reason."""
    if template_id not in catalog:
        raise HTTPException(status_code=404, detail=f"no template {template_id!r} is installed")
    return promote(catalog[template_id])


def _computed(ir: IR, template: Template) -> AnswerEnvelope:
    artifact = create_artifact(ir, template)
    return AnswerEnvelope(answer=artifact.answer, artifact=artifact)


def _resolved_or_missing(ir: IR, template: Template) -> AskResponse:
    resolution = Resolver().resolve_missing(ir, template)
    if resolution.ir.missing_inputs:
        declared = {inp.name: inp for inp in template.inputs}
        needed = [
            MissingInput(
                name=name,
                dimension=declared[name].dimension if name in declared else "",
                reason=resolution.unresolved.get(name, ""),
            )
            for name in resolution.ir.missing_inputs
        ]
        return MissingInputsEnvelope(task=template.id, needed=needed, ir=resolution.ir)
    return _computed(resolution.ir, template)


def create_app() -> FastAPI:
    """Build the API app. Stateless: every response carries what the caller needs to
    resume (an IR to complete, an artifact to rerun)."""
    app = FastAPI(title="assay", version=__version__)

    refusals = (
        ExecutionError,
        ArtifactError,
        CandidateIRError,
        ProposalError,
        PromotionError,
        TemplateValidationError,
    )

    async def _refused(request: Any, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    for refusal in refusals:
        app.add_exception_handler(refusal, _refused)

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        """The web glass box (E3.2, UX §9.5): the same answer object, rendered —
        concise first, IR/method/provenance one click deep, never a form. One
        self-contained file: inline CSS/JS, no external assets, no build step."""
        return (resources.files("assay.api") / "index.html").read_text(encoding="utf-8")

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        environment = capture_environment()
        return {
            "status": "ok",
            "assay_version": environment.assay_version,
            "versions": environment.versions,
        }

    @app.get("/v1/domains")
    async def domains() -> list[TemplateSummary]:
        from assay.templates import domain_placement

        placement = domain_placement()
        summaries = []
        for template in sorted(full_catalog(), key=lambda t: (t.domain, t.id)):
            subject, topic = placement.get(template.domain, ("", ""))
            summaries.append(
                TemplateSummary(
                    id=template.id,
                    domain=template.domain,
                    subject=subject,
                    topic=topic,
                    description=template.description,
                    status=template.provenance.status,
                )
            )
        return summaries

    @app.get("/v1/taxonomy")
    async def taxonomy_tree() -> dict[str, Any]:
        from assay.templates import taxonomy

        return taxonomy()

    @app.post("/v1/ask")
    async def ask(request: AskRequest) -> AskResponse:
        catalog = full_catalog()
        proposal = DeterministicBackend().propose(request.question, catalog)
        if isinstance(proposal, OutOfScope):
            return OutOfScopeEnvelope(reason=proposal.reason, covered=proposal.covered)
        if isinstance(proposal, Ambiguity):
            if request.pick is None:
                return AmbiguousEnvelope(
                    options=[
                        AmbiguousOption(pick=index, task=option.ir.task, reading=option.reading)
                        for index, option in enumerate(proposal.options, start=1)
                    ]
                )
            if not 1 <= request.pick <= len(proposal.options):
                raise HTTPException(
                    status_code=400, detail=f"pick must be 1..{len(proposal.options)}"
                )
            proposal = proposal.options[request.pick - 1]
        assert isinstance(proposal, ProposedIR)
        validate_candidate(proposal.ir, catalog)  # the pre-execution gate (A-5)
        template = _served(proposal.ir.task, {t.id: t for t in catalog})
        return _resolved_or_missing(proposal.ir, template)

    @app.post("/v1/run")
    async def run(request: RunRequest) -> AskResponse:
        """Execute a caller-built (or completed) IR — the agent's primary verb."""
        catalog = {template.id: template for template in full_catalog()}
        validate_candidate(request.ir, tuple(catalog.values()))
        template = _served(request.ir.task, catalog)
        return _resolved_or_missing(request.ir, template)

    @app.post("/v1/rerun")
    async def rerun_artifact(request: RerunRequest) -> Reproduction:
        """Reproduce an artifact (NFR-2): exact / within-tolerance / failed, stated."""
        return rerun(request.artifact)

    return app
