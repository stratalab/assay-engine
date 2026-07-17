"""The llama.cpp binding (E2.1): a real NL→IR model behind the same seam.

Interim by design: Assay binds llama.cpp directly via ``llama-cpp-python`` (MIT — the
optional ``assay[llm]`` extra) serving a **local GGUF model** — on-device, no network,
no provider — until the embedded Strata inference layer ships and replaces this module
behind the unchanged ``InferenceBackend`` interface (engineering §3). The model only
*proposes* (task selection + input extraction + missing-flags, PRD §11); every proposal
still passes ``validate_candidate`` before execution, and facts still come from the
resolver, never from here (A-2, A-5). Generation is pinned (temperature 0, fixed seed,
JSON-object response format) — as deterministic as the served model allows.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from string import Formatter
from typing import Any

from pydantic import ValidationError

from assay import __version__
from assay.inference import (
    Ambiguity,
    Attribution,
    OutOfScope,
    Proposal,
    ProposalError,
    ProposedIR,
)
from assay.ir import IR, Quantity
from assay.resolver import FactTable, builtin_tables
from assay.templates import SymbolicMethod, Template

__all__ = ["LlamaBackend"]

_SEED = 7  # fixed: same model + same question → the same proposal (NFR-1 posture)

_RULES = """\
You translate a user's question into ONE JSON object — a candidate for a computational
engine that will validate and execute it. You never compute, never state facts, and
never invent values.

Respond with exactly one JSON object, no prose, in one of these shapes:
- {"outcome": "candidate", "task": "<template id>", "setup": {...},
   "inputs": {"<name>": {"value": <number>, "unit": "<unit>"}},
   "missing_inputs": ["<name>", ...]}
- {"outcome": "ambiguous", "options": [<two or more candidate objects,
   each with a short "reading" field>]}
- {"outcome": "out_of_scope", "reason": "<why no template fits>"}

Rules:
- "task" must be one of the template ids listed below.
- Extract an input's value ONLY if it is stated in the question; otherwise list that
  input's name in "missing_inputs". NEVER invent, estimate, or default a value.
- Every required input must appear in "inputs" or "missing_inputs".
- Setup values must come from the question (or the allowed values listed below).
"""


def _catalog_block(catalog: Sequence[Template], tables: dict[str, FactTable]) -> str:
    lines = ["Task templates:"]
    for template in catalog:
        if isinstance(template.method, SymbolicMethod):
            contract = 'setup: {"expression": "<the equation or expression>"}'
        else:
            inputs = ", ".join(f"{inp.name} ({inp.dimension})" for inp in template.inputs)
            contract = f"inputs: {inputs}"
            for inp in template.inputs:
                if inp.resolve is None:
                    continue
                table = tables.get(inp.resolve.library)
                placeholders = [
                    field
                    for _, field, _, _ in Formatter().parse(inp.resolve.key)
                    if field
                ]
                for placeholder in placeholders:
                    allowed = (
                        sorted({key.rsplit(".", 1)[0] for key in table.facts}) if table else []
                    )
                    values = f" (allowed: {', '.join(allowed)})" if allowed else ""
                    contract += f'; setup: {{"{placeholder}": "<{placeholder}>"{values}}}'
        lines.append(f"- {template.id} — {template.description or template.domain}; {contract}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ProposalError(f"the model did not emit a JSON proposal: {stripped[:200]!r}")
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProposalError(f"the model's output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProposalError("the model emitted JSON that is not an object")
    return data


class LlamaBackend:
    """NL→IR via a local llama.cpp model (the ``assay[llm]`` extra) — or, for tests,
    any injected ``complete`` callable, so the seam's contract is provable without the
    native dependency or a model file."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        complete: Callable[[str], str] | None = None,
        n_ctx: int = 0,  # 0: the model's own training context
        tables: Iterable[FactTable] | None = None,
    ) -> None:
        if complete is None and model_path is None:
            raise ProposalError("LlamaBackend needs a GGUF model path (or an injected completer)")
        model_name = Path(model_path).name if model_path is not None else "injected"
        self.attribution = Attribution(provider="llama.cpp", model=model_name)
        self._tables = {
            table.library: table
            for table in (builtin_tables() if tables is None else tables)
        }
        if complete is not None:
            self._complete = complete
            return
        assert model_path is not None
        if not Path(model_path).is_file():
            raise ProposalError(f"model file not found: {model_path} (expected a local GGUF)")
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ProposalError(
                "llama-cpp-python is not installed — pip install 'assay[llm]'"
            ) from exc
        try:
            llama = Llama(model_path=str(model_path), n_ctx=n_ctx, seed=_SEED, verbose=False)
        except Exception as exc:  # llama.cpp raises plain ValueError/RuntimeError
            raise ProposalError(f"could not load model {model_path}: {exc}") from exc

        def _complete(prompt: str) -> str:
            response = llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=_SEED,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            content = response["choices"][0]["message"]["content"]
            return str(content) if content is not None else ""

        self._complete = _complete

    def propose(self, question: str, catalog: Sequence[Template]) -> Proposal:
        prompt = (
            f"{_RULES}\n{_catalog_block(catalog, self._tables)}\n\nQuestion: {question}\n"
        )
        data = _extract_json(self._complete(prompt))
        outcome = data.get("outcome")
        if outcome == "candidate":
            return self._candidate(data, question, catalog)
        if outcome == "ambiguous":
            options = data.get("options")
            if not isinstance(options, list) or len(options) < 2:
                raise ProposalError("an ambiguous proposal needs two or more options")
            return Ambiguity(
                options=[self._candidate(option, question, catalog) for option in options],
                attribution=self.attribution,
            )
        if outcome == "out_of_scope":
            return OutOfScope(
                reason=str(data.get("reason") or "no task template fits the question"),
                covered=sorted({template.domain for template in catalog}),
                attribution=self.attribution,
            )
        raise ProposalError(f"the model emitted an unknown outcome {outcome!r}")

    def _candidate(
        self, data: Any, question: str, catalog: Sequence[Template]
    ) -> ProposedIR:
        """One candidate object → a ProposedIR. The task must exist (a hallucinated
        task is a model failure, stated as such); everything else is data the
        pre-execution gate (``validate_candidate``) judges."""
        if not isinstance(data, dict):
            raise ProposalError("a candidate must be a JSON object")
        task = data.get("task")
        templates = {template.id: template for template in catalog}
        if not isinstance(task, str) or task not in templates:
            raise ProposalError(f"the model proposed an unknown task {task!r} — refusing to guess")
        template = templates[task]
        raw_inputs = data.get("inputs") or {}
        if not isinstance(raw_inputs, dict):
            raise ProposalError("candidate 'inputs' must be an object")
        try:
            inputs = {
                str(name): Quantity.model_validate(entry) for name, entry in raw_inputs.items()
            }
            ir = IR(
                assay_version=__version__,
                query=question,
                produced_by=self.attribution,
                domain=template.domain,
                task=task,
                setup=data.get("setup") or {},
                inputs=inputs,
                missing_inputs=[str(name) for name in data.get("missing_inputs") or []],
            )
        except (ValidationError, TypeError) as exc:
            raise ProposalError(f"the model's candidate IR is malformed: {exc}") from exc
        reading = str(data.get("reading") or template.description or task)
        return ProposedIR(ir=ir, reading=reading, attribution=self.attribution)
