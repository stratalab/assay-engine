"""Template plugins (E2.4, A-9): new domains arrive as installed packages, not core edits.

Discovery is the packaging ecosystem's own machinery: a distribution declares an entry
point in the ``assay.templates`` group whose target is a zero-argument callable
returning template **records** (plain mappings — data, not code). Assay validates every
record through the A-15 seam (``validate_template``) and serves nothing that hasn't
passed the fixture gate (E2.2, A-14): **install grants presence, never trust** — a
plugin's template is a candidate like any other until its own fixtures pass under the
generic executor. Each template's origin is recorded as the providing distribution's
name + version (the "versioned plugin" of the epic).

Failure is contained and stated (A-12): a plugin that fails to import, whose provider
raises, or whose records don't validate is reported entry by entry — it never takes
down discovery, the other plugins, or the engine.

The counterpart file a plugin ships (pyproject.toml):

    [project.entry-points."assay.templates"]
    beamkit = "beamkit:templates"        # templates() -> iterable of records
"""

from __future__ import annotations

from importlib import metadata

from pydantic import BaseModel, ConfigDict

from assay.templates import (
    Template,
    TemplateRegistry,
    TemplateValidationError,
    chisel_fixture_attachments,
    chisel_templates,
    golden_templates,
    validate_template,
)

__all__ = [
    "GROUP",
    "DiscoveredPlugin",
    "PluginIngest",
    "discover_plugins",
    "full_catalog",
    "ingest_plugins",
    "plugin_templates",
]

GROUP = "assay.templates"


class DiscoveredPlugin(BaseModel):
    """One entry point's outcome: its validated candidate templates, and every
    problem it had — stated, not swallowed."""

    model_config = ConfigDict(extra="forbid")
    name: str
    distribution: str  # "beamkit 0.1.0" — the versioned provenance
    templates: list[Template] = []
    errors: list[str] = []


class PluginIngest(BaseModel):
    """A gated batch load: per-template promotion reports plus every contained error."""

    model_config = ConfigDict(extra="forbid")
    verified: list[str] = []
    quarantined: list[str] = []
    errors: list[str] = []


def discover_plugins() -> list[DiscoveredPlugin]:
    """Find every installed ``assay.templates`` entry point and validate what it
    offers. Deterministic order (by entry-point name); pure — nothing registers here.
    """
    plugins: list[DiscoveredPlugin] = []
    for entry_point in sorted(metadata.entry_points(group=GROUP), key=lambda ep: ep.name):
        dist = entry_point.dist
        label = f"{dist.name} {dist.version}" if dist is not None else "unknown distribution"
        plugin = DiscoveredPlugin(name=entry_point.name, distribution=label)
        plugins.append(plugin)
        try:
            provider = entry_point.load()
            records = list(provider())
        except Exception as exc:  # a broken plugin is contained, never fatal
            plugin.errors.append(f"plugin {entry_point.name!r} ({label}) failed to load: {exc}")
            continue
        for index, record in enumerate(records):
            try:
                plugin.templates.append(_validated(record))
            except TemplateValidationError as exc:
                plugin.errors.append(f"{label} record {index}: {exc}")
    return plugins


def _validated(record: object) -> Template:
    """One door for every record: the A-15 seam."""
    if not isinstance(record, dict):
        raise TemplateValidationError(
            f"a plugin record must be a mapping, got {type(record).__name__}"
        )
    return validate_template(record)


def plugin_templates() -> tuple[Template, ...]:
    """Every valid plugin template (candidates — the gate still stands between them
    and serving), flattened across plugins."""
    return tuple(
        template for plugin in discover_plugins() for template in plugin.templates
    )


def full_catalog() -> tuple[Template, ...]:
    """Everything installed (A-9): the shipped goldens (with any Chisel fixture
    attachments merged — independent proof landing on the same ids), the shipped
    Chisel batches (E2.3), then every valid plugin template — all candidates until the
    gate promotes them. On an id collision the earlier tier wins (nothing shadows a
    shipped template)."""
    templates = {template.id: template for template in golden_templates()}
    for target, fixtures in chisel_fixture_attachments().items():
        existing = templates.get(target)
        if existing is None:
            continue  # a plugin target, or a future id: nothing to strengthen here
        merged = existing.model_copy(update={"fixtures": [*existing.fixtures, *fixtures]})
        templates[target] = validate_template(merged.model_dump(mode="json"))
    for template in chisel_templates():
        templates.setdefault(template.id, template)
    for template in plugin_templates():
        templates.setdefault(template.id, template)
    return tuple(templates.values())


def ingest_plugins(registry: TemplateRegistry) -> PluginIngest:
    """The gated batch load (E2.2): every discovered template goes validate → fixture
    gate → register. Passing templates register verified and serve; failing ones
    register quarantined; duplicates and broken plugins are reported, not fatal."""
    from assay.templates.promote import ingest

    result = PluginIngest()
    for plugin in discover_plugins():
        result.errors.extend(plugin.errors)
        for template in plugin.templates:
            if template.id in registry:
                result.errors.append(
                    f"{plugin.distribution}: template {template.id!r} is already"
                    " registered — ignored"
                )
                continue
            report = ingest(template, registry)
            if report.promoted:
                result.verified.append(template.id)
            else:
                result.quarantined.append(template.id)
                details = "; ".join(f.detail for f in report.failures)
                result.errors.append(
                    f"{plugin.distribution}: template {template.id!r} quarantined —"
                    f" fixtures failed: {details}"
                )
    return result
