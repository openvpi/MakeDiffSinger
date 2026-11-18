"""Separate computation scope from permission to replace annotation layers."""

from copy import deepcopy
from dataclasses import dataclass, field

from .annotations import TIERS, tier_signatures
from .types import Annotation, Issue, Outcome
from .validation import validate


@dataclass
class ReannotationPlan:
    layers: tuple[str, ...]
    expected: dict[str, dict[str, str]]
    replace_edited: bool = False
    stages: list[str] = field(default_factory=list)


def plan_reannotation(annotations, *, layers=("notes",), sample_keys=None, replace_edited=False):
    layers = tuple(dict.fromkeys(layer for name in layers for layer in
                                (("texts", "words", "phones") if name == "fa" else
                                 TIERS if name == "all" else (name,))))
    if not layers or set(layers) - set(TIERS):
        raise ValueError("Select fa, all, or explicit texts/words/phones/notes layers.")
    expected = {a.key: tier_signatures(a.tiers) for a in annotations
                if sample_keys is None or a.key in sample_keys}
    stages = (["tifa"] if set(layers) & {"texts", "words", "phones"} else []) + (["game"] if "notes" in layers else [])
    return ReannotationPlan(layers, expected, replace_edited, stages)


def apply_annotation_updates(original: Annotation, candidate: Annotation, plan: ReannotationPlan,
                             *, profiles=None) -> Outcome:
    if original.key != candidate.key or original.key not in plan.expected:
        return Outcome(original.sample, "conflict", issues=[Issue("outside_scope", "Sample is outside the update plan.", "update")])
    if tier_signatures(original.tiers) != plan.expected[original.key]:
        return Outcome(original.sample, "conflict", issues=[Issue("concurrent_edit", "Layers changed since planning.", "update")])
    edited = set(original.edited_layers) & set(plan.layers)
    if edited and not plan.replace_edited:
        return Outcome(original.sample, "conflict", issues=[Issue("manual_edit", f"Edited layers require explicit replacement: {sorted(edited)}", "update")])
    missing = set(plan.layers) - candidate.tiers.keys()
    if missing:
        return Outcome(original.sample, "conflict", issues=[Issue("missing_candidate", f"Candidate lacks {sorted(missing)}.", "update")])
    merged = deepcopy(original)
    for layer in plan.layers:
        merged.tiers[layer] = candidate.tiers[layer]
    changed_fa = set(plan.layers) & {"texts", "words", "phones"}
    if changed_fa:
        merged.sample = candidate.sample
        merged.duration = candidate.duration
        if "notes" not in plan.layers and "notes" in original.tiers:
            merged.stale = sorted(set(merged.stale) | {"notes"})
        if {"texts", "words", "phones"} <= set(plan.layers):
            merged.quality = candidate.quality
            for key in ("tifa", "raw_phone_spans_frames", "pronunciation_choices",
                        "pronunciation_candidates", "pronunciation_scores"):
                if key in candidate.provenance:
                    merged.provenance[key] = candidate.provenance[key]
                else:
                    merged.provenance.pop(key, None)
            merged.provenance["time_resolution_seconds"] = candidate.provenance.get("time_resolution_seconds", 0.0)
            merged.stale = [name for name in merged.stale if name != "quality"]
        else:
            merged.stale = sorted(set(merged.stale) | {"quality"})
        if set(plan.layers) & {"phones", "words"}:
            merged.provenance.pop("imported_ph_num", None)
            merged.provenance.pop("imported_phones_signature", None)
        if {"phones", "words"} <= set(plan.layers):
            merged.provenance["groups_verified"] = candidate.provenance.get("groups_verified", False)
    if "notes" in plan.layers:
        for key in ("game", "raw_notes", "syllable_input", "game_input_padding_samples"):
            if key in candidate.provenance:
                merged.provenance[key] = candidate.provenance[key]
        merged.stale = [name for name in merged.stale if name != "notes"]
    merged.edited_layers = [name for name in original.edited_layers if name not in plan.layers]
    origins = merged.provenance.setdefault("layer_origins", {})
    for layer in plan.layers:
        origins[layer] = candidate.provenance.get("layer_origins", {}).get(layer, "candidate")
    merged.provenance["last_update"] = {"layers": list(plan.layers), "stages": plan.stages}
    merged.status = "annotated" if set(TIERS) <= merged.tiers.keys() else "partial"
    issues = validate(merged, profiles=profiles, ignore_stale=True)
    if any(i.severity == "error" for i in issues):
        return Outcome(original.sample, "conflict", candidate,
                       [Issue("invalid_merge", i.message, "update") for i in issues if i.severity == "error"])
    return Outcome(merged.sample, merged.status, merged, issues)
