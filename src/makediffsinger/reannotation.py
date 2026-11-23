"""Separate computation scope from permission to replace annotation layers."""

from copy import deepcopy
from dataclasses import dataclass, field

from .annotations import TIERS, tier_signatures
from .phonology import derive_syllables
from .timing import align_tiers
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
            merged.stale = [name for name in merged.stale if name != "quality"]
        else:
            merged.stale = sorted(set(merged.stale) | {"quality"})
        if set(plan.layers) & {"phones", "words"}:
            merged.provenance.pop("imported_ph_num", None)
            merged.provenance.pop("imported_phones_signature", None)
            merged.provenance.pop("imported_phone_labels", None)
        if {"phones", "words"} <= set(plan.layers):
            merged.provenance["groups_verified"] = candidate.provenance.get("groups_verified", False)
    if "notes" in plan.layers:
        if "game_grid" in candidate.provenance and not changed_fa:
            # Timing projection of the supplied tiers belongs to the notes result.
            merged.duration = candidate.duration
            for name, tier in original.tiers.items():
                if name == "notes":
                    continue
                projected = candidate.tiers.get(name, [])
                if [i.label for i in tier] != [i.label for i in projected]:
                    return Outcome(original.sample, "conflict", candidate, [
                        Issue("invalid_merge", f"Notes update would replace labels in {name}.", "update")])
                merged.tiers[name] = projected
            for key in ("imported_ph_num", "imported_phone_labels", "imported_phones_signature"):
                if key in candidate.provenance:
                    merged.provenance[key] = candidate.provenance[key]
                else:
                    merged.provenance.pop(key, None)
        for key in ("game", "game_grid", "raw_notes", "syllable_input", "game_input_padding_samples"):
            if key in candidate.provenance:
                merged.provenance[key] = candidate.provenance[key]
        merged.stale = [name for name in merged.stale if name != "notes"]
    merged.edited_layers = [name for name in original.edited_layers if name not in plan.layers]
    origins = merged.provenance.setdefault("layer_origins", {})
    for layer in plan.layers:
        origins[layer] = candidate.provenance.get("layer_origins", {}).get(layer, "candidate")
    merged.provenance["last_update"] = {"layers": list(plan.layers), "stages": plan.stages}
    merged.status = "annotated" if set(TIERS) <= merged.tiers.keys() else "partial"
    if "notes" in plan.layers and changed_fa and "game_grid" in candidate.provenance:
        grid = candidate.provenance["game_grid"]
        step = grid["hop_size"] / grid["sample_rate"]
        try:
            merged = align_tiers(merged, derive_syllables(merged, profiles), timestep=step,
                                 end_frame=round(merged.duration / step))
        except ValueError as exc:
            return Outcome(original.sample, "conflict", candidate, [Issue("invalid_merge", str(exc), "update")])
    issues = validate(merged, profiles=profiles, ignore_stale=True)
    if any(i.severity == "error" for i in issues):
        return Outcome(original.sample, "conflict", candidate,
                       [Issue("invalid_merge", i.message, "update") for i in issues if i.severity == "error"])
    return Outcome(merged.sample, merged.status, merged, issues)
