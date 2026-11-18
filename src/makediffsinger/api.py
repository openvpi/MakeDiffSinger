"""Shared orchestration for scripts and agents, with lazy persistent model sessions."""

from dataclasses import asdict, replace

from .backends.session import GameSession, TifaSession
from .audio import audio_info
from .config import AlignmentOptions, NoteOptions, Resources, Runtime
from .importers import import_annotations, import_intervals, import_transcriptions
from .inputs import discover_pairs, inspect_dataset
from .phonology import PhoneProfiles, derive_syllables
from .reannotation import apply_annotation_updates, plan_reannotation
from .selection import QualityRule, Selection, select, select_quality, summarize
from .storage import load_annotations, read_annotations, save_result, write_annotations
from .types import Annotation, BatchResult, Issue, Outcome
from .validation import validate
from .exporters.diffsinger import export_diffsinger
from .slicing import plan_slices, replay_slices
from .review import make_review_pack
from .quality import quality_summary, write_overview


def _unique(samples):
    keys = [sample.key for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Sample identities must be unique within a call.")


class Pipeline:
    def __init__(self, resources: Resources | None = None, runtime: Runtime | None = None,
                 alignment: AlignmentOptions | None = None, notes: NoteOptions | None = None):
        self.resources, self.runtime = resources or Resources(), runtime or Runtime()
        self.alignment, self.notes = alignment or AlignmentOptions(), notes or NoteOptions()
        self._tifa = self._game = self._profiles = None

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = PhoneProfiles.from_resources(self.resources)
        return self._profiles

    @property
    def tifa(self):
        if self._tifa is None:
            self._tifa = TifaSession(self.resources, self.runtime, self.alignment)
        return self._tifa

    @property
    def game(self):
        if self._game is None:
            self._game = GameSession(self.resources, self.runtime, self.notes)
        return self._game

    def preflight(self, samples):
        samples = list(samples)
        _unique(samples)
        return self.tifa.preflight(samples) if samples else BatchResult([])

    def align(self, samples):
        samples = list(samples)
        _unique(samples)
        return self.tifa.align(samples) if samples else BatchResult([])

    def annotate_notes(self, annotations):
        annotations = list(annotations)
        _unique([a.sample for a in annotations])
        requests, problems = [], {}
        profiles = self.profiles if any("imported_ph_num" not in a.provenance for a in annotations) else PhoneProfiles()
        for annotation in annotations:
            try:
                syllables = derive_syllables(annotation, profiles)
                inputs = replace(annotation, tiers={k: v for k, v in annotation.tiers.items() if k != "notes"},
                                 stale=[name for name in annotation.stale if name != "notes"])
                issues = validate(inputs, required=("phones",), ignore_stale=True)
                if any(i.severity == "error" for i in issues):
                    raise ValueError("; ".join(i.message for i in issues if i.severity == "error"))
                provenance = dict(annotation.provenance)
                provenance["syllable_input"] = {
                    "method": "imported_ph_num" if "imported_ph_num" in annotation.provenance else "beat_phone",
                    "syllables": [asdict(syllable) for syllable in syllables],
                    "profile_sources": dict(profiles.sources),
                    "dictionary_sources": dict(profiles.dictionaries),
                    "global_phone_classes": dict(profiles.global_classes),
                    "inline_phone_classes": {lang: profiles.languages[lang] for lang, origin in profiles.sources.items()
                                             if origin == "explicit mapping"},
                }
                requests.append((replace(annotation, provenance=provenance), syllables))
            except ValueError as exc:
                problems[annotation.key] = Outcome(annotation.sample, "failed", annotation,
                                                   issues=[Issue("syllable_input", str(exc), "game")])
        if requests:
            originals = {a.key: a for a in annotations}
            for outcome in self.game.annotate(requests).outcomes:
                if outcome.annotation is not None:
                    issues = validate(outcome.annotation, required=("phones", "notes"), profiles=profiles)
                    outcome.issues.extend(issues)
                    if any(i.severity == "error" for i in issues):
                        outcome.status = "failed"
                if not outcome.ok and outcome.annotation is None:
                    outcome.annotation = originals[outcome.sample.key]
                problems[outcome.sample.key] = outcome
        return BatchResult([problems[a.key] for a in annotations])

    def run(self, samples):
        aligned = self.align(samples)
        noted = self.annotate_notes(aligned.annotations)
        outcomes = {o.sample.key: o for o in aligned.problems + noted.outcomes}
        return BatchResult([outcomes[o.sample.key] for o in aligned.outcomes])

    def extract_notes(self, samples):
        """Explicitly extract note-only assets without phoneme or syllable alignment."""
        samples = list(samples)
        _unique(samples)
        requests, outcomes = [], {}
        for sample in samples:
            try:
                duration = audio_info(sample.audio, require_mono=True)["clip_seconds"]
                annotation = Annotation(sample, duration, {}, provenance={
                    "syllable_input": {"method": "unconstrained", "syllables": []},
                })
                requests.append((annotation, None))
            except (OSError, ValueError, RuntimeError) as exc:
                outcomes[sample.key] = Outcome(sample, "failed", issues=[Issue("invalid_note_input", str(exc), "game")])
        if requests:
            for outcome in self.game.annotate(requests).outcomes:
                if outcome.annotation is not None:
                    issues = validate(outcome.annotation, required=("notes",), check_audio=True)
                    outcome.issues.extend(issues)
                    if any(i.severity == "error" for i in issues):
                        outcome.status = "failed"
                outcomes[outcome.sample.key] = outcome
        return BatchResult([outcomes[sample.key] for sample in samples])

    def reannotate(self, annotations, *, layers=("notes",), sample_keys=None, replace_edited=False):
        annotations = list(annotations)
        plan = plan_reannotation(annotations, layers=layers, sample_keys=sample_keys, replace_edited=replace_edited)
        originals = {a.key: a for a in annotations if a.key in plan.expected}
        candidates = list(originals.values())
        problems = {}
        if "tifa" in plan.stages:
            result = self.align([a.sample for a in candidates])
            candidates = result.annotations
            problems.update((o.sample.key, o) for o in result.problems)
        if "game" in plan.stages:
            result = self.annotate_notes(candidates)
            candidates = result.annotations
            problems.update((o.sample.key, o) for o in result.problems)
        for candidate in candidates:
            problems[candidate.key] = apply_annotation_updates(
                originals[candidate.key], candidate, plan, profiles=self.profiles)
        return BatchResult([problems[key] for key in originals])

    def close(self):
        for session in (self._tifa, self._game):
            if session is not None:
                session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
