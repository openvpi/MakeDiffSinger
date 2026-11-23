"""GAME adapter preserving raw pitch/presence decisions and model frame positions."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import librosa
import lightning.pytorch as pl
import torch

from inference.api import load_inference_model
from inference.data import DiffSingerTranscriptionsDataset
from inference.me_infer_module import InferenceModule
from lib.config.schema import ValidationConfig

from ..config import is_device_unsupported
from ..timing import align_tiers
from ..types import BatchResult, Interval, Issue, Outcome, Syllable
from .common import MetadataTransferMixin, ResultCollector, configure_precision, model_record
from .game_input import prepare_input


class GameInputs(torch.utils.data.Dataset):
    def __init__(self, requests, features, language):
        self.requests, self.features, self.language = requests, features, language

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        annotation, syllables = self.requests[index]
        try:
            annotation, durations, waveform, padding, issues = prepare_input(annotation, syllables, self.features)
        except (OSError, ValueError, RuntimeError) as exc:
            return {"skip": True, "annotation": annotation,
                    "issues": [Issue("invalid_note_input", str(exc), "game")]}
        waveform = torch.from_numpy(waveform).float()
        return {"skip": False, "annotation": annotation, "issues": issues,
                "input_padding_samples": padding,
                "index": "memory", "name": annotation.key, "waveform": waveform,
                "samplerate": self.features.audio_sample_rate,
                "known_durations": torch.tensor(durations, dtype=torch.float32),
                "language": self.language}

    @staticmethod
    def collate(items):
        valid = [i for i in items if not i["skip"]]
        batch = DiffSingerTranscriptionsDataset.collate(valid) if valid else {}
        batch["records"] = [{"annotation": i["annotation"], "skip": i["skip"], "issues": i["issues"],
                             "input_padding_samples": i.get("input_padding_samples", 0)} for i in items]
        return batch


class SafeInferenceModule(MetadataTransferMixin, InferenceModule):
    def predict_step(self, batch, *args, **kwargs):
        if "waveform" not in batch:
            return {}
        return super().predict_step(batch, *args, **kwargs)


class GameAdapter:
    def __init__(self, resources, runtime, options):
        if not resources.game_model:
            raise ValueError("A GAME model checkpoint is required.")
        self.resources, self.runtime, self.options = resources, runtime, options
        configure_precision(runtime)
        self.info = model_record("game", resources, runtime, resources.game_model, options=asdict(options))
        pl.seed_everything(runtime.seed, workers=True)
        self.model, languages = load_inference_model(Path(resources.game_model))
        if options.language and languages and options.language not in languages:
            raise ValueError(f"Unsupported GAME language: {options.language}")
        self.language = languages[options.language] if options.language and languages else 0
        self.info.update(sample_rate=self.model.inference_config.features.audio_sample_rate,
                         hop_size=self.model.inference_config.features.hop_size, language_id=self.language)
        config = ValidationConfig(
            d3pm_sample_t0=options.t0, d3pm_sample_steps=options.steps, d3pm_sample_ts=options.timesteps,
            boundary_decoding_threshold=options.boundary_threshold,
            boundary_decoding_radius=round(options.boundary_radius_ms / (1000 * self.model.timestep)),
            note_presence_threshold=options.presence_threshold,
        )
        self.module = SafeInferenceModule(model=self.model, config=config)
        self.collector = ResultCollector(self._convert)
        self.trainer = pl.Trainer(**runtime.trainer_options(), callbacks=[self.collector])

    def _predict(self, requests):
        self.collector.results = []
        dataset = GameInputs(requests, self.model.inference_config.features, self.language)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.runtime.batch_size,
                                            num_workers=self.runtime.num_workers, shuffle=False,
                                            collate_fn=GameInputs.collate)
        self.trainer.predict(self.module, dataloaders=loader, return_predictions=False)
        return self.collector.results

    def annotate(self, requests):
        pl.seed_everything(self.runtime.seed, workers=True)
        try:
            results = self._predict(requests)
        except (RuntimeError, ValueError) as exc:
            if is_device_unsupported(self.runtime.device, exc):
                raise
            results = list(self.collector.results)
            finished = {o.sample.key for o in results}
            for request in requests:
                if request[0].key in finished:
                    continue
                if self.runtime.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                self.trainer = pl.Trainer(**self.runtime.trainer_options(), callbacks=[self.collector])
                try:
                    results.extend(self._predict([request]))
                except (RuntimeError, ValueError) as exc:
                    if is_device_unsupported(self.runtime.device, exc):
                        raise
                    results.append(Outcome(request[0].sample, "failed", issues=[Issue("inference_failed", str(exc), "game")]))
        by_key = {o.sample.key: o for o in results}
        return BatchResult([by_key[a.key] for a, _ in requests])

    def _convert(self, outputs, batch):
        records = {r["annotation"].key: r for r in batch["records"]}
        results = [Outcome(r["annotation"].sample, "failed", issues=r["issues"])
                   for r in batch["records"] if r["skip"]]
        for row, key in enumerate(batch.get("name", [])):
            annotation = deepcopy(records[key]["annotation"])
            valid = outputs["durations"][row] > 0
            durations = outputs["durations"][row][valid].detach().cpu().tolist()
            scores = outputs["scores"][row][valid].detach().cpu().tolist()
            presence = outputs["presence"][row][valid].detach().cpu().tolist()
            timestep = self.model.timestep
            frames = [round(duration / timestep) for duration in durations]
            notes, cursor = [], 0
            for count, pitch, voiced in zip(frames, scores, presence):
                label = librosa.midi_to_note(pitch, unicode=False, cents=True) if voiced else "rest"
                notes.append(Interval(cursor * timestep, (cursor + count) * timestep, label))
                cursor += count
            annotation.tiers["notes"] = notes
            issues = list(records[key]["issues"])
            if annotation.provenance.get("syllable_input", {}).get("method") == "unconstrained":
                annotation.duration = cursor * timestep
                issues.append(Issue("unconstrained_notes", "No syllable boundaries were supplied; phoneme alignment is not guaranteed.",
                                    "game", "warning"))
            annotation.provenance["game"] = self.info
            annotation.provenance["game_input_padding_samples"] = records[key]["input_padding_samples"]
            annotation.provenance["raw_notes"] = {"durations": durations, "midi": scores,
                                                  "presence": presence, "frames": frames,
                                                  "time_basis": "model_frames"}
            if annotation.provenance["syllable_input"]["method"] != "unconstrained":
                syllables = [Syllable(**row) for row in annotation.provenance["syllable_input"]["syllables"]]
                try:
                    annotation = align_tiers(annotation, syllables, timestep=timestep,
                                             end_frame=round(annotation.duration / timestep))
                except ValueError as exc:
                    results.append(Outcome(annotation.sample, "failed", annotation,
                                           [*issues, Issue("note_alignment", str(exc), "game")]))
                    continue
            annotation.provenance.setdefault("layer_origins", {})["notes"] = "game"
            annotation.stale = [name for name in annotation.stale if name not in {"notes", "export"}]
            annotation.edited_layers = [name for name in annotation.edited_layers if name != "notes"]
            annotation.status = "annotated" if all(name in annotation.tiers for name in ("texts", "words", "phones")) else "partial"
            results.append(Outcome(annotation.sample, annotation.status, annotation, issues))
        return results
