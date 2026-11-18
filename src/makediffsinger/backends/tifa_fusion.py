"""Fuse lexical pronunciation paths before TIFA's forced-alignment pass."""

from dataclasses import dataclass
from itertools import chain, product
import math
from xml.sax.saxutils import escape, quoteattr

from g2pflow import G2PConversionError, PFMLError
from inference.scoring import prepare_scoring, score_fragments, select_scored_paths
from lib.g2p_encoding import G2PEncodingError, _pack_paths, encode_paths
from lib.path_traversal import first_choices
import lightning.pytorch as pl
import torch

from ..audio import load_audio
from ..lyric_pronunciation import LyricUnit, PhoneGraph, pronunciation_blocks
from .tifa_source_scores import block_scores


@dataclass(frozen=True)
class Path:
    tokens: tuple
    groups: tuple


@dataclass(frozen=True)
class Word:
    text: str
    paths: tuple[Path, ...]


def _words(adapter, text, languages, *, recognition=False, pfml=False):
    convert = adapter.g2p.convert_pfml if pfml else adapter.g2p.convert
    words = convert(text, languages=languages)
    numeric, lexicon, texts = encode_paths(
        words, adapter.vocabulary, "force" if recognition else adapter.options.oov_handling,
        languages=languages, stop_symbols=adapter.events,
    )
    result = []
    for index, (label, readings) in enumerate(zip(texts, lexicon, strict=True), 1):
        rows = numeric["words"] == index
        paths = []
        for column, reading in enumerate(readings):
            tokens = numeric["paths"][rows, column]
            ids = numeric["groups"][rows, column][tokens != 0].tolist()
            groups = [[] for _ in reading["scripts"]]
            for group, phone in zip(ids, reading["phonemes"], strict=True):
                groups[group - 1].append(phone)
            if ids:
                paths.append(Path(tuple(tokens[tokens != 0].tolist()), tuple(
                    (script, tuple(phones)) for script, phones in zip(reading["scripts"], groups, strict=True)
                )))
        if paths:
            result.append(Word(label, tuple(dict.fromkeys(paths))))
        elif not recognition:
            raise G2PEncodingError(f"No encodable pronunciation for {label!r}.")
    return result


def _graph(words):
    return PhoneGraph.from_units([
        LyricUnit(1, i, i + 1, (tuple(dict.fromkeys(path.tokens for path in word.paths)),))
        for i, word in enumerate(words)
    ])


def _alternatives(reference, recognition, spans=None):
    blocks = []
    if spans is None:
        spans = pronunciation_blocks(_graph(reference), _graph(recognition))
    for left, right in spans:
        alternatives = {}
        for source, words in (("reference", reference[left]), ("asr", recognition[right])):
            for choice in product(*(word.paths for word in words)):
                tokens = tuple(chain.from_iterable(path.tokens for path in choice))
                paths = tuple(Word(word.text, (path,)) for word, path in zip(words, choice, strict=True))
                alternatives.setdefault(tokens, {}).setdefault(source, paths)
        blocks.append(alternatives)
    return blocks


def _score(backend, spec, blocks, score_unit):
    numeric = _pack_paths([[{"tokens": list(tokens), "groups": [1] * len(tokens)}
                            for tokens in block] for block in blocks])
    paths = torch.from_numpy(numeric["paths"]).unsqueeze(0).to(spec.features.device)
    words = torch.from_numpy(numeric["words"]).unsqueeze(0).to(paths.device)
    candidates = torch.from_numpy(numeric["candidates"]).unsqueeze(0).to(paths.device)
    template, segments, mapping = prepare_scoring(paths, words, candidates, unit=score_unit)
    masked_slots = int((segments > 0).sum())
    if not masked_slots:
        return candidates, None, None, masked_slots

    _, _, _, logits = backend.model(spec.features, template, spec.mask, template != 0)
    log_probs = logits.float().log_softmax(-1)
    fragments = score_fragments(log_probs, paths, words, segments, mapping)
    return candidates, fragments, logits.shape[-1], masked_slots


def _shortlist(backend, spec, reference, recognition, score_unit, min_score, top_k):
    spans = pronunciation_blocks(_graph(reference), _graph(recognition))
    blocks = _alternatives(reference, recognition, spans)
    by_source = {}
    for source, words, index in (("reference", reference, 0), ("asr", recognition, 1)):
        paths = [tuple(dict.fromkeys(path.tokens for path in word.paths)) for word in words]
        _, fragments, vocab, _ = _score(backend, spec, paths, score_unit)
        by_source[source] = block_scores(paths, [pair[index] for pair in spans], fragments, vocab)

    retained, counts = [], []
    for block_index, block in enumerate(blocks):
        keep = {}
        for source, scores in by_source.items():
            values = scores[block_index]
            ranked = sorted((tokens for tokens in block if source in block[tokens]),
                            key=lambda tokens: -values.get(tokens, -math.inf))
            if not ranked or not math.isfinite(values.get(ranked[0], -math.inf)):
                raise ValueError(f"No complete {source} pronunciation fits the scoring template.")
            keep[source] = {tokens for tokens in ranked[:top_k]
                            if values.get(tokens, -math.inf) >= min_score} or {ranked[0]}
        # Preserve the original path order for template construction and ties.
        limited = {}
        for tokens, origins in block.items():
            surviving = {source: reading for source, reading in origins.items() if tokens in keep[source]}
            if surviving:
                limited[tokens] = surviving
        retained.append(limited)
        counts.append({source: len(tokens) for source, tokens in keep.items()})
    return retained, counts


def _joint_choices(backend, spec, blocks, score_unit):
    candidates, fragments, vocab, masked_slots = _score(backend, spec, blocks, score_unit)
    if fragments is None:
        return first_choices(candidates)[0].tolist(), None, masked_slots

    selected, scores = select_scored_paths(candidates, *fragments, vocab_size=vocab)
    best = scores.gather(2, (selected - 1).unsqueeze(-1)).max().item()
    runner_up = scores.scatter(2, (selected - 1).unsqueeze(-1), -torch.inf).max().item()
    margin = max(0.0, best - runner_up) if math.isfinite(runner_up) else None
    return selected[0].tolist(), margin, masked_slots


def _fixed_word(word):
    groups = "".join(f"<group script={quoteattr(script)} phonemes={quoteattr(' '.join(phones))}/>"
                     for script, phones in word.paths[0].groups)
    return f"<word text={quoteattr(word.text)}><reading>{groups}</reading></word>"


def _render(adapter, selected, languages):
    # Plain fragments must round-trip independently when surrounded by PFML.
    def fragments(words):
        if not words:
            return []
        text = " ".join(word.text for word in words)
        try:
            natural = _words(adapter, text, languages)
        except (G2PConversionError, G2PEncodingError, PFMLError):
            natural = []
        if natural == words:
            return [(text, False)]
        start = 0
        while start < min(len(natural), len(words)) and words[start] == natural[start]:
            start += 1
        bound = min(start, len(words) - 1)
        return [*fragments(words[:bound]), (_fixed_word(words[bound]), True), *fragments(words[bound + 1:])]

    parts = fragments(selected)
    if any(fixed for _, fixed in parts):
        content = " ".join(text if fixed else escape(text) for text, fixed in parts)
        if _words(adapter, content, languages, pfml=True) != selected:
            raise ValueError("Selected pronunciation could not be preserved in PFML.")
        return {"content": content, "format": "pfml"}
    return {"content": parts[0][0], "format": "text"}


class LyricFusion(pl.LightningModule):
    def __init__(self, backend, adapter):
        super().__init__()
        self.backend = backend
        self.adapter = adapter

    def predict_step(self, request, _batch_idx):
        sample, reference_text, asr_text, languages, min_score, top_k = request
        try:
            reference = _words(self.adapter, reference_text, languages)
            recognition = _words(self.adapter, asr_text, languages, recognition=True)
            if not reference or not recognition:
                raise ValueError("Fusion needs encodable reference and ASR pronunciations.")
            waveform, rate = load_audio(sample.audio, self.backend.sample_rate)
            spec = self.backend.spectrogram(torch.as_tensor(waveform, device=self.device, dtype=torch.float32)[None],
                                            torch.tensor([len(waveform) / rate], device=self.device))
            score_unit = self.adapter.options.score_unit
            blocks, counts = _shortlist(self.backend, spec, reference, recognition, score_unit, min_score, top_k)
            selected, margin, masked = _joint_choices(self.backend, spec, blocks, score_unit)
            choices = [list(block.values())[index - 1] for block, index in zip(blocks, selected, strict=True)]
            # Identical phone paths share a score. Keep the reference spelling when it exists.
            winners = [next(iter(choice.items())) for choice in choices]
            words = list(chain.from_iterable(reading for _, reading in winners))
            if not words:
                raise ValueError("The selected path contains no lyrics.")
            return {"status": "scored", "texts": [word.text for word in words],
                    "transcript": _render(self.adapter, words, languages), "margin": margin,
                    "masked_slots": masked,
                    "blocks": [{"source": source, "texts": [word.text for word in reading],
                                "shortlist_counts": count,
                                "groups": [list(group) for word in reading for group in word.paths[0].groups]}
                               for (source, reading), count in zip(winners, counts, strict=True)]}
        except (G2PConversionError, G2PEncodingError, PFMLError, OSError, RuntimeError, ValueError) as exc:
            return {"status": "unavailable", "reason": str(exc)}
        except KeyError as exc:
            message = exc.args[0] if len(exc.args) == 1 else None
            if not isinstance(message, str) or not (
                message.startswith("Script token ") and message.endswith(" not found in script-to-phoneme dict.")
            ):
                raise
            return {"status": "unavailable", "reason": message}
