"""Language-bound phone classes and the shared beat-bearing-phone rules."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import yaml

from .annotations import tier_signatures
from .timing import match_boundaries, tier_edges, tolerance
from .types import Annotation, Syllable


RESOURCE_DIR = Path(__file__).parent / "resources"


def _catalog():
    return json.loads((RESOURCE_DIR / "phone-profiles.json").read_text(encoding="utf-8"))


def _validated_classes(classes, origin):
    if not isinstance(classes, dict) or any(
        not isinstance(phone, str) or not phone or not isinstance(kind, str)
        or kind not in {"consonant", "medial", "vowel"}
        for phone, kind in classes.items()
    ):
        raise ValueError(f"Invalid phone-type record: {origin}.")
    return dict(classes)


def _read_classes(value):
    if isinstance(value, str):
        classes = json.loads(Path(value).read_text(encoding="utf-8"))
        return _validated_classes(classes, value), value
    return _validated_classes(value, "explicit mapping"), "explicit mapping"


def _dictionary_phones(path):
    """Read the inventory only; pronunciation positions never assign types."""
    phones = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        _, separator, pronunciation = line.strip().partition("\t")
        if separator and pronunciation:
            phones.update(pronunciation.split())
    if not phones:
        raise ValueError(f"No phone inventory found in tab-separated dictionary {path}.")
    return phones


@dataclass
class PhoneProfiles:
    languages: dict[str, dict[str, str]] = field(default_factory=dict)
    sources: dict[str, str | list[str]] = field(default_factory=dict)
    global_classes: dict[str, str] = field(default_factory=lambda: _catalog()["global_classes"])
    dictionaries: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.global_classes = _validated_classes(self.global_classes, "global classes")
        self.languages = {language: _validated_classes(classes, language) for language, classes in self.languages.items()}

    def kind(self, phone: str, language: str | None) -> str:
        if "/" in phone:
            language, phone = phone.split("/", 1)
        if phone in self.global_classes:
            return self.global_classes[phone]
        if language not in self.languages or phone not in self.languages[language]:
            raise ValueError(f"No phone class for {language}/{phone}; supply the language and dictionary profile.")
        return self.languages[language][phone]

    @classmethod
    def from_resources(cls, resources):
        catalog = _catalog()
        dictionaries = {language: [Path(path).resolve()] for language, path in resources.dictionaries.items()}
        if resources.g2p_config:
            path = Path(resources.g2p_config)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            model_root = Path(resources.tifa_model).parent if resources.tifa_model else None
            base = path.parent if model_root and path.is_relative_to(model_root) else Path.cwd()
        elif resources.tifa_model:
            base = Path(resources.tifa_model).parent
            config_path = base / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")).get("inference", {}).get("g2p", {})
        else:
            config, base = {}, Path.cwd()
        for converter in (config or {}).get("converters", []):
            raw = converter.get("kwargs", {}).get("dict_path")
            if not raw:
                continue
            dictionary = (base / raw[1:]).resolve() if raw.startswith("@") else Path(raw).resolve()
            registration = catalog["dictionaries"].get(dictionary.name, {})
            language = (converter.get("language") or catalog["converter_languages"].get(converter["id"])
                        or registration.get("language"))
            if language is None:
                language = next((lang for lang, paths in dictionaries.items() if dictionary in paths), None)
            if language is None:
                raise ValueError(f"Dictionary language is unspecified for {dictionary}; declare it in G2P or resources.dictionaries.")
            if language not in resources.dictionaries:
                paths = dictionaries.setdefault(language, [])
                if dictionary not in paths:
                    paths.append(dictionary)
        result = cls(global_classes=catalog["global_classes"])
        for language in dictionaries.keys() | resources.phone_classes.keys():
            explicit = resources.phone_classes.get(language)
            paths = dictionaries.get(language, [])
            origins = []
            for dictionary in paths or [None]:
                if explicit is not None:
                    classes, origin = _read_classes(explicit)
                else:
                    registration = catalog["dictionaries"].get(dictionary.name) if dictionary is not None else None
                    if registration is None or registration["language"] != language:
                        raise ValueError(f"Missing phone-type record for {language} dictionary {dictionary}; supply phone_classes.")
                    classes, origin = _read_classes(str(RESOURCE_DIR / registration["classes"]))
                if dictionary is not None:
                    missing = _dictionary_phones(dictionary) - classes.keys() - result.global_classes.keys()
                    if missing:
                        raise ValueError(f"Missing phone types in {origin} for {dictionary}: {', '.join(sorted(missing))}")
                merged = result.languages.setdefault(language, {})
                conflicts = [phone for phone, kind in classes.items()
                             if (phone in merged and merged[phone] != kind)
                             or (phone in result.global_classes and result.global_classes[phone] != kind)]
                if conflicts:
                    raise ValueError(f"Conflicting phone types for {language}: {', '.join(sorted(conflicts))}")
                merged.update(classes)
                if origin not in origins:
                    origins.append(origin)
            result.sources[language] = origins[0] if len(origins) == 1 else origins
            result.dictionaries[language] = [str(path) for path in paths]
        return result


def group_onsets(kinds: list[str]) -> list[int]:
    onsets, index = [], 0
    while index < len(kinds):
        if kinds[index:index + 3] in (["consonant", "medial", "vowel"], ["medial", "medial", "vowel"]):
            onsets.append(index + 1)
            index += 3
        elif kinds[index] == "vowel":
            onsets.append(index)
            index += 1
        else:
            index += 1
    return onsets or ([0] if kinds else [])


def syllables_from_counts(annotation: Annotation, counts: list[int]) -> list[Syllable]:
    phones = annotation.tiers["phones"]
    if not counts or any(n < 1 for n in counts) or sum(counts) != len(phones):
        raise ValueError("ph_num must contain positive counts covering every phone exactly once.")
    result, cursor = [], 0
    for count in counts:
        result.append(Syllable(cursor, count, phones[cursor].start, phones[cursor + count - 1].end))
        cursor += count
    return result


def current_phone_counts(annotation):
    counts = annotation.provenance.get("imported_ph_num")
    if counts is None:
        return None
    phones = annotation.tiers.get("phones", [])
    labels = annotation.provenance.get("imported_phone_labels")
    if labels is not None:
        return counts if labels == [p.label for p in phones] else None
    if tier_signatures({"phones": phones})["phones"] == annotation.provenance.get("imported_phones_signature"):
        return counts
    return None


def derive_syllables(annotation: Annotation, profiles: PhoneProfiles | None = None) -> list[Syllable]:
    phones = annotation.tiers.get("phones", [])
    if not phones:
        raise ValueError("A nonempty phones layer is required.")
    imported = current_phone_counts(annotation)
    if imported is not None:
        return syllables_from_counts(annotation, imported)
    if "imported_ph_num" in annotation.provenance and not (profiles and annotation.provenance.get("groups_verified")):
        raise ValueError("Imported ph_num is stale after phone edits.")
    if profiles is None:
        raise ValueError("Phone profiles are required to derive syllables from FA groups.")
    if annotation.provenance.get("groups_verified") is False:
        raise ValueError("The source words tier has not been identified as pronunciation groups.")
    groups = annotation.tiers.get("words")
    if not groups:
        raise ValueError("Pronunciation groups are missing; provide verified groups or source ph_num.")
    ms = tolerance(annotation)
    indices = match_boundaries(tier_edges(groups, annotation.duration, ms), tier_edges(phones, annotation.duration, ms), ms)
    onsets = []
    for start, cursor in zip(indices, indices[1:]):
        kinds = [profiles.kind(p.label, annotation.sample.language) for p in phones[start:cursor]]
        onsets.extend(start + i for i in group_onsets(kinds))
    prefix = onsets[0] != 0
    if prefix:
        onsets.insert(0, 0)
    ends = onsets[1:] + [len(phones)]
    return [Syllable(start, end - start, phones[start].start, phones[end - 1].end, prefix and i == 0)
            for i, (start, end) in enumerate(zip(onsets, ends))]
