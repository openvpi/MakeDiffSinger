"""Shared delivery tables, indexed by paths relative to the delivery root."""

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import gettempdir, NamedTemporaryFile

from filelock import FileLock


STAGES = {".work", "01-clips", "02-lyrics", "03-annotations", "04-quality",
          "05-review-and-decisions", "06-diffsinger"}


def collection_root(directory):
    directory = Path(directory).resolve()
    for parent in (directory, *directory.parents):
        if (parent / ".metadata/dataset.json").is_file():
            return parent
        if parent.name in STAGES:
            return parent.parent
        if any((parent / ".metadata" / name).is_file() for name in ("annotations.json", "sources.json")):
            return parent
    return directory


def write_json(path, data):
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=16)
def _read(path, mtime, size):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _read(path, stat.st_mtime_ns, stat.st_size)


class Tables:
    def __init__(self, root):
        self.root = root
        self.changed = {}

    def set(self, path, key, value):
        if path not in self.changed:
            current = table(path)
            if value is None and key not in current:
                return
            self.changed[path] = current.copy()
        if value is None:
            self.changed[path].pop(key, None)
        else:
            self.changed[path][key] = value

    def update(self, path, key, fields):
        self.set(path, key, {**self.get(path, key), **fields})

    def get(self, path, key):
        rows = self.changed[path] if path in self.changed else table(path)
        return rows.get(key, {})

    def flush(self):
        for path, rows in self.changed.items():
            if rows:
                write_json(path, rows)
            else:
                path.unlink(missing_ok=True)


@contextmanager
def edit_tables(root):
    # Synchronize independent scripts without putting lock files in the delivery.
    name = sha256(os.path.normcase(str(root)).encode()).hexdigest()
    with FileLock(Path(gettempdir()) / f"makediffsinger-{name}.lock"):
        batch = Tables(root)
        try:
            yield batch
        finally:
            batch.flush()


def _local_bindings():
    from .config import repository_root
    return repository_root() / "local" / "dataset-roots.json"


def dataset_root(delivery):
    """Resolve the single original dataset associated with a delivery."""
    binding = table(delivery / ".metadata/dataset.json")
    if not binding:
        return None
    return resolve_locator(delivery, binding)


def resolve_locator(delivery, binding, *, local_key=None):
    if binding.get("relative_root") is not None:
        return (delivery / binding["relative_root"]).resolve()
    location = table(_local_bindings()).get(str(local_key or delivery))
    if not location:
        if local_key and (collection_root(delivery) / ".metadata/dataset.json").is_file():
            return dataset_root(collection_root(delivery))
        raise ValueError("Reconnect the original dataset with scripts/prepare/bind_dataset.py --relocate.")
    return Path(location)


def dataset_locator(delivery, source, *, local_key=None):
    try:
        relative = Path(os.path.relpath(source, delivery)).as_posix()
    except ValueError:  # Different Windows volumes have no relative spelling.
        local = _local_bindings()
        with edit_tables(local) as batch:
            batch.set(local, str(local_key or delivery), str(source))
        relative = None
    return {"name": source.name, "relative_root": relative}


def bind_dataset(delivery, source, *, relocate=False):
    """Bind an exact delivery directory. Keep cross-volume locators on this machine."""
    delivery, source = Path(delivery).resolve(), Path(source).resolve()
    with edit_tables(delivery):
        path = delivery / ".metadata/dataset.json"
        if path.exists() and not relocate:
            if dataset_root(delivery) != source:
                raise ValueError("This delivery belongs to another dataset. Choose a separate output directory.")
            return source
        if not source.is_dir():
            raise NotADirectoryError(source)
        write_json(path, dataset_locator(delivery, source))
    return source


def input_dataset(path):
    path = Path(path).resolve()
    directory = path.parent if path.is_file() else path
    return dataset_root(collection_root(directory)) or directory


def sample_dataset(samples):
    """Identify an explicit input boundary independently of language, singer or subset."""
    declared = {sample.metadata["dataset_root"] for sample in samples if sample.metadata.get("dataset_root")}
    roots = {Path(value).resolve() for value in declared}
    if len(roots) > 1:
        raise ValueError("Use separate deliveries and quality populations for different input datasets.")
    return next(iter(roots), None)


def path_reference(value, delivery, source):
    """Unqualified paths address the delivery; dataset: paths address original inputs."""
    if not value or not Path(value).is_absolute():
        return value
    path = Path(value)
    if source and source != delivery and source.is_relative_to(delivery) and path.is_relative_to(source):
        return "dataset:" + path.relative_to(source).as_posix()
    if path.is_relative_to(delivery):
        return path.relative_to(delivery).as_posix()
    if source and path.is_relative_to(source):
        return "dataset:" + path.relative_to(source).as_posix()
    # A separately supplied file may be alongside the original tree.
    try:
        if source and source.drive == path.drive:
            return "dataset:" + Path(os.path.relpath(path, source)).as_posix()
        return Path(os.path.relpath(path, delivery)).as_posix()
    except ValueError:
        # An explicitly supplied auxiliary file on a third volume has neither
        # relative base. Keep its actual location in the same local registry.
        local = _local_bindings()
        with edit_tables(local) as batch:
            key = "local-file:" + path.name
            serial = 1
            while table(local).get(key) not in (None, str(path)):
                serial += 1
                key = f"local-file:{serial}/{path.name}"
            batch.set(local, key, str(path))
        return key


def resolve_reference(value, delivery, source=None):
    if value.startswith("local-file:"):
        location = table(_local_bindings()).get(value)
        if not location:
            raise ValueError(f"Missing local binding for auxiliary file {value!r}.")
        return location
    if value.startswith("dataset:"):
        source = source or dataset_root(delivery)
        if source is None:
            raise ValueError("Missing original dataset binding.")
        return str((source / value.removeprefix("dataset:")).resolve())
    return str((delivery / value).resolve())


def provenance_paths(value, delivery, *, source=None, restore=False):
    """Tag paths in provenance so that ordinary source labels survive a round trip."""
    source = source or dataset_root(delivery)
    def convert(v):
        if isinstance(v, dict):
            if restore and set(v) == {"$file"}:
                return resolve_reference(v["$file"], delivery, source)
            return {k: convert(item) for k, item in v.items()}
        if isinstance(v, list):
            return [convert(item) for item in v]
        if not restore and isinstance(v, str) and Path(v).is_absolute():
            return {"$file": path_reference(v, delivery, source)}
        return v
    return convert(value)


def portable_message(message, delivery, source):
    roots = {delivery: "", **({source: "dataset:"} if source and source != delivery else {})}
    for base in sorted(roots, key=lambda p: len(str(p)), reverse=True):
        if message == str(base) or message == base.as_posix():
            return roots[base] + "."
        for prefix in {str(base).rstrip("\\/") + os.sep, base.as_posix().rstrip("/") + "/"}:
            message = message.replace(prefix, roots[base])
    return message


def store_origins(directory, records, *, source_root=None):
    root = collection_root(directory)
    records = list(records)
    if not records:
        return
    source = (Path(source_root).resolve() if source_root else dataset_root(root)
              or Path(records[0][1]["origin_recording"]["path"]).resolve().parent)
    for audio, record in records:
        if not Path(audio).resolve().is_relative_to(root) or not Path(record["origin_recording"]["path"]).resolve().is_relative_to(source):
            raise ValueError("The clip and original recording must belong to this delivery and its input dataset.")
    bind_dataset(root, source)
    path = root / ".metadata/sources.json"
    with edit_tables(root) as batch:
        for audio, record in records:
            key = Path(audio).resolve().relative_to(root).as_posix()
            recording = record["origin_recording"]
            original = Path(recording["path"]).resolve().relative_to(source).as_posix()
            batch.set(root / ".metadata/recordings.json", original, dict(recording, path=original))
            region = {k: v for k, v in record["origin_slice"].items() if k in ("id", "start_sample", "end_sample", "channel")}
            batch.update(path, key, {"recording": original, "slice": region})


def association_for(audio):
    audio = Path(audio).resolve()
    root = collection_root(audio.parent)
    return root, table(root / ".metadata/sources.json").get(audio.relative_to(root).as_posix(), {})


def source_for(audio):
    root, row = association_for(audio)
    if row and "recording" in row:
        recording = deepcopy(table(root / ".metadata/recordings.json")[row["recording"]])
        if not Path(recording["path"]).is_absolute():
            recording["path"] = str(dataset_root(root) / recording["path"])
        return {"format": "makediffsinger.slice", "origin_recording": recording,
                "origin_slice": dict(row["slice"])}
    return None
