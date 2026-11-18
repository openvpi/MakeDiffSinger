"""Small inference adapters shared by the two isolated backend processes."""

from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess

import lightning.pytorch as pl
import torch


def model_record(backend, resources, runtime, model_path, **extra):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {backend} checkpoint: {path}")
    revision, dirty = "unknown", None
    try:
        result = subprocess.run(["git", "-C", str(resources.repo(backend)), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            revision = result.stdout.strip()
            status = subprocess.run(["git", "-C", str(resources.repo(backend)), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True)
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        pass
    return {"backend": backend, "revision": revision, "working_tree_dirty": dirty,
            "checkpoint": str(path), "checkpoint_bytes": path.stat().st_size,
            "checkpoint_mtime_ns": path.stat().st_mtime_ns,
            "runtime": asdict(runtime),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "lightning")},
            **extra}


def configure_precision(runtime):
    torch.backends.cuda.matmul.allow_tf32 = runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = runtime.allow_tf32


class ResultCollector(pl.Callback):
    def __init__(self, convert):
        self.convert = convert
        self.results = []

    def on_predict_batch_end(self, trainer, module, outputs, batch, *args, **kwargs):
        self.results.extend(self.convert(outputs, batch))


class MetadataTransferMixin:
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        payload = {key: value for key, value in batch.items() if key != "records"}
        moved = super().transfer_batch_to_device(payload, device, dataloader_idx)
        moved["records"] = batch["records"]
        return moved


