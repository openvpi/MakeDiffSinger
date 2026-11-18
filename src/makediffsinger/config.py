"""Portable resource references and explicit runtime options."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import re


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "external").is_dir():
            return parent
    raise FileNotFoundError("Use an editable checkout or supply explicit TIFA and GAME repository paths.")


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("Select an explicit device: cpu, cuda, or cuda:N.")
    return device


def resolve_device(device: str | None = None, *, state_path: Path | None = None) -> str:
    if device is not None:
        return _validate_device(device)
    try:
        path = state_path if state_path is not None else repository_root() / "local/agent-setup.json"
        setup = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        setup = {}
    if not isinstance(setup, dict):
        raise ValueError("Agent setup must be a JSON object.")
    if setup.get("device") is None:
        raise ValueError("Specify a device or set device in local/agent-setup.json.")
    return _validate_device(setup["device"])


def is_device_unsupported(device: str | None, error: Exception) -> bool:
    if not device or not device.startswith("cuda"):
        return False
    message = str(error).lower()
    return ("no kernel image is available for execution on the device" in message
            or "does not support bfloat16" in message
            or bool(re.search(r"(?:not implemented|not supported) (?:for|on) (?:the )?['\"]?cuda\b", message))
            or bool(re.search(r"with arguments from the 'cuda' backend", message)))


@dataclass
class Resources:
    tifa_model: str | None = None
    ved_model: str | None = None
    game_model: str | None = None
    g2p_config: str | None = None
    tifa_repo: str | None = None
    game_repo: str | None = None
    phone_classes: dict[str, str | dict] = field(default_factory=dict)
    dictionaries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("tifa_model", "ved_model", "game_model", "g2p_config", "tifa_repo", "game_repo"):
            value = getattr(self, name)
            if value:
                setattr(self, name, str(Path(value).resolve()))

    @classmethod
    def read(cls, path: str | Path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Resource configuration must be an object.")
        unknown = data.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown resource fields: {', '.join(sorted(unknown))}. "
                             "Environment choices belong outside the resource file.")
        for name in ("tifa_model", "ved_model", "game_model", "g2p_config", "tifa_repo", "game_repo"):
            if data.get(name):
                data[name] = str((path.parent / data[name]).resolve())
        for name in ("dictionaries", "phone_classes"):
            data[name] = {k: str((path.parent / v).resolve()) if isinstance(v, str) else v
                          for k, v in data.get(name, {}).items()}
        return cls(**data)

    def repo(self, backend: str) -> Path:
        value = getattr(self, f"{backend}_repo")
        return Path(value) if value else repository_root() / "external" / backend.upper()


@dataclass
class Runtime:
    device: str | None = None
    precision: str = "32-true"
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 42
    allow_tf32: bool = False

    def __post_init__(self):
        if self.device is not None:
            _validate_device(self.device)
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative.")

    def trainer_options(self) -> dict:
        device = resolve_device(self.device)
        options = {"precision": self.precision, "logger": False, "enable_checkpointing": False,
                   "enable_progress_bar": False, "enable_model_summary": False}
        if device.startswith("cuda:"):
            options.update(accelerator="gpu", devices=[int(device.split(":")[1])])
        else:
            options.update(accelerator="gpu" if device == "cuda" else device, devices=[0] if device == "cuda" else 1)
        return options


@dataclass
class AlignmentOptions:
    events: tuple[str, ...] = ("AP",)
    oov_handling: str = "raise"
    score_unit: str = "levenshtein"
    skip_penalty: float = 0.5


@dataclass
class NoteOptions:
    language: str | None = None
    boundary_threshold: float = 0.2
    boundary_radius_seconds: float = 0.02
    presence_threshold: float = 0.2
    t0: float = 0.0
    steps: int = 8
    timesteps: list[float] | None = None


def config_dict(value) -> dict:
    return asdict(value)
