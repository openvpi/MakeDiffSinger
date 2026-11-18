"""Inspect the current interpreter and resources; optionally remember Agent setup."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def git_output(directory, *args):
    try:
        result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_device(device):
    import torch

    probe = torch.ones(1, device=device) + 1
    if probe.item() != 2:
        raise RuntimeError("The device calculation returned an unexpected result.")
    return {"selected": str(probe.device), "torch": torch.__version__,
            "name": torch.cuda.get_device_name(probe.device) if probe.is_cuda else "CPU"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--state", type=Path, help="Ignored Agent state file; defaults to local/agent-setup.json in the checkout.")
    parser.add_argument("--require", choices=("core", "tifa", "game", "all"), default="core")
    parser.add_argument("--check-device", action="store_true", help="Run a small calculation in the selected environment.")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N. Otherwise uses the saved device.")
    parser.add_argument("--remember", action="store_true", help="Save successful choices in the separate ignored Agent state file.")
    parser.add_argument("--environment", help="User-supplied environment name; required with --remember.")
    args = parser.parse_args()
    if args.remember and not args.environment:
        parser.error("--remember requires --environment.")
    spec = importlib.util.find_spec("makediffsinger")
    candidates = [args.tool_root] if args.tool_root else [
        *(Path(spec.origin).resolve().parents if spec and spec.origin else []),
        *Path(__file__).resolve().parents,
    ]
    root = next((p.resolve() for p in candidates if p is not None
                 and (p / "pyproject.toml").is_file() and (p / "src" / "makediffsinger").is_dir()), None)
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    matches = origin.is_relative_to(root / "src") if origin and root else None
    result = {"python": sys.executable, "core_installed": spec is not None, "core_origin": str(origin) if origin else None,
              "tool_root": str(root) if root else None, "checkout_matches_import": matches,
              "packages": {}, "repositories": {}, "models": {}, "problems": [], "next_steps": []}
    state_path = args.state.resolve() if args.state else (root / "local/agent-setup.json" if root else None)
    result["state_file"] = str(state_path) if state_path else None
    setup = {}
    if state_path and state_path.exists():
        try:
            setup = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(setup, dict):
                raise ValueError("Expected an object.")
        except (OSError, ValueError) as exc:
            result["problems"].append(f"Cannot read Agent setup: {exc}")
            setup = {}
    result["saved_setup"] = setup
    core_packages = ("makediffsinger", "numpy", "soundfile", "textgrid", "PyYAML")
    inference_packages = ("torch", "lightning", "librosa", "g2pflow")
    for name in (*core_packages, *inference_packages):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if args.tool_root and root is None:
        result["problems"].append("The supplied tool root is not a MakeDiffSinger checkout.")
    if matches is False:
        result["problems"].append("The selected interpreter imports the core from a different checkout.")
    missing = [name for name in core_packages if result["packages"][name] is None]
    if missing:
        result["problems"].append("Missing core packages: " + ", ".join(missing))
        result["next_steps"].append("Install this checkout into the selected interpreter, or select an installed environment.")
    path = args.resources.resolve() if args.resources else None
    if path is None and root and (root / "configs/resources.local.json").is_file():
        path = root / "configs/resources.local.json"
    result["resource_file"] = str(path) if path else None
    if path and state_path == path:
        result["problems"].append("Agent state and resource configuration must use separate files.")
    config = {}
    if path:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Expected a JSON object.")
            for key in ("tifa_repo", "game_repo", "tifa_model", "ved_model", "game_model"):
                if config.get(key) and not isinstance(config[key], str):
                    raise ValueError(f"{key} must be a path string.")
        except (OSError, ValueError) as exc:
            result["problems"].append(f"Cannot read resource configuration: {exc}")
            config = {}
    base = path.parent if path else (root or Path.cwd())
    for backend in ("tifa", "game"):
        value = config.get(backend + "_repo")
        repo = (base / value).resolve() if value else (root / "external" / backend.upper() if root else None)
        result["repositories"][backend.upper()] = {
            "path": str(repo) if repo else None,
            "api_present": bool(repo and (repo / "inference/api.py").is_file()),
        }
        entry = result["repositories"][backend.upper()]
        entry["revision"] = git_output(repo, "rev-parse", "HEAD") if entry["api_present"] else None
        entry["working_tree"] = git_output(repo, "status", "--porcelain") if entry["api_present"] else None
        pin = git_output(root, "ls-files", "--stage", "--", f"external/{backend.upper()}") if root else None
        entry["pinned_revision"] = pin.split()[1] if pin and pin.startswith("160000 ") else None
        entry["matches_pin"] = (entry["revision"] == entry["pinned_revision"]
                                if not value and entry["pinned_revision"] else None)
    result["submodules"] = {name: info["api_present"] for name, info in result["repositories"].items()}
    for name in ("tifa", "ved", "game"):
        value = config.get(name + "_model")
        checkpoint = (base / value).resolve() if value else None
        required = ("config.yaml", "vocabulary.json") if name != "game" else ("config.yaml",)
        result["models"][name] = {
            "path": str(checkpoint) if checkpoint else None,
            "exists": bool(checkpoint and checkpoint.is_file()),
            "bundle_files": {filename: bool(checkpoint and (checkpoint.parent / filename).is_file())
                             for filename in required},
        }
    backends = ("tifa", "game") if args.require == "all" else (() if args.require == "core" else (args.require,))
    if backends:
        needed = inference_packages if "tifa" in backends else inference_packages[:-1]
        missing = [name for name in needed if result["packages"][name] is None]
        if missing:
            result["problems"].append("Missing inference packages: " + ", ".join(missing))
            result["next_steps"].append("Install the inference extra into this same interpreter.")
        for name in backends:
            if not result["submodules"][name.upper()]:
                result["problems"].append(f"Missing {name.upper()} Python API repository.")
            model = result["models"][name]
            if not model["exists"] or not all(model["bundle_files"].values()):
                result["problems"].append(f"Missing or incomplete {name.upper()} model bundle.")
            repo = result["repositories"][name.upper()]
            if repo["matches_pin"] is False or repo["working_tree"]:
                result["problems"].append(f"{name.upper()} has a different revision or local changes; ask the user how to proceed.")
        if "tifa" in backends and config.get("ved_model"):
            model = result["models"]["ved"]
            if not model["exists"] or not all(model["bundle_files"].values()):
                result["problems"].append("Missing or incomplete VED model bundle.")
        result["next_steps"].append("Run a small preflight/inference pilot to check actual resource and language compatibility.")
    if args.check_device or args.remember:
        try:
            from makediffsinger.config import resolve_device
            selected_device = resolve_device(args.device, state_path=state_path)
            if args.check_device or backends:
                result["device"] = check_device(selected_device)
        except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as exc:
            result["problems"].append(f"Cannot use the selected device: {exc}")
    if args.remember and not result["problems"]:
        ignored = (git_output(root, "check-ignore", "--", str(state_path))
                   if root and state_path and state_path.is_relative_to(root) else None)
        if not ignored:
            result["problems"].append("Save Agent setup only in a Git-ignored state file inside this checkout.")
        else:
            setup = {
                "environment": args.environment, "python": sys.executable,
                "device": selected_device,
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result["saved_setup"] = setup
            result["remembered"] = True
    result["ready"] = not result["problems"]
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
