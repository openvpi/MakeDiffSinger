"""Run exactly one upstream namespace for the lifetime of a process."""

import contextlib
from dataclasses import asdict, replace
import os
import sys
import traceback

from .protocol import receive, send
from ..config import is_device_unsupported, resolve_device


def initialize(backend, request):
    if backend == "tifa":
        from .tifa import TifaAdapter
        return TifaAdapter(**request)
    if backend == "game":
        from .game import GameAdapter
        return GameAdapter(**request)
    raise ValueError(f"Unknown backend: {backend}")


def invoke(adapter, operation, request):
    if adapter is None or operation not in {"preflight", "align", "annotate", "pronunciations", "fuse_lyrics"}:
        raise ValueError(f"Unsupported operation: {operation}")
    return getattr(adapter, operation)(**request)


def main():
    backend, repo = sys.argv[1:3]
    wire_in = sys.stdin.buffer
    wire_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    sys.path.insert(0, repo)
    adapter = initialization = None
    while True:
        try:
            request = receive(wire_in)
        except EOFError:
            return
        operation = request.pop("operation")
        if operation == "close":
            return
        try:
            with contextlib.redirect_stdout(sys.stderr):
                previous_device = adapter.runtime.device if adapter is not None else None
                if operation == "initialize":
                    initialization = request
                    if backend == "game":
                        runtime = request["runtime"]
                        initialization = {**request, "runtime": replace(runtime, device=resolve_device(runtime.device))}
                elif adapter is not None and previous_device is None and operation in {"align", "fuse_lyrics"}:
                    adapter.runtime = replace(adapter.runtime, device=resolve_device())
                    adapter.info["runtime"] = asdict(adapter.runtime)

                try:
                    if operation == "initialize":
                        adapter = initialize(backend, initialization)
                        result = adapter.info
                    else:
                        result = invoke(adapter, operation, request)
                except (RuntimeError, ValueError) as exc:
                    runtime = adapter.runtime if adapter is not None else (initialization or {}).get("runtime")
                    if runtime is None or not is_device_unsupported(runtime.device, exc):
                        raise
                    fallback = {"requested_device": runtime.device, "reason": str(exc)}
                    print(f"{backend.upper()} cannot run on {runtime.device}: {exc}. Retrying on CPU.")
                    adapter = initialize(backend, {**initialization, "runtime": replace(runtime, device="cpu")})
                    adapter.info["device_fallback"] = fallback
                    result = adapter.info if operation == "initialize" else invoke(adapter, operation, request)

            response = {"ok": True, "result": result}
            if operation != "initialize" and previous_device != adapter.runtime.device:
                response["info"] = adapter.info
            send(wire_out, response)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send(wire_out, {"ok": False, "type": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    main()
