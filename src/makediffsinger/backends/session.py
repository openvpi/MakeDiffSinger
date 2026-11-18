"""Persistent subprocesses sharing one configured Python environment."""

import os
import subprocess
import sys
import threading

from ..config import AlignmentOptions, NoteOptions, Resources, Runtime
from ..types import ResourceError
from .protocol import receive, send


class BackendSession:
    def __init__(self, backend: str, resources: Resources, runtime: Runtime | None = None, options=None):
        self.backend, self.resources = backend, resources
        self.runtime = runtime or Runtime()
        self.options = options or (AlignmentOptions() if backend == "tifa" else NoteOptions())
        self.process = None
        self.info = {}
        self._lock = threading.Lock()

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def start(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise ResourceError(f"The {self.backend} worker has exited.")
            return self
        repo = self.resources.repo(self.backend)
        if not (repo / "inference" / "api.py").is_file():
            raise ResourceError(f"Missing {self.backend} submodule at {repo}; run git submodule update --init --recursive.")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "makediffsinger.backends.worker", self.backend, str(repo)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            self.info = self.call("initialize", resources=self.resources, runtime=self.runtime, options=self.options)
        except BaseException:
            self.close()
            raise
        return self

    def call(self, operation: str, **payload):
        if self.process is None:
            self.start()
        with self._lock:
            try:
                send(self.process.stdin, {"operation": operation, **payload})
                response = receive(self.process.stdout)
            except (OSError, EOFError) as exc:
                raise ResourceError(f"{self.backend} worker connection failed: {exc}") from exc
        if not response["ok"]:
            raise ResourceError(f"{self.backend}: {response['type']}: {response['message']}")
        if "info" in response:
            self.info = response["info"]
        return response["result"]

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            if process.poll() is None:
                send(process.stdin, {"operation": "close"})
                process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=10)
        finally:
            process.stdin.close()
            process.stdout.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class TifaSession(BackendSession):
    def __init__(self, resources, runtime=None, options=None):
        super().__init__("tifa", resources, runtime, options)

    def preflight(self, samples):
        return self.call("preflight", samples=list(samples))

    def align(self, samples):
        return self.call("align", samples=list(samples))

    def pronunciations(self, texts, languages=None):
        return self.call("pronunciations", texts=list(texts), languages=languages)

    def fuse_lyrics(self, requests, languages=None, *, min_score=0.0, top_k=3):
        return self.call("fuse_lyrics", requests=list(requests), languages=languages,
                         min_score=min_score, top_k=top_k)


class GameSession(BackendSession):
    def __init__(self, resources, runtime=None, options=None):
        super().__init__("game", resources, runtime, options)

    def annotate(self, requests):
        """Requests contain an annotation and explicitly derived syllables."""
        return self.call("annotate", requests=list(requests))
