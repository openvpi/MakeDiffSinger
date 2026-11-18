"""Private framed transport between a parent and its own trusted Python worker."""

import pickle
import struct


def receive(stream):
    header = stream.read(8)
    if not header:
        raise EOFError("Worker connection closed.")
    if len(header) != 8:
        raise EOFError("Incomplete worker message header.")
    size = struct.unpack("!Q", header)[0]
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Incomplete worker message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return pickle.loads(b"".join(chunks))


def send(stream, message):
    payload = pickle.dumps(message, protocol=5)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()
