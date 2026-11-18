"""Associate a delivery with its original input directory."""

import argparse
from pathlib import Path

from makediffsinger.catalog import bind_dataset
from makediffsinger.script_utils import run


def execute(args):
    source = bind_dataset(args.output, args.input, relocate=args.relocate)
    return {"dataset": source.name, "delivery": str(args.output)}, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Delivery root beside the numbered stages.")
    parser.add_argument("--relocate", action="store_true", help="Reconnect the same dataset after moving it.")
    raise SystemExit(run(parser, execute))
