"""Freeze code, protocol and inputs before fitting or scoring."""

import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(run, name, include_data=False, include_checkpoints=False):
    run = Path(run)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "experiments").rglob("*.py"))
    paths += sorted((root / "basket_simulator").glob("*.py"))
    paths += [run / "protocol.json"]
    if include_data:
        paths += sorted(p for p in (run / "data").iterdir() if p.is_file())
    if include_checkpoints:
        paths += sorted(p for p in (run / "selected").rglob("*") if p.is_file())
    manifest = {str(p.resolve()): digest(p) for p in paths}
    target = run / name
    if target.exists():
        if manifest != json.loads(target.read_text()):
            raise RuntimeError(
                f"Source/input change since {target}; use a new run or record a reviewed amendment"
            )
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
