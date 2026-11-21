#!/usr/bin/env python3
"""Merge obs_plugin output into entry/app/obs.json.gz.

  python3 entry/tools/obs_merge.py OUT_DIR/obs-*.json

Build (one container, whole pinned suite, native arch is fine since only
types and dict reads are recorded; repeat with a second PYTHONHASHSEED and
merge both):

  docker run --rm --network none -v $PWD/entry/tools:/tools:ro -v $OUT:/out \\
    -e OBS_OUT=/out -e OBS_ROOT=/opt/sqlglot -e PYTHONHASHSEED=0 \\
    -e PYTHONPATH=/tools:/opt/sqlglot -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    -w /opt/sqlglot <entry image> python -m pytest -q -p obs_plugin \\
    -p xdist -n 4 -p no:cacheprovider -o addopts= tests
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "obs.json.gz"


def main() -> None:
    files: set = set()
    cmp: dict = {}
    reads: dict = {}
    outcomes: dict = {}
    for p in sys.argv[1:]:
        d = json.loads(Path(p).read_text())
        files |= set(d["files"])
        for k, v in d["cmp"].items():
            cmp[k] = min(cmp.get(k, 1), v)
        for k, v in d["dict"].items():
            reads.setdefault(k, set()).update(v)
        for t, o in d["outcomes"].items():
            if o == "failed" or t not in outcomes:
                outcomes[t] = o
    data = {
        "files": sorted(files),
        "cmp": cmp,
        "dict": {k: sorted(v) for k, v in reads.items()},
        "outcomes": outcomes,
    }
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    ok = sum(o == "ok" for o in outcomes.values())
    print(f"obs: {len(files)} files, {len(cmp)} comparisons ({sum(v == 0 for v in cmp.values())} unsafe), "
          f"{len(reads)} dicts ({sum('*' in v for v in reads.values())} bulk), "
          f"{ok}/{len(outcomes)} tests ok -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)", file=sys.stderr)


if __name__ == "__main__":
    main()
