#!/usr/bin/env python3
"""Build entry/app/tbl.json.gz with tools/tbl_plugin.py.

  python3 entry/tools/tbl_build.py --image IMAGE [--shards 8] [--keep DIR] [--merge-only]

Runs the pinned suite in shards inside the entry image, each test in its own
fork, and merges what tbl_plugin recorded. Output, with test bitsets as
indexes into "sets" (hex, bit i = i-th test of kit/sample/tests.json):

  sites: {site: {"entries": [[lo, hi, [key names]]...],
                 "keys": {key name: set}, "all": set,
                 "imp_keys": [names read at import], "imp_all": bool}}
  unk: set of tests that did not run to a pass under the plugin
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1]
TOOLS = ENTRY / "tools"
OUT = ENTRY / "app" / "tbl.json.gz"
sys.path.insert(0, str(TOOLS))
from build_map import shard_files  # noqa: E402


def run_shard(image: str, index: int, files: list[str], workdir: Path) -> Path:
    out = workdir / f"tbl{index}.json"
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
        "-v", f"{TOOLS}:/tools:ro", "-v", f"{workdir}:/out",
        "-e", "SKIP_INTEGRATION=0", "-e", "TBL_ROOT=/opt/sqlglot",
        "-e", f"TBL_OUT=/out/tbl{index}.json", "-e", "PYTHONPATH=/tools:/opt/sqlglot",
        "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "-w", "/opt/sqlglot", image,
        "python", "-m", "pytest", "-q", "-p", "tbl_plugin", "-p", "no:cacheprovider",
        "-o", "addopts=", *files,
    ]
    log = workdir / f"tbl{index}.log"
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if not out.exists():
        sys.exit(f"tbl shard {index} wrote nothing, see {log}")
    print(f"tbl shard {index}: {log.read_text().strip().splitlines()[-1]}", file=sys.stderr)
    return out


def merge(paths: list[Path], tests: list[dict]) -> dict:
    ids = [t["id"] for t in tests]
    index = {tid: i for i, tid in enumerate(ids)}
    by_file: dict = defaultdict(int)
    for i, tid in enumerate(ids):
        by_file[tid.split("::")[0]] |= 1 << i
    entries: dict = {}
    keys: dict = defaultdict(lambda: defaultdict(int))
    alls: dict = defaultdict(int)
    imp_keys: dict = defaultdict(set)
    imp_all: set = set()
    outcomes: dict = {}
    for p in paths:
        d = json.loads(p.read_text())
        for s, e in d["entries"].items():
            cur = {(lo, hi): set(n) for lo, hi, n in entries.get(s, [])}
            for lo, hi, n in e:
                cur.setdefault((lo, hi), set()).update(n)
            entries[s] = [[lo, hi, sorted(n)] for (lo, hi), n in sorted(cur.items())]
        for t, o in d["outcomes"].items():
            if o == "failed" or t not in outcomes:
                outcomes[t] = o
        for key, b in d["buckets"].items():
            kind, _, rest = key.partition(":")
            if kind == "test":
                bits = 1 << index[rest] if rest in index else 0
            elif kind == "collect":
                bits = by_file.get(rest.replace("/opt/sqlglot/", "", 1), 0)
            else:
                bits = None  # import, other: state every test may inherit
            if not bits:
                if bits is None:
                    for s, n in b["look"]:
                        imp_keys[s].add(n)
                    imp_all.update(b["all"])
                continue
            for s, n in b["look"]:
                keys[s][n] |= bits
            for s in b["all"]:
                alls[s] |= bits
    unk = 0
    for i, tid in enumerate(ids):
        if outcomes.get(tid) != "ok":
            unk |= 1 << i

    sets: list = []
    sidx: dict = {}

    def sid(bits: int) -> int:
        if bits not in sidx:
            sidx[bits] = len(sets)
            sets.append(format(bits, "x"))
        return sidx[bits]

    sites = {}
    for s in sorted(entries):
        sites[s] = {
            "entries": entries[s],
            "keys": {n: sid(b) for n, b in sorted(keys.get(s, {}).items())},
            "all": sid(alls.get(s, 0)),
            "imp_keys": sorted(imp_keys.get(s, ())),
            "imp_all": s in imp_all,
        }
    print(f"tbl: {len(sites)} sites, {sum(len(v['keys']) for v in sites.values())} keyed reads, "
          f"{len(sets)} sets, {bin(unk).count('1')} tests not ok", file=sys.stderr)
    return {"sets": sets, "sites": sites, "unk": sid(unk)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=str(ENTRY.parent / "kit" / "sample" / "tests.json"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--keep")
    ap.add_argument("--merge-only", action="store_true")
    opts = ap.parse_args()
    tests = json.loads(Path(opts.tests).read_text())["tests"]
    workdir = Path(opts.keep) if opts.keep else Path(tempfile.mkdtemp(prefix="tbl"))
    workdir.mkdir(parents=True, exist_ok=True)
    if opts.merge_only:
        paths = sorted(workdir.glob("tbl*.json"))
    else:
        groups = shard_files(tests, opts.shards)
        with ThreadPoolExecutor(len(groups)) as pool:
            paths = list(pool.map(lambda a: run_shard(opts.image, a[0], a[1], workdir), enumerate(groups)))
    data = merge(paths, tests)
    raw = json.dumps(data, separators=(",", ":")).encode()
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, raw {len(raw) / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
