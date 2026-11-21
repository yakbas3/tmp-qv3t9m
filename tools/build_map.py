#!/usr/bin/env python3
"""Build entry/app/map.json.gz from the pinned sqlglot commit.

  python3 entry/tools/build_map.py [--tests kit/sample/tests.json] [--shards 4]

Builds the entry image (linux/amd64), runs sqlglot's own test suite inside it
in a few shards with tools/map_plugin.py loaded, and merges what each test
touched into one compressed map. Nothing here reads any patch or answer file.
Needs Docker. Two passes: one process per shard, then one fork per test (see
MAP_FORK in map_plugin.py). The map is their union.
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
OUT = ENTRY / "app" / "map.json.gz"
IMAGE = "blast-radius-map"


def shard_files(tests: list[dict], shards: int) -> list[list[str]]:
    cost: dict[str, float] = defaultdict(float)
    for t in tests:
        cost[t["file"]] += t["seconds"] + 0.05
    groups: list[list[str]] = [[] for _ in range(shards)]
    loads = [0.0] * shards
    for f in sorted(cost, key=lambda f: -cost[f]):
        i = loads.index(min(loads))
        groups[i].append(f)
        loads[i] += cost[f]
    return groups


def run_shard(index: int, files: list[str], workdir: Path, fork: bool = False) -> Path:
    name = f"shard{index}{'f' if fork else ''}"
    out = workdir / f"{name}.json"
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
        "-v", f"{TOOLS}:/tools:ro", "-v", f"{workdir}:/out",
        "-e", "SKIP_INTEGRATION=0", "-e", "MAP_ROOT=/opt/sqlglot",
        "-e", f"MAP_FORK={1 if fork else 0}",
        "-e", f"MAP_OUT=/out/{name}.json", "-e", "PYTHONPATH=/tools:/opt/sqlglot",
        "-w", "/opt/sqlglot", IMAGE,
        "python", "-m", "pytest", "-q", "-p", "map_plugin", "-p", "no:cacheprovider",
        "--timeout=600", "-o", "addopts=", *files,
    ]
    log = workdir / f"{name}.log"
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if not out.exists():
        sys.exit(f"shard {name} wrote no map, see {log}")
    print(f"shard {name} done: {log.read_text().strip().splitlines()[-1]}", file=sys.stderr)
    return out


def merge(paths: list[Path], tests: list[dict]) -> dict:
    ids = [t["id"] for t in tests]
    index = {tid: i for i, tid in enumerate(ids)}
    by_file: dict[str, list[int]] = defaultdict(list)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(ids):
        by_file[tid.split("::")[0]].append(i)
        by_class[tid.rsplit("::", 1)[0]].append(i)

    line_tests: dict[tuple, int] = defaultdict(int)   # (file, line) -> bitset
    opened: dict[str, int] = defaultdict(int)
    listed: dict[str, int] = defaultdict(int)
    types: dict[str, int] = defaultdict(int)
    spawn = 0
    import_lines: dict[str, set] = defaultdict(set)
    outcomes: dict[str, str] = {}

    def bits_for(key: str) -> int:
        kind, _, rest = key.partition(":")
        if kind == "call":
            i = index.get(rest)
            return 0 if i is None else 1 << i
        if kind in ("setup", "teardown"):
            members = by_class.get(rest.rsplit("::", 1)[0], [])
            if rest in index and index[rest] not in members:
                members = members + [index[rest]]
            return sum(1 << i for i in members)
        if kind == "collect":
            rel = rest.replace("/opt/sqlglot/", "", 1)
            return sum(1 << i for i in by_file.get(rel, []))
        return 0

    for path in paths:
        data = json.loads(path.read_text())
        outcomes.update(data["outcomes"])
        for key, b in data["buckets"].items():
            bits = bits_for(key)
            if not bits:
                for f, lines in b["lines"].items():
                    import_lines[f].update(lines)
                continue
            for f, lines in b["lines"].items():
                for ln in lines:
                    line_tests[(f, ln)] |= bits
            for p in b["files"]:
                if "__pycache__" in p:
                    continue
                opened[p] |= bits
            for d in b["dirs"]:
                listed[d] |= bits
            for t in b["types"]:
                types[t] |= bits
            if b["spawn"]:
                spawn |= bits

    sets: list[str] = []
    set_index: dict[int, int] = {}

    def sid(bits: int) -> int:
        if bits not in set_index:
            set_index[bits] = len(sets)
            sets.append(format(bits, "x"))
        return set_index[bits]

    lines_out: dict[str, list] = defaultdict(list)
    for (f, ln), bits in sorted(line_tests.items()):
        lines_out[f].append([ln, sid(bits)])
    missing = [tid for tid in ids if tid not in outcomes]
    failed = [tid for tid, o in outcomes.items() if o == "failed"]
    print(f"map: {len(line_tests)} lines, {len(sets)} unique sets, "
          f"{len(missing)} tests not run, {len(failed)} failed in base", file=sys.stderr)
    for tid in missing[:20]:
        print(f"  not run: {tid}", file=sys.stderr)
    for tid in failed[:20]:
        print(f"  failed: {tid}", file=sys.stderr)
    return {
        "tests": ids,
        "sets": sets,
        "lines": lines_out,
        "import_lines": {f: sorted(v) for f, v in import_lines.items()},
        "opened": {p: sid(b) for p, b in opened.items()},
        "listed": {d: sid(b) for d, b in listed.items()},
        "types": {t: sid(b) for t, b in types.items()},
        "spawn": sid(spawn),
        "not_run": missing,
        "base_failed": failed,
    }


def main() -> None:
    global IMAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=str(ENTRY.parent / "kit" / "sample" / "tests.json"))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--keep", help="directory to keep raw shard output in")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--fork-shards", type=int, default=8,
                    help="shards for the second pass, one fork per test (0 skips it)")
    opts = ap.parse_args()
    tests = json.loads(Path(opts.tests).read_text())["tests"]

    workdir = Path(opts.keep) if opts.keep else Path(tempfile.mkdtemp(prefix="blastmap"))
    workdir.mkdir(parents=True, exist_ok=True)
    IMAGE = opts.image
    if not opts.merge_only:
        if not opts.no_build:
            subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", IMAGE, str(ENTRY)], check=True)
        # Pass 1: one process per shard, as the tests normally run.
        # Pass 2: every test in its own fork of the collected session, so code
        # that runs once per process is credited to every test that needs it.
        # The map is the union of both.
        jobs = [(i, g, False) for i, g in enumerate(shard_files(tests, opts.shards))]
        if opts.fork_shards:
            jobs += [(i, g, True) for i, g in enumerate(shard_files(tests, opts.fork_shards))]
        with ThreadPoolExecutor(len(jobs)) as pool:
            paths = list(pool.map(lambda a: run_shard(a[0], a[1], workdir, a[2]), jobs))
    else:
        paths = sorted(workdir.glob("shard*.json"))
    data = merge(paths, tests)
    raw = json.dumps(data, separators=(",", ":")).encode()
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, raw {len(raw) / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
