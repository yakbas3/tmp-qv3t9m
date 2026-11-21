#!/usr/bin/env python3
"""Build entry/app/rq.json.gz: what the TPC executor tests reach outside
the front end that app/proxy.py replays, per fixture query.

  python3 entry/tools/rq_build.py [--image TAG] [--no-build] [--keep DIR] [--merge-only]

Builds the entry image (linux/amd64), then runs inside it
  - tools/rq_plugin.py on the three TPC tests, a few times (real workers)
  - tools/rq_profile.py (one query at a time)
and merges them. Nothing here reads any patch or answer file. Needs Docker.

The file maps every line and class to the set of TPC-DS queries whose run in
the worker reaches it. One more bit, numbered after the last query, stands
for the rest of the three tests: what they do outside the front end and
outside any query's run.

  {"nq": 99, "cost": [cpu seconds per query, 0 when not executed],
   "sets": [hex bitsets], "lines": {file: [[line, set], ...]}, "types": {key: set}}
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1]
TOOLS = ENTRY / "tools"
OUT = ENTRY / "app" / "rq.json.gz"
TESTS = [
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
]


def docker(image: str, workdir: Path, env: dict, cmd: list, log: str) -> None:
    args = ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
            "-v", f"{TOOLS}:/tools:ro", "-v", f"{ENTRY / 'app'}:/appsrc:ro", "-v", f"{workdir}:/out",
            "-e", "SKIP_INTEGRATION=0", "-e", "MAP_ROOT=/opt/sqlglot",
            "-e", "PYTHONPATH=/tools:/appsrc:/opt/sqlglot"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    args += ["-w", "/opt/sqlglot", image, *cmd]
    with open(workdir / log, "w") as fh:
        rc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT).returncode
    tail = (workdir / log).read_text().strip().splitlines()[-1:]
    print(f"{log}: rc {rc} {tail}", file=sys.stderr)
    if rc != 0:
        sys.exit(f"{log} failed, see {workdir / log}")


def merge(real_paths: list, profile_path: Path, front_path: Path) -> dict:
    prof = json.loads(profile_path.read_text())
    fr = json.loads(front_path.read_text())
    nq = prof["nq"]
    rest = 1 << nq
    line_bits: dict = defaultdict(int)
    type_bits: dict = defaultdict(int)
    cost = [0.0] * nq
    for i_s, q in prof["q"].items():
        i = int(i_s)
        if set(q["modes"].values()) != {"ok"} or len(q["modes"]) < 3:
            print(f"rq: query {i} modes {q['modes']}", file=sys.stderr)
        if q["modes"].get("fresh") != "ok" or q["modes"].get("seq") != "ok":
            sys.exit(f"query {i} did not run cleanly: {q['modes']}")
        cost[i] = q["cost"]
        for f, lines in q["lines"].items():
            for ln in lines:
                line_bits[(f, ln)] |= 1 << i
        for k in q["types"]:
            type_bits[k] |= 1 << i
    for path in real_paths:
        real = json.loads(Path(path).read_text())
        for name, reg in real.items():
            bit = rest if name == "rest" else 1 << int(name.split(":")[1])
            for f, lines in reg["lines"].items():
                for ln in lines:
                    line_bits[(f, ln)] |= bit
            for k in reg["types"]:
                type_bits[k] |= bit
    sets: list = []
    index: dict = {}

    def sid(bits: int) -> int:
        if bits not in index:
            index[bits] = len(sets)
            sets.append(format(bits, "x"))
        return index[bits]

    lines_out: dict = defaultdict(list)
    for (f, ln), bits in sorted(line_bits.items()):
        if f.startswith("sqlglot/"):
            lines_out[f].append([ln, sid(bits)])
    types_out = {k: sid(b) for k, b in sorted(type_bits.items())}
    if fr["nq"] != nq:
        sys.exit(f"rq_front has {fr['nq']} queries, the profile {nq}")
    front_bits: dict = defaultdict(int)
    fcost = [0.0] * nq
    for i_s, q in fr["q"].items():
        i = int(i_s)
        if set(q["modes"].values()) != {"ok"} or len(q["modes"]) < 3:
            sys.exit(f"front of query {i} did not run cleanly: {q['modes']}")
        fcost[i] = q["cost"]
        for f, lines in q["lines"].items():
            for ln in lines:
                front_bits[(f, ln)] |= 1 << i
    front_out: dict = defaultdict(list)
    for (f, ln), bits in sorted(front_bits.items()):
        if f.startswith("sqlglot/"):
            front_out[f].append([ln, sid(bits)])
    n_rest = sum(1 for b in line_bits.values() if b & rest)
    print(f"rq: {len(line_bits)} lines ({n_rest} in the rest), {len(type_bits)} classes, "
          f"{len(prof['exec'])} queries, {sum(cost):.1f}s to run them all", file=sys.stderr)
    print(f"rq: front {len(front_bits)} lines, {sum(fcost):.1f}s for every query's front end",
          file=sys.stderr)
    return {"nq": nq, "cost": cost, "sets": sets, "lines": lines_out, "types": types_out,
            "front": front_out, "fcost": fcost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="blast-radius-map")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep", help="directory to keep raw output in")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--runs", type=int, default=3, help="runs of the real tests")
    ap.add_argument("--keep-profile", action="store_true", help="reuse rq_profile.json in --keep")
    ap.add_argument("--front-only", action="store_true", help="only rerun tools/rq_front.py")
    ap.add_argument("--cost-only", action="store_true",
                    help="only measure the CPU seconds in --keep's rq_profile.json again (idle machine)")
    opts = ap.parse_args()
    workdir = Path(opts.keep) if opts.keep else Path(tempfile.mkdtemp(prefix="blastrq"))
    workdir.mkdir(parents=True, exist_ok=True)
    if opts.cost_only:
        docker(opts.image, workdir, {}, ["python", "/tools/rq_profile.py", "/out/rq_profile.json", "--cost-only"],
               "rq_cost.log")
        docker(opts.image, workdir, {}, ["python", "/tools/rq_front.py", "/out/rq_front.json", "--cost-only"],
               "rq_front_cost.log")
    elif opts.front_only:
        docker(opts.image, workdir, {}, ["python", "/tools/rq_front.py", "/out/rq_front.json"], "rq_front.log")
    elif not opts.merge_only:
        if not opts.no_build:
            subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", opts.image, str(ENTRY)], check=True)
        for k in range(opts.runs):
            docker(opts.image, workdir, {"RQ_OUT": f"/out/rq_real{k}.json", "PYTHONHASHSEED": str(k)},
                   ["python", "-m", "pytest", "-q", "-p", "rq_plugin", "-p", "no:cacheprovider",
                    "-o", "addopts=", *TESTS], f"rq_real{k}.log")
        if not (opts.keep_profile and (workdir / "rq_profile.json").exists()):
            docker(opts.image, workdir, {}, ["python", "/tools/rq_profile.py", "/out/rq_profile.json"],
                   "rq_profile.log")
        docker(opts.image, workdir, {}, ["python", "/tools/rq_front.py", "/out/rq_front.json"], "rq_front.log")
    data = merge(sorted(workdir.glob("rq_real*.json")), workdir / "rq_profile.json",
                 workdir / "rq_front.json")
    raw = json.dumps(data, separators=(",", ":")).encode()
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, raw {len(raw) / 1e3:.0f} kB)", file=sys.stderr)


if __name__ == "__main__":
    main()
