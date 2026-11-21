"""Merge eqprof_plugin outputs into app/eqprof.json.gz.

  python tools/eqprof_merge.py OUT.json.gz run1.json run2.json ...

A site is safe for a test only if every run of that test saw it safe. A test
with any failed run is left out, so it is never cleared by this rule.
"""

from __future__ import annotations

import gzip
import json
import sys


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    merged: dict = {}
    bad: set = set()
    for p in paths:
        d = json.load(open(p))
        if len(d["tests"]) != 1:
            raise SystemExit(f"{p}: expected one test, got {d['tests']}")
        tid = d["tests"][0]
        if not d["passed"]:
            bad.add(tid)
        s = merged.setdefault(tid, {})
        for k, v in d["sites"].items():
            s[k] = min(s.get(k, 1), int(v))
    for tid in bad:
        merged.pop(tid, None)
    blob = json.dumps(merged, sort_keys=True, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(gzip.compress(blob, 9, mtime=0))
    print({t: (len(s), sum(s.values())) for t, s in merged.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
