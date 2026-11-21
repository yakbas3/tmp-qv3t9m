#!/usr/bin/env python3
"""Merge cval_plugin output into entry/app/cval.json.gz.

  python3 entry/tools/cval_merge.py OUT_DIR/cval-*.json

Layout: {"owners": [test id or "M:file", ...], "ok": [owner index of every
test that passed], "import": [site, ...], "import_t": [site seen at
import with test-suite code on the stack, ...], "sites": {site: {"v": {repr:
[owner index, ...]}, "s": [owner index, ...]}}}. A site seen with more than
MAX_VALUES distinct values over the suite keeps only its "s" list, every
owner that ran it.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "cval.json.gz"
MAX_VALUES = 64


def main() -> None:
    owners: dict = {}

    def idx(o):
        k = owners.get(o)
        if k is None:
            k = owners[o] = len(owners)
        return k

    sites: dict = {}
    imp: set = set()
    imp_t: set = set()
    outcomes: dict = {}
    for p in sys.argv[1:]:
        d = json.loads(Path(p).read_text())
        imp |= set(d["import"])
        imp_t |= set(d.get("import_t", d["import"]))
        for t, o in d["outcomes"].items():
            if o == "failed" or t not in outcomes:
                outcomes[t] = o
        for site, recs in d["rec"].items():
            e = sites.setdefault(site, {"v": {}, "s": set()})
            for o, vals in recs.items():
                i = idx(o)
                if vals == "*":
                    e["s"].add(i)
                else:
                    for r in vals:
                        e["v"].setdefault(r, set()).add(i)
    out_sites = {}
    wide = 0
    for site, e in sites.items():
        if len(e["v"]) > MAX_VALUES:
            wide += 1
            s = set(e["s"])
            for v in e["v"].values():
                s |= v
            out_sites[site] = {"v": {}, "s": sorted(s)}
        else:
            out_sites[site] = {"v": {r: sorted(v) for r, v in e["v"].items()}, "s": sorted(e["s"])}
    ok = sorted(idx(t) for t, o in outcomes.items() if o == "ok")
    order = sorted(owners, key=owners.get)
    data = {"owners": order, "ok": ok, "import": sorted(imp), "import_t": sorted(imp_t), "sites": out_sites}
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    print(f"cval: {len(out_sites)} sites ({wide} wide), {len(imp)} seen at import, "
          f"{len(ok)}/{len(outcomes)} tests ok -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)", file=sys.stderr)


if __name__ == "__main__":
    main()
