"""Per query evidence for the front end app/proxy.py replays for
test_execute_tpcds: which lines Replay.front(i) reaches, and its CPU cost.

Used only to build app/rq.json.gz, never at scoring time. Replay.front(i)
does what the test does with fixture query i before the plan runs: parse_one
and transpile in the test process, the pickle round trip to the worker, and
ensure_schema, optimize, the debug SQL and Plan in the worker. When none of
the lines it reaches changed, it computes the pinned digests, so the proxy
need not recompute them. Caches may be warm or cold, so each query runs
three ways and the union is kept:

  fresh  in a fork that has run nothing else (a cold query reaches every
         line whose result it uses)
  seq    after every earlier query, in one process
  rev    after every later query, in one process

Run inside the entry image, from /opt/sqlglot:

  docker run --rm --platform linux/amd64 --network none \\
    -v $PWD/entry/tools:/tools:ro -v $PWD/entry/app:/appsrc:ro -v $OUT:/out \\
    -e MAP_ROOT=/opt/sqlglot -e PYTHONPATH=/tools:/appsrc:/opt/sqlglot -w /opt/sqlglot \\
    <entry image> python /tools/rq_front.py /out/rq_front.json [--cost-only]
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import sys

import rq_common as rq

rq.install()

import proxy  # noqa: E402


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _child(fn):
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        code = 0
        try:
            data = pickle.dumps(fn())
            with os.fdopen(wfd, "wb") as fh:
                fh.write(data)
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            os._exit(code)
    os.close(wfd)
    with os.fdopen(rfd, "rb") as fh:
        data = fh.read()
    os.waitpid(pid, 0)
    return pickle.loads(data) if data else None


def main():
    out_path = sys.argv[1]
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    rq.start()
    rp = proxy.Replay()
    n = len(rp.pairs)
    rq.take("rest")

    def run(i):
        rq.begin("f")
        try:
            d = rp.front(i)
            err = None
        except BaseException as e:  # noqa: BLE001
            d, err = None, f"{type(e).__name__}: {e}"[:300]
        rq.begin("rest")
        lines, _keys = rq.take("f")
        return {"lines": lines, "d": None if d is None else [d[0], d[1]], "err": err}

    def order(seq):
        return lambda: {i: run(i) for i in seq}

    def cost():
        rq.stop()
        out = {}
        for _rep in range(2):
            for i in range(n):
                c0 = _cpu()
                rp.front(i)
                d = _cpu() - c0
                out[i] = min(out.get(i, d), d)
        return out

    if "--cost-only" in sys.argv[2:]:
        out = json.load(open(out_path))
        costs = _child(cost) or {}
        for i in range(n):
            out["q"][str(i)]["cost"] = round(costs[i], 4)
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"rq_front: costs measured again, {sum(costs.values()):.1f}s", file=sys.stderr)
        return

    res = {i: {"lines": set(), "modes": {}, "d": None} for i in range(n)}

    def merge(i, mode, r):
        if r is None:
            res[i]["modes"][mode] = "no result"
            return
        res[i]["modes"][mode] = r["err"] or "ok"
        res[i]["lines"].update(tuple(x) for x in r["lines"])
        if r["d"] is not None:
            if res[i]["d"] is None:
                res[i]["d"] = r["d"]
            elif res[i]["d"] != r["d"]:
                res[i]["modes"][mode] = "other digest"

    for i in range(n):
        merge(i, "fresh", _child(lambda i=i: run(i)))
    print("rq_front: fresh done", file=sys.stderr)
    for mode, seq in (("seq", range(n)), ("rev", range(n - 1, -1, -1))):
        for i, r in (_child(order(list(seq))) or {}).items():
            merge(i, mode, r)
        print(f"rq_front: {mode} done", file=sys.stderr)
    costs = _child(cost) or {}
    out = {"nq": n, "q": {}}
    bad = 0
    for i in range(n):
        lines: dict = {}
        for f, ln in res[i]["lines"]:
            lines.setdefault(f, []).append(ln)
        modes = res[i]["modes"]
        if set(modes.values()) != {"ok"} or len(modes) < 3:
            bad += 1
            print(f"rq_front: query {i}: {modes}", file=sys.stderr)
        out["q"][str(i)] = {"lines": {f: sorted(v) for f, v in sorted(lines.items())},
                            "cost": round(costs.get(i, 0.0), 4), "modes": modes}
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"rq_front: wrote {out_path}, {bad} queries not clean, cost {sum(costs.values()):.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
