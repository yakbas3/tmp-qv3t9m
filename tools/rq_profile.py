"""Per query evidence for test_execute_tpcds: what the worker does with one
fixture query once the front end is done.

Used only to build app/rq.json.gz, never at scoring time. For every query
the test executes, tests.test_executor.mp_execute is run on the pickled
expression with the front end (see rq_common.patch_executor) left out, and
the lines, the classes and the CPU seconds are recorded. A worker of the
real test serves many queries, so caches may be warm or cold when a query
arrives. Each query is therefore run three ways and the union is kept:

  fresh  in a fork that has run nothing else
  seq    after every earlier query, in one process
  cold   in a fork where the front end ran in another process and only its
         pickled results were handed over, so no cache it fills is warm

Run inside the entry image, from /opt/sqlglot, with the tools on the path:

  docker run --rm --platform linux/amd64 --network none \\
    -v $PWD/entry/tools:/tools:ro -v $OUT:/out -e MAP_ROOT=/opt/sqlglot \\
    -e PYTHONPATH=/tools:/app:/opt/sqlglot -w /opt/sqlglot <entry image> \\
    python /tools/rq_profile.py /out/rq_profile.json [--cost-only]

With --cost-only the CPU seconds in an existing file are measured again and
nothing else changes. Do that on an idle machine: the selector spends a CPU
budget by these numbers.
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import sys

import rq_common as rq

rq.install()

import proxy  # noqa: E402  (app/proxy.py: the table loader the replay uses)


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _child(fn):
    """Run fn() in a fork; its return value comes back through pickle."""
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
    import sqlglot.executor as ex
    import tests.test_executor as te
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    rq.start()
    rq.patch_executor()
    pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    directory = te.DIR_TPCDS
    tables = {t: proxy.load_table(directory, t, cols) for t, cols in TPCDS_SCHEMA.items()}
    # the worker's globals: pickled copies, as the pool's initargs arrive
    te.initializer(*pickle.loads(pickle.dumps((TPCDS_SCHEMA, tables))))

    def parse_all():
        from sqlglot import parse_one

        return [pickle.dumps((parse_one(sql), meta)) for meta, sql, _ in pairs]

    blobs = _child(parse_all)  # parsed elsewhere: this process stays cold
    execd = [i for i, (meta, _, _) in enumerate(pairs) if meta.get("execute")]
    print(f"rq: {len(pairs)} queries, {len(execd)} executed", file=sys.stderr)

    def run(i):
        expr, meta = pickle.loads(blobs[i])
        types: set = set()
        rq.walk(expr, types)
        rq.begin("back")
        err = None
        digest = None
        try:
            table = te.mp_execute(expr, meta)
            digest = proxy.table_digest(table)
        except BaseException as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
        rq.begin("rest")
        lines, keys = rq.take("back")
        return {"lines": lines, "types": sorted(set(keys) | rq.type_keys(types)), "digest": digest, "err": err}

    def fresh(i):
        return lambda: run(i)

    def seq():
        return {i: run(i) for i in execd}

    def cold(i):
        def front_only():
            expr, meta = pickle.loads(blobs[i])
            got: dict = {}
            snap: list = []
            real_schema, real_opt, real_plan = ex.ensure_schema, ex.optimize, ex.Plan

            def keep(name, real):
                def call(*args, **kwargs):
                    got[name] = real(*args, **kwargs)
                    if name == "Plan":
                        # now, before the plan runs; one pickle, so what
                        # the plan shares with the tree stays shared
                        snap.append(pickle.dumps(got))
                    return got[name]

                return call

            ex.ensure_schema = keep("ensure_schema", real_schema)
            ex.optimize = keep("optimize", real_opt)
            ex.Plan = keep("Plan", real_plan)
            te.mp_execute(expr, meta)
            return snap[0]

        def body():
            data = _child(front_only)
            if data is None:
                return {"err": "front end results do not pickle"}
            got = pickle.loads(data)
            rq.COLD.update(got)
            rq.COLD["sql"] = True
            return run(i)

        return body

    def cost():
        rq.stop()
        out = {}
        for rep in range(3):
            for i in execd:
                expr, meta = pickle.loads(blobs[i])
                c0 = _cpu()
                te.mp_execute(expr, meta)
                d = _cpu() - c0
                out[i] = min(out.get(i, d), d)
        return out

    if "--cost-only" in sys.argv[2:]:
        out = json.load(open(out_path))
        costs = _child(cost) or {}
        for i in execd:
            out["q"][str(i)]["cost"] = round(costs[i], 4)
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"rq: costs in {out_path} measured again, {sum(costs.values()):.1f}s", file=sys.stderr)
        return
    res: dict = {i: {"lines": set(), "types": set(), "modes": {}} for i in execd}
    digests: dict = {}

    def merge(i, mode, r):
        if r is None:
            r = {"err": "no result"}
        res[i]["modes"][mode] = r.get("err") or "ok"
        if "lines" not in r:
            return
        res[i]["lines"].update(tuple(x) for x in r["lines"])
        res[i]["types"].update(r["types"])
        if r.get("digest") is not None:
            if digests.setdefault(i, r["digest"]) != r["digest"]:
                res[i]["modes"][mode] = "other table"

    for i in execd:
        merge(i, "fresh", _child(fresh(i)))
    print("rq: fresh done", file=sys.stderr)
    for i, r in (_child(seq) or {}).items():
        merge(i, "seq", r)
    print("rq: seq done", file=sys.stderr)
    for i in execd:
        merge(i, "cold", _child(cold(i)))
    print("rq: cold done", file=sys.stderr)
    costs = _child(cost) or {}
    out = {"nq": len(pairs), "exec": execd, "q": {}}
    for i in execd:
        lines: dict = {}
        for f, ln in res[i]["lines"]:
            lines.setdefault(f, []).append(ln)
        out["q"][str(i)] = {
            "lines": {f: sorted(v) for f, v in sorted(lines.items())},
            "types": sorted(res[i]["types"]),
            "cost": round(costs.get(i, 0.0), 4),
            "modes": res[i]["modes"],
        }
        bad = {m: v for m, v in res[i]["modes"].items() if v != "ok"}
        if bad or len(res[i]["modes"]) < 3:
            print(f"rq: query {i}: {res[i]['modes']}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"rq: wrote {out_path}, cost {sum(costs.values()):.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
