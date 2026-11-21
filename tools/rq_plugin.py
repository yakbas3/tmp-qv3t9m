"""pytest plugin: what the TPC executor tests run outside the replayed front
end, split in two regions.

Used only to build app/rq.json.gz, never at scoring time. The front end is
every call app/proxy.py replays and fingerprints: parse_one and transpile in
the test module, the pickle round trip of each call sent to a worker, and
ensure_schema, optimize, the debug SQL and Plan inside sqlglot.executor.
Outside it, lines and classes (see rq_common.py) are filed under

  back:<i>  inside tests.test_executor.mp_execute: a worker running the
            TPC-DS query number i (the plugin tags each query's meta dict)
  rest      everything else: collection, setUpClass, the pool, result
            checks, and mp_execute on a TPC-H query (which returns at once)

Workers are real spawned processes, as in the test. Writes JSON
{region: {"lines": {relpath: [lines]}, "types": [keys]}} to $RQ_OUT. Run it
a few times: which worker serves which query varies.

  docker run --rm --platform linux/amd64 --network none \\
    -v $PWD/entry/tools:/tools:ro -v $OUT:/out -e MAP_ROOT=/opt/sqlglot \\
    -e RQ_OUT=/out/rq_real.json -e PYTHONPATH=/tools:/opt/sqlglot -w /opt/sqlglot \\
    <entry image> python -m pytest -q -p rq_plugin -p no:cacheprovider -o addopts= \\
    tests/test_executor.py::TestExecutor::test_execute_tpcds \\
    tests/test_executor.py::TestExecutor::test_execute_tpch \\
    tests/test_executor.py::TestExecutor::test_optimized_tpch
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import rq_common as rq

OUT = os.environ.get("RQ_OUT", "/out/rq_real.json")
TAG = "_rq_query"

# at import: in a spawned worker this module loads before sqlglot does
rq.install()


def _dump(path):
    data: dict = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path))
        except Exception:
            data = {}
    for name in sorted(rq._regions):
        lines, keys = rq.take(name)
        if not lines and not keys:
            continue
        reg = data.setdefault(name, {"lines": {}, "types": []})
        for f, ln in lines:
            reg["lines"].setdefault(f, [])
            if ln not in reg["lines"][f]:
                reg["lines"][f].append(ln)
        reg["types"] = sorted(set(reg["types"]) | set(keys))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def child_init(*args):
    import multiprocessing.reduction as red

    rq.start()
    rq.patch_executor()
    # a call arriving from the test process: its unpickling is front end,
    # and what it yields is handed to mp_execute, which files it
    red.ForkingPickler.loads = staticmethod(rq.front(pickle.loads, handed=True))
    import tests.test_executor as te

    return te.initializer(*args)


def child_task(*args):
    import tests.test_executor as te

    types: set = set()
    meta = args[1] if isinstance(args[1], dict) else {}
    if meta.get("execute"):
        # otherwise mp_execute returns at once and the tree is dropped unread
        rq.walk(args, types)
    i = meta.get(TAG)
    name = "rest" if i is None else f"back:{i}"
    rq.begin(name)
    try:
        return te.mp_execute(*args)
    finally:
        rq._regions[name][1].update(types)
        rq.begin("rest")
        _dump(f"{OUT}.child{os.getpid()}")


def pytest_collection_finish(session):
    import multiprocessing.reduction as red

    import tests.test_executor as te

    rq.start()
    run_pool = te.TestExecutor._mp_execute

    def tagged(self, schema, tables, sqls, tpch):
        if not tpch:
            for i, (meta, _sql, _opt) in enumerate(sqls):
                meta[TAG] = i
        return run_pool(self, schema, tables, sqls, tpch)

    te.TestExecutor._mp_execute = tagged
    te.parse_one = rq.front(te.parse_one, handed=True)
    te.transpile = rq.front(te.transpile)
    te.initializer = child_init
    te.mp_execute = child_task
    rq.patch_executor()
    plain = red.ForkingPickler.dumps.__func__
    marked = rq.front(plain)

    def dumps(cls, obj, protocol=None):
        # a call on its way to a worker; results and the rest stay outside
        if type(obj).__name__ == "_CallItem":
            return marked(cls, obj, protocol)
        return plain(cls, obj, protocol)

    red.ForkingPickler.dumps = classmethod(dumps)


def pytest_unconfigure(config):
    import glob

    _dump(OUT)
    merged = json.load(open(OUT))
    for p in glob.glob(f"{OUT}.child*"):
        if p.endswith(".tmp"):
            continue
        for name, reg in json.load(open(p)).items():
            m = merged.setdefault(name, {"lines": {}, "types": []})
            for f, v in reg["lines"].items():
                m["lines"][f] = sorted(set(m["lines"].get(f, [])) | set(v))
            m["types"] = sorted(set(m["types"]) | set(reg["types"]))
        os.unlink(p)
    with open(OUT, "w") as fh:
        json.dump(merged, fh)
