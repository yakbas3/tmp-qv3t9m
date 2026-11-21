"""pytest plugin: lines the TPC executor tests run outside the proxied front end.

Used only to build app/rcov.json.gz, never at scoring time. The front end is
every call the proxy in app/proxy.py replays and fingerprints: parse_one and
transpile in the test module, optimize and Plan inside sqlglot.executor.
Lines run anywhere else (setUpClass, pickling, the spawned workers, the
executor itself, result checks) are recorded. A line run both inside and
outside the front end is recorded. Workers are real spawned processes, so
caches start cold there just as in the real test.

Writes JSON {relpath: [lines]} to $RCOV_OUT. Rebuild (about 2 minutes):

  docker run --rm --platform linux/amd64 --network none \\
    -v $PWD/entry/tools:/tools:ro -v $OUT:/out -e MAP_ROOT=/opt/sqlglot \\
    -e RCOV_OUT=/out/rcov.json -e PYTHONPATH=/tools:/opt/sqlglot -w /opt/sqlglot \\
    <entry image> python -m pytest -q -p rcov_plugin -p no:cacheprovider -o addopts= \\
    tests/test_executor.py::TestExecutor::test_execute_tpcds \\
    tests/test_executor.py::TestExecutor::test_execute_tpch \\
    tests/test_executor.py::TestExecutor::test_optimized_tpch

then keep the sqlglot/ entries, gzipped, as app/rcov.json.gz.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading

from firstline import first_line

ROOT = os.path.realpath(os.environ.get("MAP_ROOT", os.getcwd())) + os.sep
OUT = os.environ.get("RCOV_OUT", "/out/rcov.json")
TOOL = 3
mon = sys.monitoring

_lines: set = set()
_local = threading.local()


def _paused() -> bool:
    return getattr(_local, "depth", 0) > 0


def _line(code, line):
    try:
        if _paused():
            return None
        fn = code.co_filename
        if fn.startswith(ROOT):
            _lines.add((fn[len(ROOT):], line))
        return mon.DISABLE
    except Exception:
        return None


def _start_ev(code, offset):
    try:
        if _paused():
            return None
        fn = code.co_filename
        if fn.startswith(ROOT):
            ln = first_line(code)
            if ln is not None:
                _lines.add((fn[len(ROOT):], ln))
        return mon.DISABLE
    except Exception:
        return None


def _install():
    try:
        mon.use_tool_id(TOOL, "rcov")
    except ValueError:
        return
    mon.register_callback(TOOL, mon.events.LINE, _line)
    mon.register_callback(TOOL, mon.events.PY_START, _start_ev)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)


def front(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _local.depth = getattr(_local, "depth", 0) + 1
        try:
            return fn(*args, **kwargs)
        finally:
            _local.depth -= 1

    wrapper.__rcov_front__ = True
    return wrapper


def _patch_executor():
    import sqlglot.executor as ex

    if not getattr(ex.optimize, "__rcov_front__", False):
        ex.optimize = front(ex.optimize)
        ex.Plan = front(ex.Plan)


def _dump(path):
    data: dict = {}
    for f, ln in _lines:
        data.setdefault(f, set()).add(ln)
    if os.path.exists(path):
        try:
            for f, v in json.load(open(path)).items():
                data.setdefault(f, set()).update(v)
        except Exception:
            pass
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({f: sorted(v) for f, v in data.items()}, fh)
    os.replace(tmp, path)


def child_init(*args):
    _install()
    _patch_executor()
    import tests.test_executor as te

    return te.initializer(*args)


def child_task(*args):
    import tests.test_executor as te

    try:
        return te.mp_execute(*args)
    finally:
        _dump(f"{OUT}.child{os.getpid()}")


def pytest_configure(config):
    _install()


def pytest_collection_finish(session):
    import tests.test_executor as te

    te.parse_one = front(te.parse_one)
    te.transpile = front(te.transpile)
    te.initializer = child_init
    te.mp_execute = child_task
    _patch_executor()


def pytest_unconfigure(config):
    import glob

    _dump(OUT)
    for p in glob.glob(f"{OUT}.child*"):
        if p.endswith(".tmp"):
            continue
        for f, v in json.load(open(p)).items():
            for ln in v:
                _lines.add((f, ln))
        os.unlink(p)
    _dump(OUT)
