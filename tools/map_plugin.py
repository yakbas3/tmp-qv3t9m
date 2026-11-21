"""pytest plugin that records, per test, what the pinned sqlglot test suite touches.

Used only to build the map, never at scoring time. Records per test:
  - executed lines of every .py file under the repo root (sys.monitoring)
  - files opened and directories listed under the repo root (audit hook)
  - classes instantiated (MRO of every instance whose class has a repo __init__)
Everything sqlglot runs at import time goes to an "import" bucket, because the
whole package is imported before collection. Process pools are replaced by an
in-process pool that pickles arguments and results, so worker code is traced
under the test that submitted it.

Writes JSON to $MAP_OUT.
"""

from __future__ import annotations

import builtins
import concurrent.futures
import functools
import json
import os
import pickle
import pkgutil
import sys

from firstline import first_line

ROOT = os.path.realpath(os.environ.get("MAP_ROOT", os.getcwd())) + os.sep
TOOL = 4
mon = sys.monitoring

_lines: set = set()
_files: set = set()
_dirs: set = set()
_types: set = set()
_spawn = [False]


def _line(code, line):
    try:
        fn = code.co_filename
        if fn.startswith(ROOT):
            _lines.add((fn, line))
        return mon.DISABLE
    except Exception:
        return None


def _start_ev(code, offset):
    try:
        fn = code.co_filename
        if fn.startswith(ROOT):
            ln = first_line(code)
            if ln is not None:
                _lines.add((fn, ln))
        return mon.DISABLE
    except Exception:
        return None


def _audit(event, args):
    if event == "open":
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            try:
                p = os.fsdecode(path)
            except Exception:
                return
            if not os.path.isabs(p):
                p = os.path.join(os.getcwd(), p)
            p = os.path.normpath(p)
            if p.startswith(ROOT):
                _files.add(p)
    elif event in ("os.listdir", "os.scandir", "glob.glob", "os.walk"):
        path = args[0] if args else "."
        try:
            p = os.fsdecode(path if path is not None else ".")
        except Exception:
            return
        if not os.path.isabs(p):
            p = os.path.join(os.getcwd(), p)
        p = os.path.normpath(p)
        if p.startswith(ROOT) or p + os.sep == ROOT:
            _dirs.add(p)
    elif event in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.fork"):
        _spawn[0] = True


def _wrap_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        _types.add(type(self))
        return init(self, *args, **kwargs)

    __init__.__map_wrapped__ = True
    return __init__


_orig_build_class = builtins.__build_class__


def _build_class(func, name, *bases, **kwargs):
    cls = _orig_build_class(func, name, *bases, **kwargs)
    try:
        if getattr(func, "__code__", None) and func.__code__.co_filename.startswith(ROOT):
            init = cls.__dict__.get("__init__")
            if callable(init) and hasattr(init, "__code__") and not getattr(init, "__map_wrapped__", False):
                type.__setattr__(cls, "__init__", _wrap_init(init))
    except Exception:
        pass
    return cls


class _SyncPool:
    """ProcessPoolExecutor stand-in that runs work in-process, through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(result)))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *a) for a in zip(*iterables)]
        return (f.result() for f in futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recorder:
    def __init__(self):
        self.buckets: dict = {}

    def switch(self, key):
        """Close the current bucket under `key`."""
        b = self.buckets.setdefault(key, {"lines": set(), "files": set(), "dirs": set(), "types": set(), "spawn": False})
        b["lines"] |= _lines
        b["files"] |= _files
        b["dirs"] |= _dirs
        b["types"] |= _types
        b["spawn"] = b["spawn"] or _spawn[0]
        _lines.clear()
        _files.clear()
        _dirs.clear()
        _types.clear()
        _spawn[0] = False
        mon.restart_events()


REC = Recorder()


def _start():
    mon.use_tool_id(TOOL, "blastmap")
    mon.register_callback(TOOL, mon.events.LINE, _line)
    mon.register_callback(TOOL, mon.events.PY_START, _start_ev)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
    sys.addaudithook(_audit)
    builtins.__build_class__ = _build_class
    concurrent.futures.ProcessPoolExecutor = _SyncPool
    import concurrent.futures.process as cfp

    cfp.ProcessPoolExecutor = _SyncPool
    # Import the whole package now so import-time work lands in one bucket.
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith("__main__"):
            continue
        try:
            __import__(info.name)
        except Exception as e:  # noqa: BLE001
            print(f"map: cannot import {info.name}: {e}", file=sys.stderr)
    REC.switch("import")


def pytest_configure(config):
    _start()


import pytest  # noqa: E402


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    REC.switch("other")
    yield
    path = getattr(collector, "path", None)
    REC.switch(f"collect:{path}" if path is not None else "other")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    REC.switch("other")
    yield
    REC.switch(f"setup:{item.nodeid}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    REC.switch("other")
    cls = getattr(item, "cls", None)
    if cls is not None:
        _types.add(cls)
    yield
    REC.switch(f"call:{item.nodeid}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    REC.switch("other")
    yield
    REC.switch(f"teardown:{item.nodeid}")


FORK = os.environ.get("MAP_FORK") == "1"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """With MAP_FORK=1, run each test in a fork of the collected session.

    Code that runs once per process (functools caches, class-level memo
    tables, dispatch maps built on first use) is otherwise credited only to
    the first test that triggers it. A fork starts every test from the state
    right after collection, so each test is credited with all the run-once
    code it depends on. Session, module and class fixtures rerun per fork.
    """
    if not FORK:
        return None
    from _pytest.runner import runtestprotocol

    REC.switch("other")
    config = item.config
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        code = 0
        try:
            REC.buckets = {}
            config.__dict__["_map_outcomes"] = {}
            reports = runtestprotocol(item, nextitem=None, log=False)
            REC.switch("other")
            buckets = {}
            for key, b in REC.buckets.items():
                if key == "other":
                    continue
                buckets[key] = {
                    "lines": b["lines"], "files": b["files"], "dirs": b["dirs"],
                    "type_keys": _type_keys(b["types"]), "spawn": b["spawn"],
                }
            payload = {
                "buckets": buckets,
                "outcomes": config.__dict__.get("_map_outcomes", {}),
                "reports": [config.hook.pytest_report_to_serializable(config=config, report=r) for r in reports],
            }
            data = pickle.dumps(payload)
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
    outcomes = config.__dict__.setdefault("_map_outcomes", {})
    if not data:
        outcomes[item.nodeid] = "failed"
        print(f"map: fork for {item.nodeid} sent nothing", file=sys.stderr)
    else:
        payload = pickle.loads(data)
        for key, cb in payload["buckets"].items():
            b = REC.buckets.setdefault(key, {"lines": set(), "files": set(), "dirs": set(), "types": set(), "spawn": False})
            b["lines"] |= cb["lines"]
            b["files"] |= cb["files"]
            b["dirs"] |= cb["dirs"]
            b.setdefault("type_keys", set()).update(cb["type_keys"])
            b["spawn"] = b["spawn"] or cb["spawn"]
        for nid, o in payload["outcomes"].items():
            if o == "failed":
                outcomes[nid] = "failed"
            else:
                outcomes.setdefault(nid, o)
        # the child ran with log=False: replay its setup/call/teardown reports
        for r in payload["reports"]:
            rep = config.hook.pytest_report_from_serializable(config=config, data=r)
            item.ihook.pytest_runtest_logreport(report=rep)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def _type_keys(types):
    keys = set()
    for t in types:
        for k in getattr(t, "__mro__", ()):
            mod = sys.modules.get(getattr(k, "__module__", ""), None)
            f = getattr(mod, "__file__", None) or ""
            if os.path.realpath(f).startswith(ROOT):
                keys.add(f"{k.__module__}:{k.__qualname__}")
    return keys


def pytest_sessionfinish(session):
    REC.switch("other")
    mon.set_events(TOOL, 0)
    out = {}
    for key, b in REC.buckets.items():
        lines: dict = {}
        for fn, ln in b["lines"]:
            lines.setdefault(os.path.relpath(fn, ROOT), []).append(ln)
        out[key] = {
            "lines": {k: sorted(v) for k, v in lines.items()},
            "files": sorted(os.path.relpath(p, ROOT) for p in b["files"]),
            "dirs": sorted(os.path.relpath(p, ROOT) for p in b["dirs"]),
            "types": sorted(_type_keys(b["types"]) | b.get("type_keys", set())),
            "spawn": b["spawn"],
        }
    outcomes = getattr(session.config, "_map_outcomes", {})
    with open(os.environ["MAP_OUT"], "w") as f:
        json.dump({"buckets": out, "outcomes": outcomes}, f)


def pytest_runtest_logreport(report):
    cfg = _CONFIG[0]
    if cfg is None:
        return
    d = cfg.__dict__.setdefault("_map_outcomes", {})
    if report.failed:
        d[report.nodeid] = "failed"
    else:
        d.setdefault(report.nodeid, "ok")


_CONFIG = [None]


def pytest_sessionstart(session):
    _CONFIG[0] = session.config
