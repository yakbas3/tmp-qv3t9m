"""pytest plugin: operand types at every `is`, `is not`, `==` and `!=` in sqlglot.

Used only to build app/eqprof.json.gz, never at scoring time. Every
single-operator comparison of those four kinds in sqlglot/ is compiled into a
call that records whether swapping `is` and `==` (or `is not` and `!=`) could
change its result for the operands it saw, then does the original comparison.

Swapping cannot change the result when both operand types compare by identity
(no __eq__ or __ne__ of their own, or NoneType), or when both are the same
Enum class whose __eq__ is object's, str's or int's (members are singletons,
an equal value is the same member).

Run one test per process, so setUpClass and everything else the test runs is
in its record. Pools run in-process through pickle, so worker code is seen.
Writes JSON {"test": id, "passed": bool, "sites": {"rel:line:col": 0|1}}
to $EQPROF_OUT. Rebuild:

  for t in <big tests>; for seed in 0 1; do
    docker run --rm --platform linux/amd64 --network none \\
      -v $PWD/entry/tools:/tools:ro -v $OUT:/out -e MAP_ROOT=/opt/sqlglot \\
      -e EQPROF_OUT=/out/<n>.json -e PYTHONHASHSEED=$seed \\
      -e PYTHONPATH=/tools:/opt/sqlglot -w /opt/sqlglot <entry image> \\
      python -m pytest -q -p eqprof_plugin -p no:cacheprovider -o addopts= $t
  done
  python tools/eqprof_merge.py $OUT/*.json   -> app/eqprof.json.gz
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import enum
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import sys

ROOT = os.path.realpath(os.environ.get("MAP_ROOT", os.getcwd())) + os.sep
PKG = ROOT + "sqlglot" + os.sep
OUT = os.environ.get("EQPROF_OUT", "/out/eqprof.json")

_sites: dict = {}
_NONE = type(None)
_OBJ_EQ = object.__eq__
_OBJ_NE = object.__ne__
_ENUM_EQ = (object.__eq__, str.__eq__, int.__eq__)
_ENUM_NE = (object.__ne__, str.__ne__, int.__ne__)
_ok_cache: dict = {}


def _ident(t) -> bool:
    return t is _NONE or (t.__eq__ is _OBJ_EQ and t.__ne__ is _OBJ_NE)


def _swap_safe(ta, tb) -> bool:
    k = (ta, tb)
    v = _ok_cache.get(k)
    if v is None:
        if _ident(ta) and _ident(tb):
            v = True
        elif (ta is tb and issubclass(ta, enum.Enum) and not issubclass(ta, enum.Flag)
              and ta.__eq__ in _ENUM_EQ and ta.__ne__ in _ENUM_NE):
            v = True
        else:
            v = False
        _ok_cache[k] = v
    return v


def _rec(key, op, a, b):
    if not _swap_safe(type(a), type(b)):
        _sites[key] = 0
    elif key not in _sites:
        _sites[key] = 1
    if op == 0:
        return a is b
    if op == 1:
        return a is not b
    if op == 2:
        return a == b
    return a != b


_OPS = {ast.Is: 0, ast.IsNot: 1, ast.Eq: 2, ast.NotEq: 3}


class _Rewrite(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in _OPS:
            return node
        key = f"{self.rel}:{node.lineno}:{node.col_offset}"
        call = ast.Call(
            func=ast.Name(id="__eqprof_rec__", ctx=ast.Load()),
            args=[ast.Constant(key), ast.Constant(_OPS[type(node.ops[0])]),
                  node.left, node.comparators[0]],
            keywords=[],
        )
        return ast.copy_location(call, node)


class _Loader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):
        rel = os.path.realpath(path)[len(ROOT):]
        tree = ast.parse(data, path)
        tree = ast.fix_missing_locations(_Rewrite(rel).visit(tree))
        return compile(tree, path, "exec", dont_inherit=True)

    def get_code(self, fullname):
        # never use or write cached bytecode for rewritten modules
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        if not os.path.realpath(spec.origin).startswith(PKG):
            return spec
        spec.loader = _Loader(fullname, spec.origin)
        return spec


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


_state = {"ids": [], "failed": False}


def pytest_configure(config):
    if any(m == "sqlglot" or m.startswith("sqlglot.") for m in sys.modules):
        raise RuntimeError("sqlglot imported before eqprof_plugin")
    builtins.__eqprof_rec__ = _rec
    sys.meta_path.insert(0, _Finder())
    concurrent.futures.ProcessPoolExecutor = _SyncPool
    import concurrent.futures.process as cfp

    cfp.ProcessPoolExecutor = _SyncPool


def pytest_runtest_logreport(report):
    if report.nodeid not in _state["ids"]:
        _state["ids"].append(report.nodeid)
    if report.failed or (report.when == "call" and not report.passed):
        _state["failed"] = True


def pytest_unconfigure(config):
    data = {"tests": _state["ids"], "passed": bool(_state["ids"]) and not _state["failed"],
            "sites": _sites}
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, OUT)
