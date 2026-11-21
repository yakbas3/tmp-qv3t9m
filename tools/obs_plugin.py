"""pytest plugin that records what values reach certain expressions in the
pinned sqlglot test suite. Used only to build app/obs.json.gz, never at
scoring time. sqlglot modules are compiled from a rewritten AST:

  - every single-operator comparison with is, is not, == or != is routed
    through a recorder that notes, per site, whether every operand type
    pair it saw makes `is` and `==` agree (eqprof_plugin's rule, here over
    the whole suite instead of a few big tests).
  - every dict literal bound to a name at module or class level becomes a
    dict subclass that notes which keys have their value read. Copies,
    iteration over values or items, comparisons and pickling count as a
    read of every key ("*").

Writes JSON to $OBS_OUT/obs-<pid>.json at session end.
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

ROOT = os.path.realpath(os.environ.get("OBS_ROOT", os.getcwd())) + os.sep
PKG = ROOT + "sqlglot" + os.sep

CMP: dict = {}     # site -> 1 if every operand pair seen was swap safe, else 0
DICT: dict = {}    # site -> set of keys whose value was read ("*" = all)
FILES: list = []   # rewritten files
OUTCOMES: dict = {}

_FAMILY = {ast.Is: 0, ast.IsNot: 1, ast.Eq: 2, ast.NotEq: 3}
_NONE = type(None)
_OBJ_EQ = object.__eq__
_OBJ_NE = object.__ne__
_ENUM_EQ = (object.__eq__, str.__eq__, int.__eq__)
_ENUM_NE = (object.__ne__, str.__ne__, int.__ne__)
_ok_cache: dict = {}


def _ident(t) -> bool:
    return t is _NONE or (t.__eq__ is _OBJ_EQ and t.__ne__ is _OBJ_NE)


def _swap_safe(ta, tb) -> bool:
    """Same criterion as eqprof_plugin: for these operand types `is` and
    `==` (and `is not` and `!=`) give the same answer for every value."""
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


def __obs_cmp__(site, op, a, b):
    if not _swap_safe(type(a), type(b)):
        CMP[site] = 0
    elif site not in CMP:
        CMP[site] = 1
    if op == 0:
        return a is b
    if op == 1:
        return a is not b
    if op == 2:
        return a == b
    return a != b


class ObsDict(dict):
    __slots__ = ("_obs",)

    def _r(self, k):
        try:
            DICT[self._obs].add(repr(k))
        except Exception:  # noqa: BLE001
            DICT[self._obs].add("*")

    def _all(self):
        DICT[self._obs].add("*")

    def __getitem__(self, k):
        self._r(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._r(k)
        return dict.get(self, k, default)

    def setdefault(self, k, default=None):
        self._r(k)
        return dict.setdefault(self, k, default)

    def pop(self, k, *a):
        self._r(k)
        return dict.pop(self, k, *a)

    def popitem(self):
        self._all()
        return dict.popitem(self)

    def items(self):
        self._all()
        return dict.items(self)

    def values(self):
        self._all()
        return dict.values(self)

    def copy(self):
        self._all()
        return dict.copy(self)

    def __iter__(self):
        # defined so that dict(d), {**d} and f(**d) go through __getitem__
        return dict.__iter__(self)

    def __reversed__(self):
        return dict.__reversed__(self)

    def __eq__(self, other):
        self._all()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._all()
        return dict.__ne__(self, other)

    __hash__ = None

    def __or__(self, other):
        self._all()
        return dict.__or__(self, other)

    def __ror__(self, other):
        self._all()
        return dict.__ror__(self, other)

    def __repr__(self):
        self._all()
        return dict.__repr__(self)

    __str__ = __repr__

    def __reduce_ex__(self, proto):
        self._all()
        return dict.__reduce_ex__(self, proto)

    def __reduce__(self):
        self._all()
        return dict.__reduce__(self)

    def __format__(self, spec):
        self._all()
        return dict.__format__(self, spec)


def __obs_dict__(site, d):
    o = ObsDict(d)
    o._obs = site
    DICT.setdefault(site, set())
    return o


class _Rewrite(ast.NodeTransformer):
    def __init__(self, rel: str):
        self.rel = rel
        self.depth = 0  # function nesting

    def _key(self, n) -> str:
        return f"{self.rel}:{n.lineno}:{n.col_offset}"

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        self.visit(node.args)
        if node.returns:
            self.visit(node.returns)
        self.depth += 1
        node.body = [self.visit(s) for s in node.body]
        self.depth -= 1
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.visit(node.args)
        self.depth += 1
        node.body = self.visit(node.body)
        self.depth -= 1
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in _FAMILY:
            return node
        call = ast.Call(
            func=ast.Name(id="__obs_cmp__", ctx=ast.Load()),
            args=[ast.Constant(self._key(node)), ast.Constant(_FAMILY[type(node.ops[0])]),
                  node.left, node.comparators[0]],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def _dict_value(self, value):
        if self.depth == 0 and isinstance(value, ast.Dict) and value.keys \
                and all(isinstance(k, ast.Constant) for k in value.keys):
            key = self._key(value)
            call = ast.Call(func=ast.Name(id="__obs_dict__", ctx=ast.Load()),
                            args=[ast.Constant(key), value], keywords=[])
            return ast.copy_location(call, value)
        return value

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            node.value = self._dict_value(node.value)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is not None and isinstance(node.target, ast.Name):
            node.value = self._dict_value(node.value)
        return node


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        src = self.get_data(path)
        rel = os.path.relpath(os.path.realpath(path), ROOT)
        tree = ast.parse(src, filename=path)
        tree = _Rewrite(rel).visit(tree)
        ast.fix_missing_locations(tree)
        FILES.append(rel)
        return compile(tree, path, "exec", dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "sqlglot" and not name.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return spec
        if not os.path.realpath(spec.origin).startswith(PKG):
            return spec
        spec.loader = _Loader(spec.loader.name, spec.loader.path)
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


builtins.__obs_cmp__ = __obs_cmp__
builtins.__obs_dict__ = __obs_dict__
if "sqlglot" in sys.modules:
    raise RuntimeError("obs_plugin must load before sqlglot")
sys.meta_path.insert(0, _Finder())
concurrent.futures.ProcessPoolExecutor = _SyncPool
import concurrent.futures.process as _cfp  # noqa: E402

_cfp.ProcessPoolExecutor = _SyncPool


def pytest_runtest_logreport(report):
    if report.failed:
        OUTCOMES[report.nodeid] = "failed"
    else:
        OUTCOMES.setdefault(report.nodeid, "ok")


def pytest_sessionfinish(session):
    out = os.path.join(os.environ["OBS_OUT"], f"obs-{os.getpid()}.json")
    with open(out, "w") as f:
        json.dump({
            "files": sorted(set(FILES)),
            "cmp": CMP,
            "dict": {k: sorted(v) for k, v in DICT.items()},
            "outcomes": OUTCOMES,
        }, f)
