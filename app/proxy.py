"""Front-end fingerprints for the TPC executor tests.

test_execute_tpcds, test_execute_tpch and test_optimized_tpch spend their time
running the python executor over TPC data. Everything sqlglot does before
that (parse_one and transpile in the test, optimize and Plan inside execute)
is replayed here and reduced to one digest per test. The digest covers the
whole object graph each step hands on: classes, every attribute, enum
members and which objects are shared.

tools/rcov_plugin.py records the lines those tests run outside this front
end. When a patch only touches function bodies none of those lines reach,
and the digest on the patched tree equals the digest on the pinned tree,
the rest of the test sees identical inputs and runs identical code, so the
test passes as it does on the pinned tree.

  python proxy.py --baseline OUT.json   (build time, pinned tree)
"""

from __future__ import annotations

import enum
import hashlib
import json
import pickle
import sys
import types

TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("tpc-ds", True, False),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("tpc-h", True, False),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("tpc-h", False, True),
}


class Opaque(Exception):
    pass


class Memo:
    """Objects already written, numbered in order. A set member that needs a
    sub-digest is written against a child memo, so cycles still end."""

    def __init__(self, parent=None):
        self.d: dict = {}
        self.parent = parent
        self.n = parent.n if parent else 0

    def get(self, k):
        m = self
        while m is not None:
            v = m.d.get(k)
            if v is not None:
                return v
            m = m.parent
        return None

    def add(self, k, o):
        self.d[k] = (self.n, o)  # keep o alive so ids stay unique
        self.n += 1


_SCALARS = frozenset((type(None), bool, int, float, complex, str, bytes))
_SLOTS: dict = {}


def _slots(t) -> tuple:
    s = _SLOTS.get(t)
    if s is None:
        names: list = []
        for c in t.__mro__:
            for n in c.__dict__.get("__slots__", ()):
                if n not in ("__dict__", "__weakref__") and n not in names:
                    names.append(n)
        s = _SLOTS[t] = tuple(names)
    return s


_SORTED: dict = {}


def _attrs(o, t) -> list:
    names = _SORTED.get(t)
    if names is None:
        names = _SORTED[t] = tuple(sorted(_slots(t)))
    out = []
    for n in names:
        try:
            out.append((n, getattr(o, n)))
        except AttributeError:
            pass
    d = getattr(o, "__dict__", None)
    if d:
        out.extend(d.items())
        out.sort(key=lambda kv: kv[0])
    return out


def _shallow(v) -> str:
    t = type(v)
    if t in _SCALARS or isinstance(v, enum.Enum):
        return f"{t.__qualname__}:{v!r}"
    parts = [f"{t.__module__}.{t.__qualname__}"]
    for name, x in _attrs(v, t):
        if (type(x) in _SCALARS and name != "_hash") or isinstance(x, enum.Enum):
            parts.append(f"{name}={x!r}")
    return "|".join(parts)


def canon(obj, h, memo):
    """Feed a canonical form of the object graph into h. Shared objects are
    written once, then by reference. Anything that cannot be described
    exactly raises Opaque."""
    buf: list = []
    _canon(obj, buf, memo)
    h.update("\x1f".join(buf).encode("utf-8", "surrogatepass"))


_KIND: dict = {}


def _kind(t) -> int:
    k = _KIND.get(t)
    if k is None:
        if t in _SCALARS:
            k = 0
        elif issubclass(t, enum.Enum):
            k = 1
        elif issubclass(t, (type, types.FunctionType, types.BuiltinFunctionType)):
            k = 2
        else:
            k = 3
        _KIND[t] = k
    return k


def _canon(o, out, memo):
    t = type(o)
    kind = _KIND.get(t)
    if kind is None:
        kind = _kind(t)
    if kind == 0:
        out.append(t.__name__[:2] + repr(o))
        return
    if kind == 1:
        out.append(f"E{t.__module__}.{t.__qualname__}.{o.name}")
        return
    if kind == 2:
        out.append(f"F{getattr(o, '__module__', '')}.{getattr(o, '__qualname__', '')}")
        return
    k = id(o)
    seen = memo.get(k)
    if seen is not None:
        out.append(f"R{seen[0]}")
        return
    memo.add(k, o)
    if t is list or t is tuple:
        out.append(("L" if t is list else "T") + str(len(o)))
        for v in o:
            _canon(v, out, memo)
        return
    if t is dict:
        out.append(f"D{len(o)}")
        for key, v in o.items():
            _canon(key, out, memo)
            _canon(v, out, memo)
        return
    if t is set or t is frozenset:
        # members in a stable order: by when they were first written,
        # else by their plain attributes, else by a full sub-digest
        keyed = []
        for v in o:
            s = memo.get(id(v))
            keyed.append([(0, s[0]) if s is not None else (1, _shallow(v)), v])
        counts: dict = {}
        for kv in keyed:
            counts[kv[0]] = counts.get(kv[0], 0) + 1
        for kv in keyed:
            if counts[kv[0]] > 1:
                hh = hashlib.sha256()
                canon(kv[1], hh, Memo(memo))
                kv[0] = kv[0] + (hh.hexdigest(),)
        keyed.sort(key=lambda kv: kv[0])
        out.append(f"Z{len(o)}")
        for kv in keyed:
            _canon(kv[1], out, memo)
        return
    mod = t.__module__
    if mod == "builtins" or not mod.startswith(("sqlglot", "tests")):
        raise Opaque(f"{mod}.{t.__qualname__}")
    out.append(f"O{mod}.{t.__qualname__}")
    for name, v in _attrs(o, t):
        out.append(name)
        if name == "_hash":
            # a cached hash() of the node: its value follows PYTHONHASHSEED
            out.append("1" if v is not None else "0")
            continue
        if name == "args" and type(v) is dict and memo.get(id(v)) is None:
            # a node's named children; the order they were set in varies
            # between runs on the pinned tree (set iteration over objects)
            memo.add(id(v), v)
            out.append(f"A{len(v)}")
            for key in sorted(v, key=repr):
                _canon(key, out, memo)
                _canon(v[key], out, memo)
            continue
        _canon(v, out, memo)
    out.append(")")


def _guard(h, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        h.update(f"!{type(e).__module__}.{type(e).__qualname__}:{e}".encode())
        return None


def digests(test_id: str):
    """Yield one digest per fixture query, in order."""
    from sqlglot import parse_one, transpile
    from sqlglot.executor import ensure_schema
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

    bench, run_exec, run_opt = TESTS[test_id]
    fixture = f"optimizer/{bench}/{bench}.sql"
    schema_src = TPCDS_SCHEMA if bench == "tpc-ds" else TPCH_SCHEMA
    pairs = list(load_sql_fixture_pairs(fixture))
    memo = Memo()
    for meta, sql, optimized in pairs:
        h = hashlib.sha256()
        h.update(repr((meta, sql, optimized)).encode("utf-8", "surrogatepass"))
        if run_opt:
            canon(_guard(h, transpile, sql, write="duckdb"), h, memo)
            canon(_guard(h, transpile, optimized, write="duckdb"), h, memo)
            yield h.hexdigest()
            continue
        expr = _guard(h, parse_one, sql)
        canon(expr, h, memo)
        if meta.get("execute") and expr is not None:
            canon(_guard(h, transpile, sql, write="duckdb"), h, memo)
            # the worker gets a pickled copy and runs execute(): optimize, then Plan
            sent = pickle.loads(pickle.dumps(expr))
            schema = ensure_schema(schema_src, dialect=None)
            opt = _guard(h, optimize, sent, schema, leave_tables_isolated=True, dialect=None)
            plan = None if opt is None else _guard(h, Plan, opt)
            # after Plan, so whatever Plan did to the tree is in the digest too
            canon((opt, plan), h, memo)
        yield h.hexdigest()


def baseline() -> dict:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    out = {}
    for tid in TESTS:
        try:
            out[tid] = list(digests(tid))
        except BaseException as e:  # noqa: BLE001
            print(f"proxy baseline {tid}: {type(e).__name__}: {e}", file=sys.stderr)
            out[tid] = []  # never matches, so the test is always kept
    return out


def check(test_id: str, base: list) -> bool:
    """True when every query digest on this tree matches the pinned one.
    Stops at the first difference."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    if not base or test_id not in TESTS:
        return False
    try:
        n = 0
        for i, d in enumerate(digests(test_id)):
            if i >= len(base) or d != base[i]:
                return False
            n += 1
        return n == len(base)
    except BaseException:  # noqa: BLE001
        return False


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--baseline":
        res = baseline()
        with open(sys.argv[2], "w") as f:
            json.dump(res, f)
        print({k: len(v) for k, v in res.items()}, file=sys.stderr)
