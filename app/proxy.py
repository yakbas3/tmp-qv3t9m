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

test_execute_tpcds is replayed one fixture query at a time (Replay). Each
query has a digest of what the test process computes (parse_one, transpile),
one of what the worker computes before it runs the plan (the pickled
expression, ensure_schema, optimize, Plan), and one of the result table.
tools/rq_plugin.py and tools/rq_profile.py record which lines and classes
each query's run of the plan reaches. A query whose front digests match and
whose run reaches no change returns the pinned table. Any other query is run
here on the real tables and its table compared with the pinned one.

  python proxy.py --baseline OUT.json   (build time, pinned tree)
"""

from __future__ import annotations

import ast
import csv
import enum
import hashlib
import json
import os
import pickle
import sys
import types

TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("tpc-ds", True, False),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("tpc-h", True, False),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("tpc-h", False, True),
}


TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"


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


def _send(h, expr, meta):
    """The pool's round trip of one call's arguments (the test submits
    every fixture query); what raises is written into h."""
    blob = _guard(h, pickle.dumps, (expr, meta))
    return None if blob is None else _guard(h, pickle.loads, blob)


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
        # every query goes to a worker, executed or not: a pickle that raises
        # either way fails the test
        sent = _send(h, expr, meta)
        if meta.get("execute") and expr is not None:
            canon(_guard(h, transpile, sql, write="duckdb"), h, memo)
            # the worker runs execute(): optimize, the debug SQL, then Plan
            canon(sent, h, memo)
            schema = ensure_schema(schema_src, dialect=None)
            opt = None if sent is None else \
                _guard(h, optimize, sent[0], schema, leave_tables_isolated=True, dialect=None)
            if opt is not None:
                _guard(h, opt.sql, pretty=True)
            plan = None if opt is None else _guard(h, Plan, opt)
            # after Plan, so whatever Plan did to the tree is in the digest too
            canon((opt, plan), h, memo)
        yield h.hexdigest()


def _open_file(file_name):
    with open(file_name, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    if gzipped:
        import gzip

        return gzip.open(file_name, "rt", newline="")
    return open(file_name, encoding="utf-8", newline="")


def load_table(directory: str, table: str, columns):
    """One fixture table, read as TestExecutor.setUpClass reads it."""
    from sqlglot.executor.table import Table

    reader = csv.reader(_open_file(f"{directory}{table}.csv.gz"), delimiter="|")
    rows = []
    ctypes: list = []
    next(reader)
    for row in reader:
        if not ctypes:
            for i, v in enumerate(row):
                try:
                    ctypes.append(type(ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    return Table(columns=columns, rows=rows)


def table_digest(table) -> str:
    """What the test reads of a result table: the columns and every cell."""
    h = hashlib.sha256()

    def enc(v):
        t = type(v)
        return f"{t.__module__}.{t.__qualname__}:{v!r}"

    h.update((type(table.columns).__qualname__ + "\x1e" + "\x1f".join(enc(c) for c in table.columns)
              + "\n" + type(table.rows).__qualname__ + "\n").encode("utf-8", "surrogatepass"))
    for r in table.rows:
        h.update((type(r).__qualname__ + "\x1e" + "\x1f".join(enc(v) for v in r) + "\n")
                 .encode("utf-8", "surrogatepass"))
    return h.hexdigest()


class Replay:
    """test_execute_tpcds, one fixture query at a time."""

    def __init__(self):
        from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, load_sql_fixture_pairs

        self.pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        self.dir = FIXTURES_DIR + "/optimizer/tpc-ds/"
        self.schema_main = TPCDS_SCHEMA
        # the workers get pickled copies of the schema and of the tables
        self.schema = pickle.loads(pickle.dumps(TPCDS_SCHEMA))
        self.tables: dict = {}
        self.marks: dict = {}

    def pair(self, i: int) -> str:
        """Digest of what the test reads of fixture pair i: its meta and
        SQL, never the optimized SQL."""
        meta, sql, _optimized = self.pairs[i]
        return hashlib.sha256(repr((meta, sql)).encode("utf-8", "surrogatepass")).hexdigest()

    def front(self, i: int):
        """(main digest, worker digest, pickled call) of query i; the last
        two are None for a query the test does not execute. Each query is
        digested on its own (fresh memo), so any subset can be checked."""
        from sqlglot import parse_one, transpile
        from sqlglot.executor import ensure_schema
        from sqlglot.optimizer import optimize
        from sqlglot.planner import Plan

        memo = Memo()
        meta, sql, _optimized = self.pairs[i]
        h = hashlib.sha256()
        h.update(self.pair(i).encode())
        expr = _guard(h, parse_one, sql)
        canon(expr, h, memo)
        if not meta.get("execute") or expr is None:
            # still submitted to a worker, which drops it unread: only a
            # pickle round trip that raises can matter
            _send(h, expr, meta)
            return h.hexdigest(), None, None
        canon(_guard(h, transpile, sql, write="duckdb"), h, memo)
        main = h.hexdigest()
        h = hashlib.sha256()
        blob = _guard(h, pickle.dumps, (expr, meta))
        sent = None if blob is None else _guard(h, pickle.loads, blob)
        canon(sent, h, memo)
        if sent is None:
            return main, h.hexdigest(), None
        # execute() in the worker: ensure_schema, optimize, the debug SQL, Plan
        schema = _guard(h, ensure_schema, self.schema, dialect=None)
        canon(schema, h, memo)
        opt = _guard(h, optimize, sent[0], schema, leave_tables_isolated=True, dialect=None)
        if opt is not None:
            # logged at debug level and dropped: only a raise, or what it
            # does to the tree, can matter
            _guard(h, opt.sql, pretty=True)
        plan = None if opt is None else _guard(h, Plan, opt)
        # after Plan, so whatever Plan did to the tree is in the digest too
        canon((opt, plan), h, memo)
        return main, h.hexdigest(), blob

    def table(self, name: str):
        t = self.tables.get(name)
        if t is None:
            t = pickle.loads(pickle.dumps(load_table(self.dir, name, self.schema_main[name])))
            self.tables[name] = t
            self.marks[name] = self._mark(t)
        return t

    @staticmethod
    def _mark(t):
        # rows are tuples of plain values: the same row objects in the same
        # order are the same table
        return (tuple(t.columns), len(t.rows), hash(tuple(map(id, t.rows))))

    def intact(self) -> bool:
        """The worker keeps one copy of the schema and of each table for
        every query it serves: a run that changed one changed the input of
        the queries after it."""
        return self.schema == self.schema_main and \
            all(self._mark(t) == self.marks[name] for name, t in self.tables.items())

    def back(self, blob) -> str:
        """Run the pickled call as the worker does; digest of the table sent back."""
        from sqlglot import find_tables
        from sqlglot.executor import execute

        expression, _meta = pickle.loads(blob)
        tables = {}
        for t in find_tables(expression):
            name = t.name
            tables[name] = self.table(name)
        result = execute(expression, schema=self.schema, tables=tables)
        return table_digest(pickle.loads(pickle.dumps(result)))


def setup_ok() -> bool:
    """TestExecutor.setUpClass reads the TPC-H fixture pairs too, and the
    class shares one result cache keyed by SQL text. The TPC-DS test sees
    them only through a raise there or through a TPC-H query equal to one
    of its own."""
    from tests.helpers import load_sql_fixture_pairs

    try:
        h = {sql for _m, sql, _o in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql")}
        ds = {sql for _m, sql, _o in load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql")}
    except Exception:  # noqa: BLE001
        return False
    return not (h & ds)


def baseline_tpcds() -> list:
    rp = Replay()
    out = []
    for i in range(len(rp.pairs)):
        main, worker, blob = rp.front(i)
        out.append([main, worker, None if blob is None else rp.back(blob), rp.pair(i)])
        if not rp.intact():
            raise ValueError(f"query {i} changes the shared tables")
    return out


class Overtime(BaseException):
    """A step ran far past its measured CPU cost: a patch that loops."""


class _limit:
    """Raise Overtime once this process has spent `sec` more CPU seconds
    (a BaseException, so no except Exception in sqlglot swallows it)."""

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        import signal

        def fire(_sig, _frame):
            raise Overtime(f"over {self.sec:.1f}s cpu")

        self.old = signal.signal(signal.SIGPROF, fire)
        signal.setitimer(signal.ITIMER_PROF, max(self.sec, 0.01))
        return self

    def __exit__(self, *exc):
        import signal

        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, self.old)
        return False


# a query whose front digest differs is run at once, while it is cheap and
# no more than PROBES have been: a commit that fails the test usually shows
# it there, before the other front ends are paid for
PROBES = 2
PROBE_MAX = 0.3


def check_tpcds(base: list, need=(), costs=None, budget: float = 0.0, info: dict | None = None,
                front=None, cap: float | None = None, fcosts=None) -> bool:
    """True when the test passes as on the pinned tree: every front digest
    the test process computes matches, and each query whose worker digest
    differs, or that `need` names, returns the pinned table when run here.

    Runs cost `costs[i]` each. The check stops as soon as the runs it has
    committed to exceed `budget`, or the CPU it has spent plus the runs
    still owed exceed `cap` seconds.

    `front` names the queries whose front end may reach a change (None:
    all). Any other query whose fixture pair is unchanged runs the pinned
    front end lines on the pinned input, so its digests are the pinned
    ones and it is not recomputed."""
    import time

    info = {} if info is None else info
    t0 = time.process_time()

    def stop(why):
        info["stop"] = why
        info["cpu"] = round(time.process_time() - t0, 2)
        return False

    if not setup_ok():
        return stop("setup")
    rp = Replay()
    if len(base) != len(rp.pairs) or any(len(b) != 4 for b in base):
        return stop("pairs")
    need = set(need)
    todo = set(range(len(rp.pairs))) if front is None else set(front) | need
    todo |= {i for i in range(len(rp.pairs)) if rp.pair(i) != base[i][3]}
    committed = 0.0
    owed = 0.0
    info["ran"] = []
    info["front"] = len(todo)

    def cost(i):
        return float(costs[i]) if costs else float("inf")

    def run(i, blob, b_table):
        info["ran"].append(i)
        with _limit(3.0 * cost(i) + 1.0):
            got = rp.back(blob)
        if got != b_table:
            return f"table {i}"
        if not rp.intact():
            return f"shared tables {i}"
        return None

    pending = []
    probes = 0
    # executed queries first, cheapest run first: a failing commit then
    # shows a differing table after one or two cheap fronts and runs
    order = sorted(todo, key=lambda i: (base[i][1] is None, cost(i), i))
    total = sum(cost(i) for i in order if base[i][1] is not None)
    seen = 0.0
    n_seen = 0
    # cheapest first: the probes are then cheap
    for i in order:
        with _limit(5.0 * (float(fcosts[i]) if fcosts else 1.0) + 0.5):
            main, worker, blob = rp.front(i)
        b_main, b_worker, b_table, _b_pair = base[i]
        if b_worker is not None:
            seen += cost(i)
            n_seen += 1
        if main != b_main:
            return stop(f"main {i}")
        if b_worker is None:
            if worker is not None:
                return stop(f"worker {i}")
            continue
        if worker == b_worker and i not in need:
            continue
        if blob is None or b_table is None:
            return stop(f"worker {i}")
        committed += cost(i)
        if committed > budget:
            return stop(f"budget {i}")
        if probes < PROBES and cost(i) <= PROBE_MAX:
            probes += 1
            why = run(i, blob, b_table)
            if why:
                return stop(why)
        else:
            pending.append((i, blob, b_table))
            owed += cost(i)
        if cap is not None and time.process_time() - t0 + owed > cap:
            return stop(f"cap {i}")
        # most worker digests differ: at this rate the runs will not fit
        if n_seen >= 6 and seen > 0 and committed / seen * total > 2 * budget:
            return stop(f"projected {i}")
    for i, blob, b_table in pending:
        why = run(i, blob, b_table)
        if why:
            return stop(why)
    info["cpu"] = round(time.process_time() - t0, 2)
    return True


def risk_tpcds(base: list, order, budget: float, costs, fcosts, info: dict | None = None) -> bool:
    """Not a proof: the risk dial (BLAST_TPCDS_RISK in selector.py). Takes
    the TPC-DS queries in `order`, recomputes each front end and runs each
    executed query, until `budget` CPU seconds are spent. False at the first
    sign the test fails (a changed fixture pair or test-process digest, a
    query the worker would now execute or no longer can send, a different
    table, a raise, a loop); True when the budget runs out, or the order
    ends, without one. A worker digest that differs is no sign by itself:
    a changed plan may still return the pinned table."""
    import time

    info = {} if info is None else info
    t0 = time.process_time()

    def stop(why):
        info["risk_stop"] = why
        info["risk_cpu"] = round(time.process_time() - t0, 2)
        return False

    if not setup_ok():
        return stop("setup")
    rp = Replay()
    if len(base) != len(rp.pairs) or any(len(b) != 4 for b in base):
        return stop("pairs")
    for i in range(len(rp.pairs)):
        if rp.pair(i) != base[i][3]:
            return stop(f"pair {i}")
    info["risk_ran"] = []
    for i in order:
        if time.process_time() - t0 >= budget:
            break
        b_main, b_worker, b_table, _b_pair = base[i]
        with _limit(5.0 * (float(fcosts[i]) if fcosts else 1.0) + 0.5):
            main, worker, blob = rp.front(i)
        if main != b_main:
            return stop(f"main {i}")
        if b_worker is None:
            if worker is not None:
                return stop(f"worker {i}")
            continue
        if blob is None or b_table is None:
            return stop(f"worker {i}")
        info["risk_ran"].append(i)
        with _limit(3.0 * (float(costs[i]) if costs else 1.0) + 1.0):
            got = rp.back(blob)
        if got != b_table:
            return stop(f"table {i}")
        if not rp.intact():
            return stop(f"shared tables {i}")
    info["risk_cpu"] = round(time.process_time() - t0, 2)
    return True


# check_tpcds stops that say nothing about the outcome: the check ran out
# of budget, not into a difference
NO_EVIDENCE = ("budget", "cap", "projected", "risk only")


def baseline() -> dict:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    out = {}
    for tid in TESTS:
        try:
            out[tid] = baseline_tpcds() if tid == TPCDS else list(digests(tid))
        except BaseException as e:  # noqa: BLE001
            print(f"proxy baseline {tid}: {type(e).__name__}: {e}", file=sys.stderr)
            out[tid] = []  # never matches, so the test is always kept
    return out


def check(test_id: str, base: list, need=(), costs=None, budget: float = 0.0,
          info: dict | None = None, front=None, cap: float | None = None, fcosts=None) -> bool:
    """True when every query digest on this tree matches the pinned one.
    Stops at the first difference."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    if not base or test_id not in TESTS:
        return False
    if test_id == TPCDS:
        import time

        t0 = time.process_time()
        try:
            return check_tpcds(base, need, costs, budget, info, front, cap, fcosts)
        except BaseException as e:  # noqa: BLE001
            if info is not None:
                info["stop"] = f"{type(e).__name__}: {e}"[:200]
                info["cpu"] = round(time.process_time() - t0, 2)
            return False
    try:
        n = 0
        # measured about 0.15 s; far past that, the patch loops
        with _limit(3.0):
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
