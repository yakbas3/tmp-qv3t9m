"""pytest plugin: per test, the values compared with a literal in sqlglot.

Used only to build app/cval.json.gz, never at scoring time. Every
single-operator comparison in sqlglot/ of the form `expr OP literal` (or
`literal OP expr`) with OP one of ==, !=, <, <=, >, >=, in, not in gets its
non-literal operand wrapped in a recorder that notes the value it had, then
hands it on unchanged. A patch that only changes that literal or that operator
changes the comparison's result only for values where old and new disagree, so
a test whose recorded values all agree runs as on the pinned tree.

Records are per test. Values seen during a test's setup or teardown are
charged to every test of its file (setUpClass runs once, before the first
test of a class). Values seen outside any test (collection, imports) mark
the site as seen at import ("import_t" when code of the test suite, such
as a test module's body, was on the stack). A value that is not a short scalar, or a test
that saw too many distinct values at a site, is recorded as "*".

Writes JSON to $CVAL_OUT/cval-<pid>.json. Rebuild:

  docker run --rm --network none -v $PWD/entry/tools:/tools:ro -v $OUT:/out \\
    -e CVAL_OUT=/out -e CVAL_ROOT=/opt/sqlglot -e PYTHONPATH=/tools:/opt/sqlglot \\
    -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -w /opt/sqlglot <entry image> \\
    python -m pytest -q -p cval_plugin -p xdist -n 8 --dist loadfile \\
    -p no:cacheprovider -o addopts= tests
  python3 entry/tools/cval_merge.py $OUT/cval-*.json
"""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import sys

ROOT = os.path.realpath(os.environ.get("CVAL_ROOT", os.getcwd())) + os.sep
PKG = ROOT + "sqlglot" + os.sep
CAP = 24  # distinct values per test and site before "*"

_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
_SCALAR = (str, int, float, bool, bytes, type(None))

CUR = [None]      # (owner key, phase) of the running test
REC: dict = {}    # site -> owner -> set of value reprs, or "*"
IMPORT: set = set()
IMPORT_T: set = set()  # seen outside a test with test-suite code on the stack
TESTS_DIR = ROOT + "tests" + os.sep
OUTCOMES: dict = {}


def __cval__(site, v):
    cur = CUR[0]
    if cur is None:
        if site not in IMPORT_T:
            f = sys._getframe(1)
            while f is not None:
                # an import statement in between: the package's own import
                # runs this, which the import state check covers
                if f.f_code.co_filename.startswith("<frozen importlib"):
                    break
                if f.f_code.co_filename.startswith(TESTS_DIR):
                    IMPORT_T.add(site)
                    break
                f = f.f_back
        IMPORT.add(site)
        return v
    d = REC.get(site)
    if d is None:
        d = REC[site] = {}
    s = d.get(cur)
    if s == "*":
        return v
    if s is None:
        s = d[cur] = set()
    t = type(v)
    if t not in _SCALAR or (t in (str, bytes) and len(v) > 64):
        d[cur] = "*"
        return v
    try:
        r = repr(v)
    except Exception:  # noqa: BLE001
        d[cur] = "*"
        return v
    s.add(r)
    if len(s) > CAP:
        d[cur] = "*"
    return v


def _literal(n) -> bool:
    try:
        ast.literal_eval(n)
    except Exception:  # noqa: BLE001
        return False
    return True


class _Rewrite(ast.NodeTransformer):
    def __init__(self, rel: str):
        self.rel = rel

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or not isinstance(node.ops[0], _OPS):
            return node
        left, right = node.left, node.comparators[0]
        key = f"{self.rel}:{node.lineno}:{node.col_offset}"
        if _literal(right) and not _literal(left):
            node.left = ast.copy_location(ast.Call(func=ast.Name(id="__cval__", ctx=ast.Load()),
                                                   args=[ast.Constant(key), left], keywords=[]), left)
        elif _literal(left) and not _literal(right):
            node.comparators[0] = ast.copy_location(
                ast.Call(func=ast.Name(id="__cval__", ctx=ast.Load()),
                         args=[ast.Constant(key), right], keywords=[]), right)
        return node


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        src = self.get_data(path)
        rel = os.path.relpath(os.path.realpath(path), ROOT)
        tree = ast.parse(src, filename=path)
        tree = _Rewrite(rel).visit(tree)
        ast.fix_missing_locations(tree)
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


builtins.__cval__ = __cval__
if "sqlglot" in sys.modules:
    raise RuntimeError("cval_plugin must load before sqlglot")
sys.meta_path.insert(0, _Finder())
concurrent.futures.ProcessPoolExecutor = _SyncPool
import concurrent.futures.process as _cfp  # noqa: E402

_cfp.ProcessPoolExecutor = _SyncPool


def _owner(item) -> str:
    # the test's file: setUpClass, setUpModule and their teardowns run in
    # the setup or teardown of whichever test of the file comes first or last
    return "M:" + item.nodeid.split("::", 1)[0]


def pytest_runtest_setup(item):
    CUR[0] = _owner(item)


def pytest_runtest_call(item):
    CUR[0] = item.nodeid


def pytest_runtest_teardown(item):
    CUR[0] = _owner(item)


def pytest_runtest_logreport(report):
    if report.failed:
        OUTCOMES[report.nodeid] = "failed"
    else:
        OUTCOMES.setdefault(report.nodeid, "ok")
    if report.when == "teardown":
        CUR[0] = None


def pytest_sessionfinish(session):
    out = os.path.join(os.environ["CVAL_OUT"], f"cval-{os.getpid()}.json")
    with open(out, "w") as f:
        json.dump({
            "rec": {k: {o: (s if s == "*" else sorted(s)) for o, s in d.items()} for k, d in REC.items()},
            "import": sorted(IMPORT),
            "import_t": sorted(IMPORT_T),
            "outcomes": OUTCOMES,
        }, f)
