"""Import-state fingerprint of the sqlglot package.

Some sqlglot modules run parser, tokenizer and generator code while they are
imported (class bodies that parse SQL templates, tables built by calling
helpers). A patch to a function those runs reach can change what the import
leaves behind, and then any test may see it. The selector treats such a
change as touching every importer, which is a very wide net.

This module imports every sqlglot module in a fixed order and reduces what
the imports left behind to one digest per module: module globals, class
dictionaries, function defaults and closure cells, and every object they
reach. When the digests on the patched tree equal the ones on the pinned
tree, the imports left the same state behind, so a test can only see the
change by running the changed lines itself, and the per-test line map
already says which tests do that.

  python impfp.py --baseline OUT.json   (build time, pinned tree)
"""

from __future__ import annotations

import enum
import hashlib
import json
import pkgutil
import sys
import types

_SCALARS = (type(None), bool, int, float, complex, str, bytes)
_SKIP_MOD = {"__builtins__", "__loader__", "__spec__", "__file__", "__cached__", "__path__",
             "__package__", "__name__"}
_SKIP_CLS = {"__dict__", "__weakref__", "_abc_impl"}

# The isinstance chain of _W.walk, in its order, as one category per type.
_CHAIN = (
    (enum.Enum, 1), (types.ModuleType, 2), (type, 3), (types.FunctionType, 4),
    ((staticmethod, classmethod), 5), (property, 6), (types.CodeType, 7),
    (types.BuiltinFunctionType, 8), ((list, tuple), 9), (dict, 10),
    ((set, frozenset), 11), (types.MethodType, 12),
)
_CAT: dict = {}


def _category(o) -> int:
    """Which branch of the chain `o` takes: what the isinstance calls in
    order would say. Cached per type, unless the object reports a
    __class__ other than its type, which isinstance also consults."""
    t = type(o)
    c = _CAT.get(t)
    if c is not None and getattr(o, "__class__", t) is t:
        return c
    c = 0
    for kind, n in _CHAIN:
        if isinstance(o, kind):
            c = n
            break
    if getattr(o, "__class__", t) is t:
        _CAT[t] = c
    return c


class _W:
    def __init__(self, loose: bool = False):
        self.loose = loose
        self.memo: dict = {}
        self.keep: list = []
        self.out: list = []

    def ref(self, o) -> bool:
        k = id(o)
        n = self.memo.get(k)
        if n is not None:
            self.out.append(f"R{n}")
            return True
        self.memo[k] = len(self.memo)
        self.keep.append(o)
        return False

    @staticmethod
    def key(v):
        """A run-independent sort key for a dict key or set member, or None."""
        if type(v) in _SCALARS:
            return (0, type(v).__name__, repr(v))
        if isinstance(v, enum.Enum):
            return (1, type(v).__module__ + "." + type(v).__qualname__, v.name)
        if isinstance(v, type):
            return (2, v.__module__, v.__qualname__)
        if type(v) is tuple:
            ks = [_W.key(x) for x in v]
            return None if None in ks else (3, tuple(ks))
        return None

    def sub(self, o) -> str:
        w = _W(self.loose)
        w.memo = dict(self.memo)
        w.keep = self.keep
        w.walk(o)
        return hashlib.sha256("\x1f".join(w.out).encode("utf-8", "surrogatepass")).hexdigest()

    def walk(self, o):
        out = self.out
        t = type(o)
        if t in _SCALARS:
            out.append(t.__name__[:2] + repr(o))
            return
        cat = _category(o)
        if cat == 1:
            out.append(f"E{t.__module__}.{t.__qualname__}.{o.name}")
            return
        if cat == 2:
            out.append("M" + o.__name__)
            return
        if cat == 3:
            out.append(f"C{o.__module__}.{o.__qualname__}")
            return
        if self.ref(o):
            return
        if cat == 4:
            out.append(f"F{o.__module__}.{o.__qualname__}")
            self.walk(o.__defaults__)
            self.walk(o.__kwdefaults__)
            cells = o.__closure__ or ()
            out.append(f"K{len(cells)}")
            for c in cells:
                try:
                    self.walk(c.cell_contents)
                except ValueError:
                    out.append("empty")
            d = getattr(o, "__dict__", None)
            if d:
                self.walk(dict(d))
            w = getattr(o, "__wrapped__", None)
            if w is not None:
                self.walk(w)
            return
        if cat == 5:
            out.append("S" + t.__name__)
            self.walk(o.__func__)
            return
        if cat == 6:
            out.append("P")
            self.walk((o.fget, o.fset, o.fdel))
            return
        if cat == 7:
            # code is what the AST diff compares; here only where it is held
            out.append(f"X{o.co_filename.rsplit('/sqlglot/', 1)[-1]}:{o.co_qualname}")
            return
        if cat == 8:
            out.append(f"B{getattr(o, '__module__', '')}.{o.__qualname__}")
            return
        if cat == 9:
            out.append(f"L{t.__qualname__}{len(o)}")
            seq = list(o)
            if self.loose:
                # a module whose pinned imports build sequences in a
                # different order from run to run: compare them as multisets
                keyed = []
                for v in seq:
                    k = self.key(v)
                    keyed.append(((9, self.sub(v)) if k is None else k, v))
                keyed.sort(key=lambda x: x[0])
                seq = [v for _, v in keyed]
            for v in seq:
                self.walk(v)
            return
        if cat == 10:
            out.append(f"D{t.__module__}.{t.__qualname__}{len(o)}")
            df = getattr(o, "default_factory", None)
            if df is not None:
                self.walk(df)
            items = list(o.items())
            keys = [self.key(k) for k, _ in items]
            if None not in keys:
                # insertion order follows set iteration over classes or
                # strings in many tables, which varies from run to run
                items = [kv for _, kv in sorted(zip(keys, items), key=lambda x: x[0])]
            for k, v in items:
                self.walk(k)
                self.walk(v)
            return
        if cat == 11:
            keyed = []
            for v in o:
                k = self.key(v)
                keyed.append(((9, self.sub(v)) if k is None else k, v))
            keyed.sort(key=lambda x: x[0])
            out.append(f"Z{t.__qualname__}{len(o)}")
            for _, v in keyed:
                self.walk(v)
            return
        if cat == 12:
            out.append("m")
            self.walk(o.__func__)
            self.walk(o.__self__)
            return
        mod = getattr(t, "__module__", "") or ""
        if t.__name__ == "Pattern" and mod == "re":
            out.append(f"re{o.pattern!r}{o.flags}")
            return
        if mod == "_thread" and t.__name__ in ("lock", "RLock"):
            out.append("lock")
            return
        if t.__name__ == "partial" and mod == "functools":
            out.append("fp")
            self.walk((o.func, o.args, o.keywords))
            return
        # anything else: its class, then its attributes
        out.append(f"O{mod}.{t.__qualname__}")
        attrs = []
        for c in t.__mro__:
            for n in c.__dict__.get("__slots__", ()):
                if isinstance(n, str) and n not in ("__dict__", "__weakref__") and n not in attrs:
                    attrs.append(n)
        vals = []
        for n in sorted(attrs):
            try:
                vals.append((n, getattr(o, n)))
            except AttributeError:
                pass
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            vals.extend(sorted(d.items(), key=lambda kv: str(kv[0])))
        elif not attrs and not mod.startswith("sqlglot"):
            r = repr(o)
            if " at 0x" in r:
                raise ValueError(f"opaque {mod}.{t.__qualname__}")
            out.append(r)
            return
        for n, v in vals:
            out.append(str(n))
            if n == "_hash":
                out.append("1" if v is not None else "0")
                continue
            self.walk(v)
        out.append(")")

    def cls(self, c, modname: str):
        """A class defined in the module: its own dictionary, nested classes too."""
        self.out.append(f"c{c.__qualname__}{[b.__module__ + '.' + b.__qualname__ for b in c.__bases__]}")
        if self.ref(c):
            return
        for k, v in list(c.__dict__.items()):
            if k in _SKIP_CLS:
                continue
            self.out.append(k)
            if isinstance(v, type) and v.__module__ == modname and v.__qualname__.startswith(c.__qualname__ + "."):
                self.cls(v, modname)
            else:
                self.walk(v)


def modules() -> list:
    import sqlglot

    names = ["sqlglot"]
    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if not info.name.endswith("__main__"):
            names.append(info.name)
    return sorted(set(names))


def fingerprint(loose=(), skip=()) -> dict:
    """{"module:name": digest} for every module global, "module": "!error"
    for a module that fails to import. Modules in `loose` compare their
    sequences regardless of order; keys in `skip` are left out."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    out: dict = {}
    import importlib

    loaded = {}
    for name in modules():
        try:
            loaded[name] = importlib.import_module(name)
        except BaseException as e:  # noqa: BLE001
            out[name] = f"!{type(e).__name__}: {e}"
    # digest after every import, so later imports that change earlier
    # modules' state are in the digest too
    def one(key, v, is_loose):
        if key in skip:
            return
        w = _W(is_loose)
        try:
            w.walk(v)
            out[key] = hashlib.sha256("\x1f".join(w.out).encode("utf-8", "surrogatepass")).hexdigest()[:20]
        except BaseException as e:  # noqa: BLE001
            out[key] = f"?{type(e).__name__}: {e}"

    def cls(prefix, c, modname, is_loose):
        # one digest per class attribute, nested classes the same way
        out[prefix] = "c" + ",".join(b.__module__ + "." + b.__qualname__ for b in c.__bases__)
        for k, v in list(c.__dict__.items()):
            if k in _SKIP_CLS:
                continue
            if isinstance(v, type) and v.__module__ == modname and v.__qualname__ == f"{c.__qualname__}.{k}":
                cls(f"{prefix}.{k}", v, modname, is_loose)
            else:
                one(f"{prefix}.{k}", v, is_loose)

    for name, mod in loaded.items():
        is_loose = name in loose
        for k, v in sorted(vars(mod).items()):
            if k in _SKIP_MOD:
                continue
            if isinstance(v, type) and v.__module__ == name and v.__qualname__ == k:
                cls(f"{name}:{k}", v, name, is_loose)
            else:
                one(f"{name}:{k}", v, is_loose)
    return out


def check(base: dict) -> bool:
    """True when every digest on this tree equals the pinned one."""
    digests = (base or {}).get("digests")
    if not digests or any(v[:1] in "!?" for v in digests.values()):
        return False
    try:
        return fingerprint(set(base["loose"]), set(base["skip"])) == digests
    except BaseException:  # noqa: BLE001
        return False


def _child(loose: list, out: str, run: int = 0):
    import os
    import subprocess

    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["IMPFP_PERTURB"] = str(run)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--one", json.dumps(loose), out],
                   check=True, env=env)
    with open(out) as f:
        return json.load(f)


def baseline(out: str, runs: int = 4) -> dict:
    """Digests from several fresh processes on the pinned tree. A module
    with a global that differs between them is compared loosely; a global
    that still differs holds a value the pinned tree does not fix (a loop
    variable left over from iterating a set), so it is left out."""
    import os

    tmp = out + ".tmp"
    strict = [_child([], tmp, i) for i in range(runs)]
    loose = sorted({k.split(":")[0] for k in strict[0] if any(r.get(k) != strict[0][k] for r in strict[1:])})
    res = [_child(loose, tmp, i) for i in range(runs)] if loose else strict
    os.unlink(tmp)
    keys = set().union(*res)
    skip = sorted(k for k in keys if ":" in k and len({r.get(k) for r in res}) > 1)
    digests = {k: v for k, v in res[0].items() if k not in skip}
    return {"loose": loose, "skip": skip, "digests": digests}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        # as under the test runner: sqlglot reads sys.modules for it at import
        import unittest  # noqa: F401

        import pytest  # noqa: F401

        # shift where the imports allocate, so orders that follow object
        # addresses (sets of classes) differ between the baseline runs
        _pad = [bytearray(64 + 48 * (i % 7)) for i in range(int(__import__("os").environ.get("IMPFP_PERTURB", "0")) * 3001)]
        with open(sys.argv[3], "w") as f:
            json.dump(fingerprint(set(json.loads(sys.argv[2]))), f)
    elif len(sys.argv) == 3 and sys.argv[1] == "--baseline":
        res = baseline(sys.argv[2])
        with open(sys.argv[2], "w") as f:
            json.dump(res, f, separators=(",", ":"), sort_keys=True)
        bad = {k: v for k, v in res["digests"].items() if v[:1] in "!?"}
        print(f"impfp: {len(res['digests'])} globals, loose {res['loose']}, skip {res['skip']}, "
              f"{len(bad)} unusable {list(bad.items())[:5]}", file=sys.stderr)
