"""Recorder shared by the TPC evidence tools (rq_plugin.py, rq_profile.py).

Used only to build app/rq.json.gz, never at scoring time. It records, for
the code the TPC executor tests run outside the front end that app/proxy.py
replays and fingerprints:

  - the lines run (sys.monitoring), filed under the current region
  - the classes instantiated (MRO of every instance whose class has a repo
    __init__, as tools/map_plugin.py does)
  - the classes of what the front end leaves behind: every object in a
    value it returns, and every instance it created that is still alive
    once the call is over (found by a scan of the collector's objects)

The front end is a set of calls wrapped with front(). Lines run inside one
are not recorded. An instance created inside one counts only if it outlives
the call. A call wrapped with handed=True returns a value whose only reader
is another front-end call (a parsed tree on its way to pickle): what it
returns is not counted.
"""

from __future__ import annotations

import builtins
import functools
import gc
import os
import sys
import threading
import types as pytypes

from firstline import first_line

ROOT = os.path.realpath(os.environ.get("MAP_ROOT", os.getcwd())) + os.sep
TOOL = 3
mon = sys.monitoring

_local = threading.local()
_regions: dict = {}
_cur = ["rest"]
OFF = [False]   # True: record nothing at all (timing runs)
_wrapped: list = []
_HAVE_GC = 1 << 14
_ATOMS = frozenset((type(None), bool, int, float, complex, str, bytes, range))
_SLOTS: dict = {}


def _region(name: str):
    r = _regions.get(name)
    if r is None:
        r = _regions[name] = (set(), set())
    return r


def begin(name: str):
    """File what follows under region `name`."""
    _cur[0] = name
    _region(name)
    mon.restart_events()


def take(name: str):
    """(lines, type keys) of a region, which is then emptied."""
    lines, types = _region(name)
    out = (sorted(lines), sorted(type_keys(types)))
    lines.clear()
    types.clear()
    return out


def paused() -> bool:
    return getattr(_local, "depth", 0) > 0


def _line(code, line):
    try:
        if getattr(_local, "depth", 0) > 0:
            return None
        fn = code.co_filename
        if fn.startswith(ROOT):
            _regions[_cur[0]][0].add((fn[len(ROOT):], line))
        return mon.DISABLE
    except Exception:
        return None


def _start_ev(code, offset):
    try:
        if getattr(_local, "depth", 0) > 0:
            return None
        fn = code.co_filename
        if fn.startswith(ROOT):
            ln = first_line(code)
            if ln is not None:
                _regions[_cur[0]][0].add((fn[len(ROOT):], ln))
        return mon.DISABLE
    except Exception:
        return None


def _wrap_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        try:
            if getattr(_local, "depth", 0) > 0:
                _local.made[id(self)] = type(self)
            else:
                _regions[_cur[0]][1].add(type(self))
        except Exception:
            pass
        return init(self, *args, **kwargs)

    __init__.__rq_wrapped__ = True
    return __init__


_orig_build_class = builtins.__build_class__


def _build_class(func, name, *bases, **kwargs):
    cls = _orig_build_class(func, name, *bases, **kwargs)
    try:
        if getattr(func, "__code__", None) and func.__code__.co_filename.startswith(ROOT):
            init = cls.__dict__.get("__init__")
            if callable(init) and hasattr(init, "__code__") and not getattr(init, "__rq_wrapped__", False):
                type.__setattr__(cls, "__init__", _wrap_init(init))
                _wrapped.append((cls, init))
    except Exception:
        pass
    return cls


_installed = [False]


def install():
    """Hook class creation. Call before sqlglot is imported, so that every
    class body is seen. Nothing is recorded until start()."""
    if _installed[0]:
        return
    _installed[0] = True
    _region("rest")
    builtins.__build_class__ = _build_class


def start():
    """Import the whole package, then start recording. What an import runs
    is left out, as in the map: the selector treats import-time code and
    class bodies by other means, and a lazy import would otherwise file a
    module's whole body under whichever query triggered it."""
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith("__main__"):
            continue
        try:
            __import__(info.name)
        except Exception as e:  # noqa: BLE001
            print(f"rq: cannot import {info.name}: {e}", file=sys.stderr)
    for lines, types in _regions.values():
        lines.clear()
        types.clear()
    try:
        mon.use_tool_id(TOOL, "rq")
    except ValueError:
        return
    mon.register_callback(TOOL, mon.events.LINE, _line)
    mon.register_callback(TOOL, mon.events.PY_START, _start_ev)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)


def stop():
    """Undo the recording, so that what runs next runs at its own speed."""
    OFF[0] = True
    mon.set_events(TOOL, 0)
    for cls, init in _wrapped:
        type.__setattr__(cls, "__init__", init)


def _slots(t) -> tuple:
    s = _SLOTS.get(t)
    if s is None:
        names: list = []
        for c in t.__mro__:
            sl = c.__dict__.get("__slots__", ())
            if isinstance(sl, str):
                sl = (sl,)
            for n in sl:
                if n not in ("__dict__", "__weakref__") and n not in names:
                    names.append(n)
        s = _SLOTS[t] = tuple(names)
    return s


_NO_DESCENT = (type, pytypes.ModuleType, pytypes.FunctionType, pytypes.BuiltinFunctionType,
               pytypes.MethodType)


def walk(root, types: set | None = None, foreign: bool = False) -> set:
    """ids of every object reachable from root through containers and repo
    objects; their classes are added to `types`. With foreign=True the
    attributes of other objects are followed too (a pool's call item)."""
    seen: set = set()
    stack = [root]
    while stack:
        o = stack.pop()
        t = type(o)
        if t in _ATOMS:
            continue
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
        if isinstance(o, _NO_DESCENT):
            continue
        mod = getattr(t, "__module__", "") or ""
        if not mod.startswith(("sqlglot", "tests")):
            if foreign:
                d = getattr(o, "__dict__", None)
                if isinstance(d, dict):
                    stack.extend(d.values())
            continue
        if types is not None:
            types.add(t)
        d = getattr(o, "__dict__", None)
        if d:
            stack.extend(d.values())
        for name in _slots(t):
            try:
                stack.append(getattr(o, name))
            except AttributeError:
                pass
    return seen


def _survivors(made: dict, ret, handed: bool):
    types = _regions[_cur[0]][1]
    skip: set = set()
    if handed:
        skip = walk(ret, foreign=True)
    else:
        walk(ret, types)
    if not made:
        return
    for i, t in made.items():
        if not (t.__flags__ & _HAVE_GC) and i not in skip:
            types.add(t)  # the collector cannot see it: assume it lives on
    gc.collect()
    for o in gc.get_objects():
        t = made.get(id(o))
        if t is not None and t is type(o) and id(o) not in skip:
            types.add(t)


def front(fn, handed: bool = False):
    """Wrap one front-end call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if OFF[0]:
            return fn(*args, **kwargs)
        depth = getattr(_local, "depth", 0)
        if depth == 0:
            _local.made = {}
        _local.depth = depth + 1
        ret = None
        try:
            ret = fn(*args, **kwargs)
            return ret
        finally:
            _local.depth -= 1
            if _local.depth == 0:
                made, _local.made = _local.made, {}
                try:
                    _survivors(made, ret, handed)
                except Exception as e:  # noqa: BLE001
                    print(f"rq: survivor scan failed: {type(e).__name__}: {e}", file=sys.stderr)
                    _regions[_cur[0]][1].update(made.values())

    wrapper.__rq_front__ = True
    return wrapper


COLD: dict = {}  # name -> value to return in place of the front-end call


def patch_executor():
    """Mark what execute() does before it runs the plan as front end:
    ensure_schema, optimize, the debug SQL and Plan."""
    import sqlglot.executor as ex
    from sqlglot import exp

    if getattr(ex.optimize, "__rq_front__", False):
        return

    def swap(name):
        real = front(getattr(ex, name))

        @functools.wraps(real)
        def call(*args, **kwargs):
            if name in COLD:
                ret = COLD[name]
                walk(ret, _regions[_cur[0]][1])
                return ret
            return real(*args, **kwargs)

        call.__rq_front__ = True
        setattr(ex, name, call)

    for name in ("ensure_schema", "optimize", "Plan"):
        swap(name)
    code = ex.execute.__code__
    owner = next(k for k in exp.Expression.__mro__ if "sql" in k.__dict__)
    orig = owner.__dict__["sql"]
    wrapped = front(orig)

    @functools.wraps(orig)
    def sql(self, *args, **kwargs):
        if sys._getframe(1).f_code is code:
            if "sql" in COLD:
                return ""
            return wrapped(self, *args, **kwargs)
        return orig(self, *args, **kwargs)

    type.__setattr__(owner, "sql", sql)


def type_keys(types) -> set:
    keys = set()
    for t in types:
        for k in getattr(t, "__mro__", ()):
            mod = sys.modules.get(getattr(k, "__module__", ""), None)
            f = getattr(mod, "__file__", None) or ""
            if os.path.realpath(f).startswith(ROOT):
                keys.add(f"{k.__module__}:{k.__qualname__}")
    return keys
