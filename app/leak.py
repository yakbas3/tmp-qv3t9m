"""Shared-process state rule.

The labels come from one run of the whole suite, many tests per process. A
test can fail there only because an earlier test in its process left state
behind: a class attribute, a module global, the dialect registry, a cache.
The line map and lone verification cannot see that: the failing test need
not run a changed line, and it passes when run alone. Which tests fail
depends on how the run spread them over its workers, so the only safe
selection for such a patch is every test.

shared_state() reads the AST diff and fires, for the whole patch, when:

  1. the outermost changed function, old or new version, anywhere in its
     body writes state that outlives the call: a store, delete or mutating
     call on an object that is not local to the call (a global name, an
     imported module, cls, type(x), x.__class__, a class-level table
     reached through any object, self.x bound to such an object or
     assigned only in the class body, self in a metaclass), a `global`
     store, setattr, an import with a computed module name, sys.modules,
     a process-wide setter. Whole function on purpose:
     a changed line can steer an unchanged write.
  2. module-level changed code writes through an import-bound name.
  3. the changed code stops copying an object (x.copy(), dict(x), {**x},
     copy=False), so later in-place edits reach the original, or a header
     change touches a caching decorator, a mutable default or copy=.
  4. the changed function refers to one of the two objects that tests
     overwrite at the pinned commit: the dialect registry
     `_Dialect._classes` (cleared by tests/test_dialect_entry_points.py)
     and `sqlglot.schema` (rebound in tests/test_optimizer.py and
     tests/test_lineage.py).

A deleted module is left to the import rules. Pure function of the syntax
trees, no I/O.
"""

from __future__ import annotations

import ast
import re

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_UPPER = re.compile(r"^_{0,2}[A-Z]")
# lower-case names that hold class-level objects in the pinned package
_CLASS_ATTRS = {
    "arg_types", "_sql_names", "_classes", "required_args", "tokenizer_class", "parser_class",
    "generator_class", "jsonpath_tokenizer_class", "__dict__", "__class__", "__defaults__",
    "__kwdefaults__", "__annotations__", "__globals__", "__bases__", "__mro__", "__wrapped__",
}
MUTATORS = {
    "append", "extend", "insert", "pop", "popitem", "remove", "clear", "update", "setdefault",
    "add", "discard", "sort", "reverse", "appendleft", "popleft", "extendleft", "rotate",
    "difference_update", "intersection_update", "symmetric_difference_update", "move_to_end",
    "__setitem__", "__delitem__", "__setattr__", "__delattr__", "cache_clear",
    # sqlglot expressions edited in place
    "set", "replace", "unnest", "unalias",
}
# calls that set process-wide switches when made on a module or a global object
_SETTERS = {
    "chdir", "putenv", "unsetenv", "setrecursionlimit", "simplefilter", "filterwarnings",
    "resetwarnings", "seed", "setlocale", "set_int_max_str_digits", "setswitchinterval",
    "settrace", "setprofile", "setLevel", "disable", "basicConfig", "addHandler", "removeHandler",
    "register", "unregister", "invalidate_caches", "reload",
}
_STR_ONLY = {"replace"}  # also a str method: only counted on class-level objects
_COPY_CALLS = {"dict", "list", "set", "frozenset", "tuple", "sorted", "deepcopy", "maybe_copy", "OrderedDict"}
_VIEWS = {"get", "values", "items", "keys", "setdefault", "__getitem__"}
_CLS_NAMES = {"cls", "klass", "mcs", "mcls", "metacls"}
_CACHE_WORD = re.compile(r"cache|memo|singleton|once", re.I)
_FRESH = (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp,
          ast.Constant, ast.JoinedStr)


_SEEN: dict = {}  # id(node) -> (node, every node under it); filled per shared_state() call


def _w(n) -> list:
    hit = _SEEN.get(id(n))
    if hit is None or hit[0] is not n:
        hit = _SEEN[id(n)] = (n, list(ast.walk(n)))
    return hit[1]


def _walk(nodes):
    for n in nodes:
        yield from _w(n)


def _top_imports(tree):
    """Import statements that run at import time (not inside a def or class)."""
    stack = list(tree.body)
    while stack:
        st = stack.pop()
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            yield st
        elif not isinstance(st, _DEFS + (ast.ClassDef,)):
            for f in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(x for x in getattr(st, f, ()) or () if isinstance(x, ast.stmt) or isinstance(x, ast.ExceptHandler))


def _dump(n) -> str:
    return ast.dump(n)


def _show(n) -> str:
    try:
        s = ast.unparse(n)
    except Exception:  # noqa: BLE001
        s = type(n).__name__
    s = " ".join(s.split())
    return s if len(s) <= 60 else s[:57] + "..."


def _chain(e):
    """(root expression, attribute names from the root outwards; None for a subscript)."""
    attrs = []
    while True:
        if isinstance(e, ast.Attribute):
            attrs.append(e.attr)
            e = e.value
        elif isinstance(e, ast.Subscript):
            attrs.append(None)
            e = e.value
        elif isinstance(e, ast.Starred):
            e = e.value
        else:
            break
    attrs.reverse()
    return e, attrs


def _targets(t):
    if isinstance(t, (ast.Tuple, ast.List)):
        for x in t.elts:
            yield from _targets(x)
    elif isinstance(t, ast.Starred):
        yield from _targets(t.value)
    else:
        yield t


def _stores(node):
    """(target, value or None, augmented) for every binding or store the node itself makes."""
    if isinstance(node, ast.Assign):
        for t in node.targets:
            for x in _targets(t):
                yield x, node.value, False
    elif isinstance(node, ast.AnnAssign):
        if node.value is not None:
            yield node.target, node.value, False
    elif isinstance(node, ast.AugAssign):
        yield node.target, node.value, True
    elif isinstance(node, ast.NamedExpr):
        yield node.target, node.value, False
    elif isinstance(node, ast.Delete):
        for t in node.targets:
            for x in _targets(t):
                yield x, None, True
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        for x in _targets(node.target):
            yield x, node.iter, False
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for it in node.items:
            if it.optional_vars is not None:
                for x in _targets(it.optional_vars):
                    yield x, it.context_expr, False


class _Env:
    """What the names of one function (or of module-level code) stand for."""

    def __init__(self, tests: bool, module_level: bool = False):
        self.tests = tests
        self.module_level = module_level
        self.local: set = set()
        self.shared: set = set()  # import-bound, `global`, or a local bound to a shared object
        self.globals: set = set()
        self.cls: set = set()
        self.self_shared: set = set()  # x in self.x bound to a shared object
        self.state = lambda x: False  # x in self.x is class state

    def absorb(self, nodes):
        """Add the bindings made anywhere under the nodes."""
        for n in _walk(nodes):
            if isinstance(n, _FUNCS):
                a = n.args
                params = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
                for p in params:
                    self.local.add(p.arg)
                first = (a.posonlyargs + a.args)[:1]
                if first and (first[0].arg in _CLS_NAMES or isinstance(n, _DEFS) and (
                        n.name == "__init_subclass__" or any(
                            _is_plain(d, "classmethod") or isinstance(d, ast.Attribute) and d.attr == "classmethod"
                            for d in n.decorator_list))):
                    self.cls.add(first[0].arg)
                if isinstance(n, _DEFS):
                    self.local.add(n.name)
            elif isinstance(n, ast.ClassDef):
                self.local.add(n.name)
            elif isinstance(n, ast.Global):
                self.globals.update(n.names)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    self.shared.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler):
                if n.name:
                    self.local.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                self.local.add(n.id)
            elif isinstance(n, (ast.MatchAs, ast.MatchStar)):
                if n.name:
                    self.local.add(n.name)
        self.shared |= self.globals
        self.local -= self.shared

    def alias(self, nodes):
        """Locals bound straight to a shared object stand for that object."""
        for _ in range(3):
            grew = False
            for n in _walk(nodes):
                for t, v, _aug in _stores(n):
                    if isinstance(t, ast.Name) and v is not None and t.id in self.local \
                            and t.id not in self.cls and t.id != "self" and self.shared_obj(v, bind=True):
                        self.local.discard(t.id)
                        self.shared.add(t.id)
                        grew = True
                    elif isinstance(t, ast.Attribute) and _is_plain(t.value, "self") and v is not None \
                            and t.attr not in self.self_shared and self.shared_obj(v, bind=True):
                        self.self_shared.add(t.attr)
                        grew = True
            if not grew:
                break

    def shared_obj(self, e, bind: bool = False) -> bool:
        """True when the expression evaluates to an object that outlives the call."""
        if isinstance(e, ast.IfExp):
            return self.shared_obj(e.body, bind) or self.shared_obj(e.orelse, bind)
        if isinstance(e, ast.BoolOp):
            return any(self.shared_obj(v, bind) for v in e.values)
        if isinstance(e, ast.NamedExpr):
            return self.shared_obj(e.value, bind)
        base, attrs = _chain(e)
        if self.module_level:
            return isinstance(base, ast.Name) and base.id in self.shared
        named = [a for a in attrs if a is not None]
        if any(a in _CLASS_ATTRS or _UPPER.match(a) for a in named):
            return True
        if isinstance(base, ast.Name):
            n = base.id
            if n in self.shared:
                # an import-bound or global name; a bare class or function taken
                # as a value (x = exp.Column) is not a container being aliased
                return not bind or bool(attrs) or n in self.globals
            if n in self.cls:
                return not self.tests and not bind
            if n == "self" and attrs and attrs[0] is not None and (attrs[0] in self.self_shared or self.state(attrs[0])):
                return True
            if n in self.local or n == "self":
                return False
            if bind:
                return n.isupper() or bool(attrs)
            return True
        if isinstance(base, ast.Call):
            f = base.func
            if isinstance(f, ast.Name):
                if f.id in ("type", "globals"):
                    return True
                if f.id in ("vars", "getattr") and base.args:
                    return self.shared_obj(base.args[0], bind)
                return False
            if isinstance(f, ast.Attribute) and f.attr in _VIEWS:
                return self.shared_obj(f.value)
        return False


def _is_plain(e, name: str) -> bool:
    return isinstance(e, ast.Name) and e.id == name


def _write_of(n, env: _Env):
    """Why this single node writes shared state, or None."""
    for t, _v, aug in _stores(n):
        if isinstance(t, ast.Name):
            if t.id in env.globals and not env.module_level:
                return f"global {t.id}"
            if aug and t.id in env.shared and not env.module_level and isinstance(n, ast.AugAssign):
                return f"in-place {t.id}"
        elif isinstance(t, (ast.Attribute, ast.Subscript)):
            if env.shared_obj(t.value):
                return f"store {_show(t)}"
    if isinstance(n, ast.Call):
        f = n.func
        if isinstance(f, ast.Attribute):
            if f.attr in MUTATORS and env.shared_obj(f.value):
                if f.attr in _STR_ONLY:
                    _b, attrs = _chain(f.value)
                    if not any(a is not None and (a in _CLASS_ATTRS or a.isupper()) for a in attrs):
                        return None
                return f"call {_show(f)}"
            if f.attr in _SETTERS and not env.module_level:
                base, _a = _chain(f.value)
                while isinstance(base, ast.Call):  # logging.getLogger(...).setLevel(...)
                    base, _a = _chain(base.func)
                if isinstance(base, ast.Name) and base.id not in env.local and base.id not in ("self", "super"):
                    return f"call {_show(f)}"
            if f.attr in ("import_module", "__import__") and not (
                    n.args and isinstance(n.args[0], ast.Constant)):
                return "import by computed name"
        elif isinstance(f, ast.Name):
            if f.id in ("setattr", "delattr") and n.args:
                a0 = n.args[0]
                if env.module_level:
                    if env.shared_obj(a0):
                        return f"{f.id} {_show(a0)}"
                elif not (_is_plain(a0, "self") and "self" not in env.cls) and not (env.tests and isinstance(a0, ast.Name) and a0.id in env.cls):
                    if env.shared_obj(a0) or not isinstance(a0, ast.Name):
                        return f"{f.id} {_show(a0)}"
            if f.id in ("globals", "exec"):
                return f"{f.id}()"
            if f.id in ("import_module", "__import__") and not (
                    n.args and isinstance(n.args[0], ast.Constant)):
                return "import by computed name"
    if isinstance(n, ast.Attribute) and n.attr == "modules" and _is_plain(n.value, "sys"):
        return "sys.modules"
    return None


def _say(who: str, what: str, w: str) -> str:
    if w == "sys.modules":  # a read of it already makes the outcome depend on earlier imports
        what = "reads or writes import state"
    return f"{who} {what} ({w})"


def _writes(nodes, env: _Env) -> list:
    out = []
    for n in _walk(nodes):
        w = _write_of(n, env)
        if w:
            out.append((n, w))
    return out


def _registry_ref(nodes, rel: str):
    """A read or write of an object that tests overwrite at the pinned commit."""
    depth = rel.count("/") - 1 if rel.startswith("sqlglot/") else None
    for n in _walk(nodes):
        if isinstance(n, ast.Attribute):
            if n.attr == "_classes":
                return "the dialect registry"
            if n.attr == "classes" and isinstance(n.value, ast.Name) and n.value.id in ("Dialect", "_Dialect", "cls"):
                return "the dialect registry"
            if n.attr == "schema" and _is_plain(n.value, "sqlglot"):
                return "sqlglot.schema"
        elif isinstance(n, ast.ImportFrom):
            top = (n.level == 0 and n.module == "sqlglot") or \
                (n.module is None and depth is not None and n.level == depth + 1)
            if top and any(al.name == "schema" for al in n.names):
                return "sqlglot.schema"
        elif isinstance(n, ast.Import):
            if any(al.name == "sqlglot.schema" and al.asname for al in n.names):
                return "sqlglot.schema"
    return None


# ------------------------------------------------------------- copies


def _copies(nodes) -> list:
    """Dumps of the expressions the nodes copy."""
    out = []
    for n in _walk(nodes):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in ("copy", "deepcopy", "__copy__", "__deepcopy__"):
                if f.attr == "copy" and not n.args:
                    out.append(f.value)
                elif n.args:
                    out.append(n.args[0])
            elif isinstance(f, ast.Name) and f.id in _COPY_CALLS:
                if n.args:
                    out.append(n.args[0])
                if f.id in ("dict", "OrderedDict"):  # dict(ENV, **env) copies both
                    out.extend(k.value for k in n.keywords if k.arg is None)
            elif isinstance(f, ast.Name) and f.id == "copy" and n.args:
                out.append(n.args[0])
        elif isinstance(n, ast.Dict):
            out.extend(v for k, v in zip(n.keys, n.values) if k is None)
        elif isinstance(n, (ast.List, ast.Set, ast.Tuple)) and isinstance(getattr(n, "ctx", ast.Load()), ast.Load):
            out.extend(e.value for e in n.elts if isinstance(e, ast.Starred))
        elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and isinstance(n.ctx, ast.Load) \
                and n.slice.lower is None and n.slice.upper is None and n.slice.step is None:
            out.append(n.value)
    return out


def _copy_flags(nodes) -> list:
    out = []
    for n in _walk(nodes):
        if isinstance(n, ast.keyword) and n.arg == "copy":
            out.append(_dump(n.value))
    return out


def _mentions_copy_flag(nodes) -> bool:
    """A statement steered by a `copy` argument, outside the call keywords."""
    for st in nodes:
        kw = {id(x) for n in _w(st) if isinstance(n, ast.keyword) and n.arg == "copy" for x in _w(n.value)}
        for n in _w(st):
            if isinstance(n, ast.Name) and n.id == "copy" and isinstance(n.ctx, ast.Load) and id(n) not in kw:
                return True
    return False


def _minus(a: list, b: list) -> list:
    b = list(b)
    out = []
    for x in a:
        if x in b:
            b.remove(x)
        else:
            out.append(x)
    return out


def _copy_drop(old_nodes, new_nodes):
    """Why the new statements copy less than the old ones did, or None.
    Returns (reason, dropped expression or None)."""
    of, nf = _copy_flags(old_nodes), _copy_flags(new_nodes)
    true = _dump(ast.Constant(value=True))
    false = _dump(ast.Constant(value=False))
    # copying more is safe: a new copy=True, or an old copy=False gone
    if [x for x in _minus(nf, of) if x != true] or [x for x in _minus(of, nf) if x != false]:
        return "copy= argument changed", None
    # a copy of a display or a comprehension guards nothing: the value is new each time
    oc = [x for x in _copies(old_nodes) if not isinstance(x, _FRESH)]
    nc = [_dump(x) for x in _copies(new_nodes)]
    for x in oc:
        d = _dump(x)
        if d in nc:
            nc.remove(d)
            continue
        # the copy is gone: does the original still flow somewhere?
        tx = type(x)
        bare_old = sum(1 for n in _walk(old_nodes) if type(n) is tx and _dump(n) == d)
        bare_new = sum(1 for n in _walk(new_nodes) if type(n) is tx and _dump(n) == d)
        rebinds = any(isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
            _dump(t) == d.replace("ctx=Load()", "ctx=Store()", 1) or
            (isinstance(t, ast.Name) and isinstance(x, ast.Name) and t.id == x.id)
            for t, _v, _a in _stores(n)) for n in _walk(old_nodes))
        if bare_new >= bare_old and bare_new > 0 or rebinds:
            return f"copy of {_show(x)} dropped", x
    if _mentions_copy_flag(old_nodes) != _mentions_copy_flag(new_nodes) or (
            _mentions_copy_flag(old_nodes) and [_dump(n) for n in old_nodes] != [_dump(n) for n in new_nodes]):
        return "code steered by a copy flag changed", None
    return None


# ------------------------------------------------------------- headers


def _mutable_default(d) -> bool:
    if d is None or isinstance(d, (ast.Constant, ast.Name, ast.Attribute, ast.Lambda)):
        return False
    if isinstance(d, ast.Tuple):
        return any(_mutable_default(x) for x in d.elts)
    if isinstance(d, ast.UnaryOp):
        return _mutable_default(d.operand)
    if isinstance(d, ast.BinOp):
        return _mutable_default(d.left) or _mutable_default(d.right)
    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in ("frozenset", "tuple", "str", "int", "float", "bool", "bytes"):
        return False
    return True


def _defaults(fn) -> dict:
    a = fn.args
    pos = a.posonlyargs + a.args
    out = {}
    for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        out[p.arg] = d
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[p.arg] = d
    return out


def _header_change(o, n):
    """Decorators and defaults of a function whose header changed."""
    od = [_dump(d) for d in o.decorator_list]
    nd = [_dump(d) for d in n.decorator_list]
    if od != nd:
        words = " ".join(_show(d) for d in o.decorator_list + n.decorator_list
                         if _dump(d) not in od or _dump(d) not in nd)
        if _CACHE_WORD.search(words):
            return f"caching decorator changed ({words})"
    odf, ndf = _defaults(o), _defaults(n)
    for p, d in ndf.items():
        if p in odf and _dump(odf[p]) == _dump(d):
            continue
        if p == "copy":
            return "default of copy changed"
        if _mutable_default(d):
            return f"mutable default for {p}"
    if "copy" in odf and "copy" not in ndf:
        return "default of copy changed"
    return None


def _outer_func(site):
    for s in site.scope:
        if s[0] == "func":
            return s[2]
    return None


def _stmts(nodes) -> list:
    return [x if isinstance(x, ast.stmt) else ast.Expr(x) for x in nodes if isinstance(x, ast.AST)]


_META_BASES = {"type", "ABCMeta", "EnumMeta", "EnumType"}


class _Classes:
    """The classes of one file, old and new tree. state(method, x): x is
    assigned in the class body and never stored on self anywhere in that
    class, so self.x is the one object all instances share. The self
    stores are collected only when a class-body name is asked about."""

    def __init__(self, trees):
        self.nodes: dict = {}  # qualified name -> ClassDef nodes
        self.body: dict = {}
        self.bases: dict = {}
        self.owner: dict = {}  # id(method) -> qualified name
        self.stored: dict = {}
        for tree in trees:
            self._visit(tree.body, ())
        meta = {q for q, b in self.bases.items() if b & _META_BASES}
        while True:
            more = {q for q, b in self.bases.items() if q not in meta and b & {m[-1] for m in meta}}
            if not more:
                break
            meta |= more
        self.meta = meta

    def _visit(self, stmts, qual):
        for st in stmts:
            if not isinstance(st, ast.ClassDef):
                continue
            q = qual + (st.name,)
            self.nodes.setdefault(q, []).append(st)
            self.bases.setdefault(q, set()).update(
                b.id if isinstance(b, ast.Name) else b.attr for b in st.bases if isinstance(b, (ast.Name, ast.Attribute)))
            names = self.body.setdefault(q, set())
            if not any("dataclass" in ast.dump(d) for d in st.decorator_list):  # fields are per instance
                for x in st.body:
                    if isinstance(x, (ast.Assign, ast.AnnAssign)):
                        names.update(t.id for t, _v, _a in _stores(x) if isinstance(t, ast.Name))
            self.owner.update((id(x), q) for x in st.body if isinstance(x, _DEFS))
            self._visit(st.body, q)

    def is_meta(self, fn) -> bool:
        return self.owner.get(id(fn)) in self.meta

    def state(self, fn, x: str) -> bool:
        q = self.owner.get(id(fn))
        if q is None or x not in self.body[q]:
            return False
        mine = self.stored.get(q)
        if mine is None:
            mine = self.stored[q] = set()
            for c in self.nodes[q]:
                for n in ast.walk(c):
                    if isinstance(n, ast.Attribute) and _is_plain(n.value, "self") and isinstance(n.ctx, (ast.Store, ast.Del)):
                        mine.add(n.attr)
                    elif isinstance(n, ast.Call) and _is_plain(n.func, "setattr") and n.args and _is_plain(n.args[0], "self"):
                        a1 = n.args[1] if len(n.args) > 1 else None
                        mine.add(a1.value if isinstance(a1, ast.Constant) and isinstance(a1.value, str) else None)
        return x not in mine and None not in mine


def _func_check(fn, extra: list, tests: bool, rel: str, info=None):
    """Whole-function check. fn is the old version; with the patched
    statements added it stands for the old and the new version together."""
    env = _Env(tests)
    if info is not None:
        env.state = lambda x: info.state(fn, x)
        if info.is_meta(fn):
            env.cls.add("self")  # self in a metaclass is the class
    env.absorb([fn] + extra)
    env.alias([fn] + extra)
    ws = _writes([fn] + extra, env)
    if ws:
        return _say("changed function", "writes shared state", ws[0][1])
    ref = _registry_ref([fn] + extra, rel)
    if ref:
        return f"changed function uses {ref}"
    return None


def shared_state(rel: str, old_tree, new_tree, sites):
    """Why every test must be selected for this file's changes, or None."""
    if not rel.endswith(".py") or not (rel.startswith("sqlglot/") or rel.startswith("tests/")):
        return None
    tests = rel.startswith("tests/")
    if old_tree.body and not new_tree.body:
        # a deleted module: its importers fail outright, other rules see that
        return None

    _SEEN.clear()
    try:
        return _shared_state(rel, old_tree, new_tree, sites, tests)
    finally:
        _SEEN.clear()


def _shared_state(rel, old_tree, new_tree, sites, tests):
    module_env = _Env(tests, module_level=True)
    for tree in (old_tree, new_tree):
        for n in _top_imports(tree):
            for al in n.names:
                module_env.shared.add((al.asname or al.name).split(".")[0])

    info = _Classes((old_tree, new_tree))
    by_func: dict = {}
    for site in sites:
        old_nodes = [] if site.kind == "insert" else list(site.nodes)
        new_nodes = list(site.nodes) if site.kind == "insert" else list(site.new or [])
        fn = _outer_func(site)
        if fn is not None:
            g = by_func.setdefault(id(fn), [fn, [], []])
            g[1].extend(old_nodes)
            g[2].extend(new_nodes)
            continue
        # module or class level: runs at import. Functions that were added,
        # removed or re-headed count as changed function code, all of it.
        for st in _stmts(old_nodes) + _stmts(new_nodes):
            inner = [n for n in _w(st) if isinstance(n, _FUNCS)]
            nested = {id(x) for f in inner for x in _w(f) if x is not f}
            outer = [f for f in inner if id(f) not in nested]
            for f in outer:
                why = _func_check(f, [], tests, rel, info)
                if why:
                    return why
            skip = {id(x) for f in outer for x in _w(f)}
            for n in _w(st):
                if id(n) in skip:
                    continue
                w = _write_of(n, module_env)
                if w:
                    return _say("import-time code", "writes another module's state", w)
            if not isinstance(st, _DEFS + (ast.ClassDef,)):
                ref = _registry_ref([st], rel)
                if ref:
                    return f"import-time code uses {ref}"
        if site.kind == "header" and old_nodes and new_nodes and isinstance(old_nodes[0], _DEFS) \
                and isinstance(new_nodes[0], _DEFS):
            why = _header_change(old_nodes[0], new_nodes[0])
            if why:
                return why
        if not all(isinstance(x, _DEFS + (ast.ClassDef,)) for x in old_nodes + new_nodes):
            cd = _copy_drop(_stmts(old_nodes), _stmts(new_nodes))
            if cd:
                return cd[0]

    for fn, old_nodes, new_nodes in by_func.values():
        why = _func_check(fn, _stmts(new_nodes), tests, rel, info)
        if why:
            return why
        cd = _copy_drop(_stmts(old_nodes), _stmts(new_nodes))
        if cd:
            return cd[0]
    return None
