"""Blast Radius selector.

For each patch in /data/commits:
  1. Parse and apply the diff in pure Python against the pinned sqlglot tree.
  2. Find what changed with an AST diff (comment and formatting edits vanish).
  3. Turn each change into candidate tests with the map built from the pinned
     commit (per-test line coverage, file reads, instantiated classes).
  4. Run the cheap candidates against the patched tree and keep only the ones
     that fail. Expensive or unverified candidates are kept as they are.
  5. Pack into workers, then fill idle worker time with extra tests for free.

The map is tools/build_map.py output. Answers are never read.
"""

from __future__ import annotations

import ast
import bisect
import difflib
import gzip
import json
import os
import re
import sys
import time

T0 = time.monotonic()
APP = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("BLAST_SRC", "/opt/sqlglot")
DATA = os.environ.get("BLAST_DATA", "/data")
TMP = os.environ.get("BLAST_TMP", "/tmp")
WORKER_MS = 2000
MAX_WORKERS = 8
PARALLEL = 2
PROXY_MIN_MS = 3000  # tests cheaper than this are not worth the proxy run

CONFIG_FILES = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py", "setup.py"}


def log(*args):
    print(f"[{time.monotonic() - T0:6.1f}s]", *args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- patches


class FileDiff:
    def __init__(self):
        self.old = None  # relative path or None for a new file
        self.new = None  # relative path or None for a deleted file
        self.hunks = []  # (old_start, old_len, new_start, new_len, [lines])
        self.binary = False


def _strip_prefix(path: str) -> str | None:
    path = path.split("\t")[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur = None
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileDiff()
            files.append(cur)
            parts = line.rstrip("\n").split(" ")
            if len(parts) >= 4:
                cur.old = _strip_prefix(parts[2])
                cur.new = _strip_prefix(parts[3])
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur is None or cur.hunks:
                cur = FileDiff()
                files.append(cur)
            cur.old = _strip_prefix(line[4:])
            cur.new = _strip_prefix(lines[i + 1][4:])
            i += 2
            continue
        if cur is not None and line.startswith(("new file mode", "deleted file mode")):
            if line.startswith("new"):
                cur.old = None
            else:
                cur.new = None
            i += 1
            continue
        if cur is not None and (line.startswith("Binary files") or line.startswith("GIT binary patch")):
            cur.binary = True
            i += 1
            continue
        if cur is not None and line.startswith("rename from "):
            cur.old = line[len("rename from "):].strip()
            i += 1
            continue
        if cur is not None and line.startswith("rename to "):
            cur.new = line[len("rename to "):].strip()
            i += 1
            continue
        m = HUNK_RE.match(line)
        if m and cur is not None:
            os_, ol, ns, nl = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
            body = []
            i += 1
            o_left, n_left = ol, nl
            while i < len(lines) and (o_left > 0 or n_left > 0 or lines[i].startswith("\\")):
                hl = lines[i]
                if hl.startswith("\\"):
                    if body:
                        body[-1] = body[-1][0] + body[-1][1:].rstrip("\n").rstrip("\r") if body[-1].endswith("\n") else body[-1]
                        if body[-1].endswith("\r"):
                            body[-1] = body[-1][:-1]
                    i += 1
                    continue
                tag = hl[:1]
                if tag == " " or hl in ("\n", "\r\n"):
                    if hl in ("\n", "\r\n"):
                        hl = " " + hl
                    o_left -= 1
                    n_left -= 1
                elif tag == "-":
                    o_left -= 1
                elif tag == "+":
                    n_left -= 1
                else:
                    break
                body.append(hl)
                i += 1
            cur.hunks.append((os_, ol, ns, nl, body))
            continue
        i += 1
    return files


def apply_hunks(old: str, hunks) -> str | None:
    src = old.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for os_, ol, ns, nl, body in hunks:
        want = [l[1:] for l in body if l[:1] in (" ", "-")]
        repl = [l[1:] for l in body if l[:1] in (" ", "+")]
        start = os_ - 1 if ol > 0 else os_
        found = None
        for delta in sorted(range(-200, 201), key=abs):
            s = start + delta
            if s < pos or s + len(want) > len(src):
                continue
            if src[s:s + len(want)] == want:
                found = s
                break
            # tolerate a missing final newline in the original
            if want and s + len(want) == len(src) and src[s:s + len(want) - 1] == want[:-1] \
                    and src[s + len(want) - 1] == want[-1].rstrip("\n"):
                found = s
                break
        if found is None:
            return None
        out.extend(src[pos:found])
        out.extend(repl)
        pos = found + len(want)
    out.extend(src[pos:])
    return "".join(out)


def changed_old_lines(hunks) -> tuple[set, list]:
    """Old line numbers that were removed or replaced, and insertion points
    (old line after which new lines were added, 0 = start of file)."""
    removed: set = set()
    inserts: list = []
    for os_, ol, ns, nl, body in hunks:
        line = os_ if ol > 0 else os_ + 1
        for l in body:
            tag = l[:1]
            if tag == " ":
                line += 1
            elif tag == "-":
                removed.add(line)
                line += 1
            elif tag == "+":
                inserts.append(line - 1)
    return removed, inserts


# ---------------------------------------------------------------- map


class Map:
    def __init__(self, path: str, tests: list[dict]):
        m = json.loads(gzip.decompress(open(path, "rb").read()))
        self.m = m
        self.sets_hex = m["sets"]
        self._sets: dict = {}
        ids = m["tests"]
        num_of = {t["id"]: t["number"] for t in tests}
        self.n_tests = len(tests)
        # map index -> test number; identity in the normal case
        self.identity = len(ids) == len(tests) and all(t["id"] == ids[t["number"]] for t in tests)
        self.idx_num = [num_of.get(tid) for tid in ids]
        mapped = {num_of[t] for t in ids if t in num_of}
        unknown = 0
        for t in tests:
            if t["number"] not in mapped:
                unknown |= 1 << t["number"]
        for tid in m.get("not_run", []):
            if tid in num_of:
                unknown |= 1 << num_of[tid]
        self.unknown = unknown
        self.lines = m["lines"]
        self._line_cache: dict = {}
        self._file_all: dict = {}
        self.import_lines = {f: set(v) for f, v in m["import_lines"].items()}
        self.opened = m["opened"]
        self.listed = m["listed"]
        self.types = m["types"]
        self.spawn = self.bits(m["spawn"])
        self.all = (1 << self.n_tests) - 1
        self.by_file: dict = {}
        for t in tests:
            self.by_file.setdefault(t["file"], 0)
            self.by_file[t["file"]] |= 1 << t["number"]

    def bits(self, sid: int) -> int:
        b = self._sets.get(sid)
        if b is None:
            raw = int(self.sets_hex[sid], 16)
            if not self.identity:
                out = 0
                i = 0
                while raw:
                    if raw & 1 and self.idx_num[i] is not None:
                        out |= 1 << self.idx_num[i]
                    raw >>= 1
                    i += 1
                raw = out
            b = raw
            self._sets[sid] = b
        return b

    def _file(self, f: str):
        c = self._line_cache.get(f)
        if c is None:
            pairs = self.lines.get(f, [])
            c = ([p[0] for p in pairs], [p[1] for p in pairs])
            self._line_cache[f] = c
        return c

    def lines_tests(self, f: str, lo: int, hi: int) -> int:
        lns, sids = self._file(f)
        out = 0
        i = bisect.bisect_left(lns, lo)
        while i < len(lns) and lns[i] <= hi:
            out |= self.bits(sids[i])
            i += 1
        return out

    def file_tests(self, f: str) -> int:
        b = self._file_all.get(f)
        if b is None:
            b = 0
            for sid in set(self._file(f)[1]):
                b |= self.bits(sid)
            self._file_all[f] = b
        return b

    def class_tests(self, key: str) -> int | None:
        sid = self.types.get(key)
        return None if sid is None else self.bits(sid)

    def module_class_tests(self, module: str) -> int:
        out = 0
        prefix = module + ":"
        for k, sid in self.types.items():
            if k.startswith(prefix):
                out |= self.bits(sid)
        return out

    def opened_tests(self, path: str) -> int:
        sid = self.opened.get(path)
        return 0 if sid is None else self.bits(sid)

    def listed_tests(self, d: str) -> int:
        sid = self.listed.get(d)
        return 0 if sid is None else self.bits(sid)

    def has_import_line(self, f: str, lo: int, hi: int) -> bool:
        s = self.import_lines.get(f)
        if not s:
            return False
        return any(l in s for l in range(lo, hi + 1))


# ---------------------------------------------------------------- import graph


def module_name(rel: str) -> str | None:
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_index(root: str) -> dict:
    """Static import graph of the pinned tree: module file -> files importing it."""
    files = {}
    for top in ("sqlglot", "tests"):
        for d, _, fs in os.walk(os.path.join(root, top)):
            if "__pycache__" in d or "/fixtures" in d:
                continue
            for f in fs:
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(d, f), root)
                    files[module_name(rel)] = rel
    importers: dict = {}
    refs: dict = {}
    strings: set = set()
    for mod, rel in files.items():
        try:
            tree = ast.parse(open(os.path.join(root, rel), encoding="utf-8").read())
        except Exception:
            continue
        pkg = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    for k in range(1, len(parts) + 1):
                        targets.add(".".join(parts[:k]))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    p = pkg.split(".")
                    p = p[: len(p) - (node.level - 1)] if node.level > 1 else p
                    base = ".".join(p + ([base] if base else []))
                parts = base.split(".")
                for k in range(1, len(parts) + 1):
                    targets.add(".".join(parts[:k]))
                for a in node.names:
                    targets.add(f"{base}.{a.name}")
        for t in targets:
            if t in files and t != mod:
                importers.setdefault(files[t], []).append(rel)
        _collect_refs(tree, rel, refs, strings)
    return {"importers": {k: sorted(v) for k, v in importers.items()},
            "refs": refs, "strings": sorted(strings)}


CONST_RE = re.compile(r"^_?[A-Z][A-Z0-9_]{2,}$")


def _collect_refs(tree, rel: str, refs: dict, strings: set):
    """Where constant-style names are read: [file, line, in_function, owner].
    owner is the last name before the attribute (Tokenizer for
    tokens.Tokenizer.KEYWORDS), empty for a bare name."""

    def visit(node, in_func: bool):
        for child in ast.iter_child_nodes(node):
            f = in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # defaults and decorators are evaluated where the def stands
                for d in getattr(child, "decorator_list", []):
                    visit_expr(d, in_func)
                for d in child.args.defaults + [x for x in child.args.kw_defaults if x is not None]:
                    visit_expr(d, in_func)
                for b in (child.body if isinstance(child.body, list) else [child.body]):
                    visit_expr(b, True) if not isinstance(b, ast.stmt) else visit_stmt(b, True)
                continue
            visit_one(child, f)
            visit(child, f)

    def visit_one(n, in_func):
        name = None
        owner = ""
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            name = n.attr
            v = n.value
            owner = v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else "?"
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and CONST_RE.match(n.value):
            strings.add(n.value)
        if name and CONST_RE.match(name):
            refs.setdefault(name, []).append([rel, n.lineno, 1 if in_func else 0, owner])

    def visit_expr(e, in_func):
        visit_one(e, in_func)
        visit(e, in_func)

    def visit_stmt(st, in_func):
        visit_one(st, in_func)
        visit(st, in_func)

    visit(tree, False)


# ---------------------------------------------------------------- AST diff

BODY_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")
SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _dump(node) -> str:
    return ast.dump(node)


def _header(node) -> str:
    parts = []
    for f in node._fields:
        if f in BODY_FIELDS:
            continue
        v = getattr(node, f, None)
        if isinstance(v, list):
            parts.append(f + "=[" + ",".join(_dump(x) if isinstance(x, ast.AST) else repr(x) for x in v) + "]")
        elif isinstance(v, ast.AST):
            parts.append(f + "=" + _dump(v))
        else:
            parts.append(f + "=" + repr(v))
    return "|".join(parts)


class _LambdaMask(ast.NodeTransformer):
    def visit_Lambda(self, node):
        return ast.Lambda(args=node.args, body=ast.Constant(value=None))


def _lambdas(node) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Lambda)]


class Site:
    """One changed spot. scope is a tuple of (kind, name, node) from the module down."""

    def __init__(self, scope, kind, nodes=(), prev=None, nxt=None, parent=None, lam=False):
        self.scope = scope
        self.kind = kind  # "replace" | "insert" | "header"
        self.nodes = list(nodes)
        self.prev = prev
        self.next = nxt
        self.parent = parent
        self.lam = lam


def diff_ast(old_tree, new_tree) -> list[Site]:
    sites: list[Site] = []

    def walk_block(ob, nb, scope, parent):
        od = [_dump(s) for s in ob]
        nd = [_dump(s) for s in nb]
        if od == nd:
            return
        sm = difflib.SequenceMatcher(None, od, nd, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    walk_stmt(ob[i1 + k], nb[j1 + k], scope, ob, i1 + k, parent)
                continue
            if i2 > i1:
                sites.append(Site(scope, "replace", ob[i1:i2], parent=parent))
            if j2 > j1:
                prev = ob[i1 - 1] if i1 > 0 else None
                nxt = ob[i2] if i2 < len(ob) else None
                sites.append(Site(scope, "insert", nb[j1:j2], prev=prev, nxt=nxt, parent=parent))

    def walk_stmt(o, n, scope, block, idx, parent):
        if type(o) is not type(n):
            sites.append(Site(scope, "replace", [o], parent=parent))
            return
        has_body = any(isinstance(getattr(o, f, None), list) and getattr(o, f) for f in BODY_FIELDS)
        if has_body:
            if _header(o) != _header(n):
                sites.append(Site(scope, "header", [o], parent=parent))
                return
            inner = scope
            if isinstance(o, SCOPED):
                kind = "class" if isinstance(o, ast.ClassDef) else "func"
                inner = scope + ((kind, o.name, o),)
            for f in BODY_FIELDS:
                ob = getattr(o, f, None)
                nb = getattr(n, f, None)
                if isinstance(ob, list) and isinstance(nb, list):
                    walk_block(ob, nb, inner, o)
            return
        # simple statement (or handler / match case without body change)
        try:
            mo = _dump(_LambdaMask().visit(_copy(o)))
            mn = _dump(_LambdaMask().visit(_copy(n)))
        except Exception:
            mo, mn = "a", "b"
        if mo == mn:
            lo, ln_ = _lambdas(o), _lambdas(n)
            if len(lo) == len(ln_):
                for a, b in zip(lo, ln_):
                    if _dump(a) != _dump(b):
                        sites.append(Site(scope + (("func", "<lambda>", a),), "replace", [a.body], parent=a, lam=True))
                return
        sites.append(Site(scope, "replace", [o], parent=parent))

    walk_block(old_tree.body, new_tree.body, (), old_tree)
    return sites


def _copy(node):
    import copy

    return copy.deepcopy(node)


def _span(node) -> tuple[int, int]:
    lo = getattr(node, "lineno", None)
    hi = getattr(node, "end_lineno", None) or lo
    decos = getattr(node, "decorator_list", None)
    if decos:
        lo = min([lo] + [d.lineno for d in decos])
    return lo, hi


# ---------------------------------------------------------------- analysis


class Analysis:
    def __init__(self, commit: str):
        self.commit = commit
        self.cand = 0       # tests that may fail
        self.broad = 0      # wider net, used to fill idle worker time
        self.full = False   # select everything, no verification
        self.reasons: list = []
        self.patched: dict = {}  # rel path -> new text or None (deleted)
        self.applied = True
        self.sqlglot_modules: list = []  # changed importable sqlglot modules
        self.rsafe = True  # every change is a function body the proxy covers


class Analyzer:
    def __init__(self, mp: Map, index: dict, rlines: dict | None = None):
        self.mp = mp
        self.importers = index["importers"]
        self.refs = index.get("refs", {})
        self.strings = set(index.get("strings", []))
        self._module_rule: dict = {}
        # lines the TPC executor tests run outside the proxied front end
        self.rlines = rlines

    # rules -------------------------------------------------------------
    def module_rule(self, rel: str) -> int:
        b = self._module_rule.get(rel)
        if b is not None:
            return b
        mp = self.mp
        b = mp.file_tests(rel) | mp.spawn
        mod = module_name(rel)
        if mod:
            b |= mp.module_class_tests(mod)
        if rel.startswith("tests/"):
            b |= mp.by_file.get(rel, 0)
        for imp in self.importers.get(rel, []):
            b |= mp.file_tests(imp)
            m2 = module_name(imp)
            if m2:
                b |= mp.module_class_tests(m2)
            if imp.startswith("tests/"):
                b |= mp.by_file.get(imp, 0)
        b |= mp.opened_tests(rel)
        self._module_rule[rel] = b
        return b

    def class_rule(self, rel: str, scope) -> int:
        mp = self.mp
        mod = module_name(rel)
        cls_scope = [s for s in scope if s[0] == "class"]
        qual = ".".join(s[1] for s in scope)
        node = scope[-1][2]
        lo, hi = _span(node)
        b = mp.lines_tests(rel, lo, hi) | mp.spawn
        known = mp.class_tests(f"{mod}:{qual}")
        if known is None:
            b |= self.module_rule(rel)
        else:
            b |= known
        if len(cls_scope) > 1:
            # a nested class also counts as part of its outer classes
            for k in range(len(scope) - 1, 0, -1):
                outer = ".".join(s[1] for s in scope[:k])
                ob = mp.class_tests(f"{mod}:{outer}")
                if ob is not None:
                    b |= ob
        return b

    def site_spans(self, site: Site) -> list:
        spans = []
        if site.kind == "insert":
            if site.prev is not None:
                spans.append(_span(site.prev))
            elif site.next is not None:
                spans.append(_span(site.next))
            p = site.parent
            if p is not None and hasattr(p, "lineno") and not isinstance(p, SCOPED + (ast.Lambda,)):
                # the header of an if/for/while/with/try reaching this block;
                # a def line runs at import, its body is what the tests reach
                spans.append((p.lineno, p.lineno))
            if site.prev is None and site.next is None and p is not None and hasattr(p, "lineno"):
                spans.append(_span(p))
        else:
            for n in site.nodes:
                spans.append(_span(n))
        return spans

    def r_hit(self, rel: str, site: Site) -> bool:
        """Whether the TPC tests may reach this change outside the front end."""
        if self.rlines is None:
            return True
        lines = self.rlines.get(rel)
        if not lines:
            return False
        return any(l in lines for lo, hi in self.site_spans(site) for l in range(lo, hi + 1))

    def import_dependents(self, site: Site) -> int:
        """Modules that read the changed names at import time (class bodies like
        KEYWORDS = {**tokens.Tokenizer.KEYWORDS} then .pop(...)): a change can
        alter or break their import, so take every test touching them."""
        mp = self.mp
        owners = {s[1] for s in site.scope if s[0] == "class"}
        names = set()
        for n in site.nodes:
            if isinstance(n, ast.Assign):
                names.update(t.id for t in n.targets if isinstance(t, ast.Name))
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
        out = 0
        files = set()
        for name in names:
            for f, line, in_func, owner in self.refs.get(name, []):
                if owners and owner not in owners and owner != "?":
                    continue  # another class's attribute of the same name
                if not owners and owner:
                    continue  # module-level name: importers are in the module rule
                if not in_func or mp.has_import_line(f, line, line):
                    files.add(f)
        for f in files:
            out |= mp.file_tests(f) | mp.by_file.get(f, 0)
            m = module_name(f)
            if m:
                out |= mp.module_class_tests(m)
        return out

    def attr_rule(self, rel: str, site: Site) -> int | None:
        """A class attribute that is only read inside functions at run time:
        tests that both instantiate the class (or a subclass) and reach a read.
        None when that cannot be shown, so the caller uses the class rule."""
        mp = self.mp
        if site.kind not in ("replace", "insert") or not site.nodes:
            return None
        names = set()
        for n in site.nodes:
            if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
                names.update(t.id for t in n.targets)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            else:
                return None
        mod = module_name(rel)
        qual = ".".join(s[1] for s in site.scope)
        inst = mp.class_tests(f"{mod}:{qual}")
        if inst is None:
            return None
        reach = 0
        for name in names:
            if not CONST_RE.match(name) or name in self.strings:
                return None
            for f, line, in_func, _owner in self.refs.get(name, []):
                if not in_func or mp.has_import_line(f, line, line):
                    return None
                reach |= mp.lines_tests(f, line, line)
        return inst & reach

    def func_rule(self, rel: str, site: Site) -> tuple[int, bool]:
        """Tests reaching the changed statements; flag if the code runs at import."""
        mp = self.mp
        spans = self.site_spans(site)
        b = 0
        at_import = False
        for lo, hi in spans:
            b |= mp.lines_tests(rel, lo, hi)
            if not site.lam and mp.has_import_line(rel, lo, hi):
                at_import = True
        if not site.lam:
            fn = [s for s in site.scope if s[0] == "func"]
            if fn:
                f = fn[-1][2]
                body = f.body
                if body and mp.has_import_line(rel, body[0].lineno, f.end_lineno):
                    at_import = True
        return b, at_import

    # entry ---------------------------------------------------------------
    def analyze(self, commit: str, text: str) -> Analysis:
        mp = self.mp
        a = Analysis(commit)
        try:
            diffs = parse_patch(text)
        except Exception as e:  # noqa: BLE001
            a.full = True
            a.applied = False
            a.reasons.append(f"unparseable patch: {e}")
            return a
        if not diffs:
            a.reasons.append("empty patch")
            return a
        for fd in diffs:
            try:
                self.analyze_file(a, fd)
            except Exception as e:  # noqa: BLE001
                a.full = True
                a.reasons.append(f"analysis error {fd.old or fd.new}: {type(e).__name__}: {e}")
        a.cand |= mp.unknown
        a.broad |= a.cand
        return a

    def analyze_file(self, a: Analysis, fd: FileDiff):
        mp = self.mp
        old_rel, new_rel = fd.old, fd.new
        rel = old_rel or new_rel
        if fd.binary or old_rel != new_rel or not (rel or "").startswith("sqlglot/") \
                or not rel.endswith(".py"):
            a.rsafe = False
        if fd.binary:
            a.full = True
            a.applied = False
            a.reasons.append(f"binary {rel}")
            return
        old_text = ""
        if old_rel is not None:
            p = os.path.join(SRC, old_rel)
            if not os.path.isfile(p):
                a.full = True
                a.applied = False
                a.reasons.append(f"missing file {old_rel}")
                return
            old_text = open(p, encoding="utf-8", newline="").read()
        new_text = None
        if new_rel is not None:
            new_text = apply_hunks(old_text, fd.hunks) if fd.hunks else old_text
            if new_text is None:
                a.full = True
                a.applied = False
                a.reasons.append(f"hunk failed {rel}")
                return
        if old_rel is not None and new_rel != old_rel:
            a.patched[old_rel] = None
        if new_rel is not None:
            a.patched[new_rel] = new_text

        for r in {old_rel, new_rel} - {None}:
            base = os.path.basename(r)
            if base in CONFIG_FILES or r.startswith("tests/") and base == "__init__.py":
                a.full = True
                a.reasons.append(f"config {r}")
                return

        # directory listings see added or removed files
        if old_rel != new_rel:
            for r in {old_rel, new_rel} - {None}:
                d = os.path.dirname(r)
                a.cand |= mp.listed_tests(d) | mp.listed_tests(d + "/__pycache__")

        if old_rel is not None and old_rel.endswith(".py") and (new_rel is None or new_rel == old_rel):
            if new_rel is None:
                a.cand |= self.module_rule(old_rel) | mp.opened_tests(old_rel)
                if old_rel.startswith("sqlglot/"):
                    a.full = True
                    a.reasons.append(f"deleted module {old_rel}")
                return
            self.analyze_py(a, old_rel, old_text, new_text)
            return
        if new_rel is not None and new_rel.endswith(".py") and old_rel is None:
            # a new module only matters once something imports it, and that
            # importer is itself a change; listings were handled above.
            if new_rel.startswith("sqlglot/"):
                a.sqlglot_modules.append(module_name(new_rel))
            a.broad |= mp.by_file.get(new_rel, 0)
            return
        if old_rel is not None and new_rel is not None and old_rel != new_rel:
            for r in (old_rel, new_rel):
                if r.endswith(".py"):
                    a.cand |= self.module_rule(r) if r == old_rel else 0
            a.cand |= mp.opened_tests(old_rel)
            if old_rel.startswith("sqlglot/") and old_rel.endswith(".py"):
                a.full = True
                a.reasons.append(f"renamed module {old_rel}")
            return
        # data file: fixtures and anything else a test reads
        b = mp.opened_tests(rel)
        a.cand |= b
        if rel.startswith("tests/fixtures/") and not b and old_rel is not None:
            a.reasons.append(f"unread file {rel}")

    def analyze_py(self, a: Analysis, rel: str, old_text: str, new_text: str):
        mp = self.mp
        a.cand |= mp.opened_tests(rel)  # tests that read the source as text
        is_pkg = rel.startswith("sqlglot/")
        try:
            new_tree = ast.parse(new_text)
        except SyntaxError:
            if is_pkg:
                a.full = True
                a.reasons.append(f"syntax error {rel}")
            else:
                a.cand |= self.module_rule(rel)
            return
        try:
            old_tree = ast.parse(old_text)
        except SyntaxError:
            a.cand |= self.module_rule(rel)
            return
        if is_pkg:
            a.sqlglot_modules.append(module_name(rel))
        if old_text != new_text and _encoding_cookie(old_text) != _encoding_cookie(new_text):
            a.cand |= self.module_rule(rel)
            a.rsafe = False
        sites = diff_ast(old_tree, new_tree)
        if not sites:
            a.reasons.append(f"no-op {rel}")
            a.broad |= mp.file_tests(rel)
            return
        for site in sites:
            scope = site.scope
            in_func = any(s[0] == "func" for s in scope)
            if in_func:
                b, at_import = self.func_rule(rel, site)
                a.cand |= b
                if at_import or self.r_hit(rel, site):
                    a.rsafe = False
            elif not (site.kind == "insert" and not scope and _new_unused_defs(site.nodes, old_text)):
                a.rsafe = False
            if in_func:
                if at_import:
                    a.cand |= self.module_rule(rel)
                    a.reasons.append(f"import-time code {rel}")
            elif scope:
                # class body: attributes, method headers, added or removed methods
                narrow = self.attr_rule(rel, site)
                if narrow is not None:
                    a.cand |= narrow | mp.spawn
                    a.broad |= self.class_rule(rel, scope)
                    a.reasons.append(f"attribute read at run time {rel}")
                    continue
                a.cand |= self.class_rule(rel, scope) | self.import_dependents(site)
                if site.kind != "insert":
                    for n in site.nodes:
                        lo, hi = _span(n)
                        a.cand |= mp.lines_tests(rel, lo, hi)
            elif site.kind == "insert" and _new_unused_defs(site.nodes, old_text):
                # a new top-level def or class under a name the module never used
                # changes nothing until other code calls it, and that code is a site
                a.broad |= mp.file_tests(rel)
                continue
            else:
                header_only = (
                    site.kind == "header"
                    and isinstance(site.nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                if header_only:
                    o = site.nodes[0]
                    lo, hi = _span(o)
                    a.cand |= mp.lines_tests(rel, lo, hi)
                    # name or decorators changed: anything importing it may break
                    a.cand |= self.module_rule(rel)
                else:
                    a.cand |= self.module_rule(rel) | self.import_dependents(site)
            a.broad |= mp.file_tests(rel)
        a.broad |= self.module_rule(rel)


def _new_unused_defs(nodes, old_text: str) -> bool:
    for n in nodes:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if n.name.startswith("__") or n.decorator_list:
            return False
        if re.search(rf"\b{re.escape(n.name)}\b", old_text):
            return False
    return True


def _encoding_cookie(text: str) -> str:
    head = "\n".join(text.splitlines()[:2])
    m = re.search(r"coding[:=]\s*([-\w.]+)", head)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- packing


def pack(nums: list[int], dur: list[int], k: int):
    workers = [[] for _ in range(k)]
    loads = [0] * k
    for t in sorted(nums, key=lambda t: -dur[t]):
        i = loads.index(min(loads))
        workers[i].append(t)
        loads[i] += dur[t]
    return workers, loads


def best_pack(nums: list[int], dur: list[int]):
    if not nums:
        return [], [], 0
    best = None
    for k in range(1, min(MAX_WORKERS, len(nums)) + 1):
        w, l = pack(nums, dur, k)
        cost = max(l) + WORKER_MS * k
        if best is None or cost < best[2]:
            best = (w, l, cost)
    return best


def pack_cost(nums, dur) -> int:
    return best_pack(nums, dur)[2]


def fill_slack(workers, loads, extra: list[int], dur: list[int]):
    """Add tests into idle time below the busiest worker. Costs nothing."""
    if not workers:
        return workers
    top = max(loads)
    loads = list(loads)
    for t in extra:
        d = dur[t]
        best = -1
        for i in range(len(workers)):
            if loads[i] + d <= top and (best < 0 or loads[i] < loads[best]):
                best = i
        if best >= 0:
            workers[best].append(t)
            loads[best] += d
    return workers


def bits_list(b: int) -> list[int]:
    out = []
    i = 0
    while b:
        low = b & -b
        i = low.bit_length() - 1
        out.append(i)
        b ^= low
    return out


# ---------------------------------------------------------------- verification


def build_farm(root: str, patched: dict):
    """A tree at `root` that looks like the pinned checkout with the patch applied.
    Directories on the path of a changed file are real, test sources are copied
    (so resolve() stays inside the farm), everything else is a symlink."""
    import shutil

    need: dict = {}
    for rel in patched:
        parts = rel.split("/")
        for k in range(len(parts)):
            need.setdefault("/".join(parts[:k]), set()).add(parts[k])

    def build(rel_dir: str):
        src_dir = os.path.join(SRC, rel_dir) if rel_dir else SRC
        dst_dir = os.path.join(root, rel_dir) if rel_dir else root
        os.makedirs(dst_dir, exist_ok=True)
        names = set(os.listdir(src_dir)) if os.path.isdir(src_dir) else set()
        names |= need.get(rel_dir, set())
        for name in names:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, name)
            if rel in patched:
                text = patched[rel]
                if text is not None:
                    with open(dst, "w", encoding="utf-8", newline="") as f:
                        f.write(text)
                continue
            if rel in need:
                build(rel)
                continue
            top = rel.split("/")[0]
            if top == "tests" and name.endswith(".py") and os.path.isfile(src):
                shutil.copy2(src, dst)
            elif top == "tests" and os.path.isdir(src) and name not in ("fixtures", "__pycache__"):
                build(rel)
            else:
                os.symlink(src, dst)

    build("")


class _SyncPool:
    """ProcessPoolExecutor stand-in: runs work in-process, through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        import pickle

        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        import concurrent.futures
        import pickle

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


class _TestTimeout(Exception):
    pass


class _Collector:
    def __init__(self, limits: dict):
        self.failed: set = set()
        self.seen: set = set()
        self.limits = limits

    def pytest_runtest_logreport(self, report):
        if report.failed or getattr(report, "outcome", "") == "failed":
            self.failed.add(report.nodeid)
        else:
            self.seen.add(report.nodeid)

    def pytest_collectreport(self, report):
        if report.failed:
            self.failed.add("collect:" + report.nodeid)

    def _alarm(self, *_):
        raise _TestTimeout("test timed out")

    def pytest_runtest_call(self, item):
        import signal

        limit = self.limits.get(item.nodeid, 30.0)
        signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, limit)

    def pytest_runtest_teardown(self, item):
        import signal

        signal.setitimer(signal.ITIMER_REAL, 0)


def child_verify(job: dict, out_path: str):
    """Runs in a forked child. Writes {"failed": [...], "passed": [...]} or an error."""
    import shutil

    result: dict = {}
    farm = job["farm"]
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        build_farm(farm, job["patched"])
        os.chdir(farm)
        sys.path.insert(0, farm)
        os.environ["PYTHONPATH"] = farm
        import concurrent.futures
        import concurrent.futures.process as cfp

        concurrent.futures.ProcessPoolExecutor = _SyncPool
        cfp.ProcessPoolExecutor = _SyncPool
        import importlib

        for mod in ["sqlglot"] + job["modules"]:
            try:
                importlib.import_module(mod)
            except BaseException as e:  # noqa: BLE001
                result = {"import_error": f"{mod}: {type(e).__name__}: {e}"}
                raise SystemExit
        failed = set()
        rc = 0
        if job["ids"]:
            import pytest

            col = _Collector(job["limits"])
            args = list(job["ids"]) + [
                "-q", "--tb=no", "-p", "no:cacheprovider", "-p", "no:randomly", "-o", "addopts=",
                "--rootdir", farm, "-o", "python_files=test_*.py",
            ]
            rc = pytest.main(args, plugins=[col])
            for nid in job["ids"]:
                if nid in col.failed or nid not in col.seen:
                    failed.add(nid)
        result = {"failed": sorted(failed), "rc": int(rc)}
        if job.get("proxy"):
            # after pytest, so the proxy cannot change what the tests saw
            result["proxy_ok"] = []
            sys.path.insert(1, APP)
            import proxy

            for tid in job["proxy"]:
                if proxy.check(tid, job["proxy_base"].get(tid)):
                    result["proxy_ok"].append(tid)
    except SystemExit:
        pass
    except BaseException as e:  # noqa: BLE001
        result = {"error": f"{type(e).__name__}: {e}"}
    try:
        with open(out_path, "w") as f:
            json.dump(result, f)
    finally:
        shutil.rmtree(farm, ignore_errors=True)
        os._exit(0)


class Verifier:
    def __init__(self, deadline_start: float, deadline_kill: float):
        self.deadline_start = deadline_start
        self.deadline_kill = deadline_kill
        self.running: dict = {}  # pid -> (job, out_path, started)
        self.results: dict = {}

    def preload(self, need_files: set):
        os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        import pytest  # noqa: F401
        import _pytest.unittest  # noqa: F401
        import _pytest.python  # noqa: F401

    def run(self, jobs: list[dict]):
        queue = list(jobs)
        while queue or self.running:
            now = time.monotonic() - T0
            while queue and len(self.running) < PARALLEL:
                job = queue[0]
                if now + job["est_wall"] > self.deadline_kill or now > self.deadline_start:
                    log(f"skip verify {job['commit']}: out of time")
                    queue.pop(0)
                    continue
                queue.pop(0)
                out_path = os.path.join(TMP, f"res_{job['commit']}.json")
                job["farm"] = os.path.join(TMP, f"farm_{job['commit']}")
                pid = os.fork()
                if pid == 0:
                    child_verify(job, out_path)
                self.running[pid] = (job, out_path, now)
            if not self.running:
                continue
            pid, status, usage = os.wait4(-1, os.WNOHANG)
            if pid == 0:
                now = time.monotonic() - T0
                for p, (job, out_path, started) in list(self.running.items()):
                    if now > self.deadline_kill or now - started > job["max_wall"]:
                        log(f"kill verify {job['commit']} after {now - started:.1f}s")
                        try:
                            os.kill(p, 9)
                        except OSError:
                            pass
                time.sleep(0.02)
                continue
            if pid in self.running:
                job, out_path, started = self.running.pop(pid)
                res = None
                try:
                    with open(out_path) as f:
                        res = json.load(f)
                    os.unlink(out_path)
                except Exception:
                    res = None
                if res is None or "error" in res:
                    import shutil

                    shutil.rmtree(job["farm"], ignore_errors=True)
                self.results[job["commit"]] = res
                took = time.monotonic() - T0 - started
                cpu = usage.ru_utime + usage.ru_stime
                log(f"verified {job['commit']}: {len(job['ids'])} tests "
                    f"({sum(job['secs'])/1:.1f}s) in {took:.1f}s cpu {cpu:.1f}s -> "
                    f"{'?' if res is None else (len(res.get('failed', [])) if 'failed' in res else res)}")


# ---------------------------------------------------------------- main


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--build-index":
        idx = build_index(SRC)
        with open(os.path.join(APP, "index.json"), "w") as f:
            json.dump(idx, f)
        return

    tests = json.loads(open(os.path.join(DATA, "tests.json")).read())["tests"]
    n = len(tests)
    dur = [0] * n
    for t in tests:
        dur[t["number"]] = round(t["seconds"] * 1000)
    ids = {t["number"]: t["id"] for t in tests}
    secs = {t["number"]: t["seconds"] for t in tests}

    commits_dir = os.path.join(DATA, "commits")
    commits = sorted(f[:-6] for f in os.listdir(commits_dir) if f.endswith(".patch"))
    n_commits = len(commits)
    wall_limit = 180.0 * max(1.0, n_commits / 30.0) if n_commits > 32 else 180.0
    deadline_start = float(os.environ.get("BLAST_DEADLINE", 0.55 * wall_limit))
    deadline_kill = 0.75 * wall_limit
    verify_on = os.environ.get("BLAST_VERIFY", "1") == "1"

    try:
        mp = Map(os.path.join(APP, "map.json.gz"), tests)
        index = json.load(open(os.path.join(APP, "index.json")))
        rlines = None
        proxy_base: dict = {}
        try:
            raw = json.loads(gzip.decompress(open(os.path.join(APP, "rcov.json.gz"), "rb").read()))
            proxy_base = json.load(open(os.path.join(APP, "proxy_base.json")))
            rlines = {f: set(v) for f, v in raw.items()}
        except Exception as e:  # noqa: BLE001
            log(f"proxy unavailable ({e})")
        an = Analyzer(mp, index, rlines)
    except Exception as e:  # noqa: BLE001
        log(f"map unavailable ({e}); selecting everything")
        everything = list(range(n))
        w, _, _ = best_pack(everything, dur)
        for c in commits:
            print(json.dumps({"commit": c, "workers": w}))
        return
    log(f"map loaded, {n_commits} commits")

    analyses: dict = {}
    for c in commits:
        try:
            text = open(os.path.join(commits_dir, c + ".patch"), encoding="utf-8", errors="surrogateescape").read()
            a = an.analyze(c, text)
        except Exception as e:  # noqa: BLE001
            a = Analysis(c)
            a.full = True
            a.applied = False
            a.reasons.append(f"error {e}")
        if a.full:
            a.cand = mp.all
        analyses[c] = a

    # plan verification per commit -------------------------------------
    jobs = []
    plan: dict = {}
    # CPU model, from timing verification runs against recorded seconds.
    # Tests in pool-using files burn more CPU than their recorded wall time.
    per_run_cpu = 0.35
    file_cpu = {"tests/test_optimizer.py": 1.4, "tests/test_executor.py": 1.8}
    test_factor = {"tests/test_optimizer.py": 2.5, "tests/test_executor.py": 2.0}
    p_pass = 0.45  # prior that a verified candidate set holds no failure
    # The model above was timed under local amd64 emulation. The board's
    # native CPU reads about half of that, so scale the charge down.
    cpu_scale = float(os.environ.get("BLAST_CPU_SCALE", "0.5"))
    num_of_id = {v: k for k, v in ids.items()}
    base_failed = set(mp.m.get("base_failed", []))
    proxy_nums = {num_of_id[tid]: tid for tid in proxy_base
                  if tid in num_of_id and proxy_base[tid] and tid not in base_failed}
    for c in commits:
        a = analyses[c]
        cand = bits_list(a.cand)
        base_cost = pack_cost(cand, dur)
        plan[c] = {"verify": [], "keep": cand, "proxy": []}
        px = []
        if a.rsafe and not a.full and a.applied and verify_on:
            px = [t for t in cand if t in proxy_nums and dur[t] >= PROXY_MIN_MS]
        if px:
            cand = [t for t in cand if t not in px]
            base_cost = pack_cost(cand, dur)
            plan[c]["keep"] = cand
            plan[c]["proxy"] = px
            jobs.append({
                "commit": c, "ids": [], "limits": {}, "secs": [],
                "patched": a.patched,
                "modules": [m for m in a.sqlglot_modules if m],
                "proxy": [proxy_nums[t] for t in px],
                "proxy_base": {proxy_nums[t]: proxy_base[proxy_nums[t]] for t in px},
                "gain": sum(dur[t] for t in px),
                "est_wall": 1.5 + 0.25 * sum(secs[t] for t in px),
                "max_wall": 30.0 + 1.0 * sum(secs[t] for t in px),
            })
            log(f"{c}: proxy {len(px)} ({sum(secs[t] for t in px):.1f}s)")
        if not cand or a.full or not a.applied or not verify_on:
            log(f"{c}: {len(cand)} cand, cost {base_cost / 1000:.1f}s, no verify {a.reasons[:3]}")
            continue
        # pick a duration threshold: verify everything at or below it
        order = sorted(cand, key=lambda t: dur[t])
        # demand a margin: the CPU model is rough, a tie is a likely loss
        best = (base_cost - 250, -1)
        files: set = set()
        cpu = per_run_cpu
        for k, t in enumerate(order):
            f = tests[t]["file"]
            cpu += secs[t] * test_factor.get(f, 1.2) + 0.003
            if f not in files:
                files.add(f)
                cpu += file_cpu.get(f, 0.1)
            if k + 1 < len(order) and dur[order[k + 1]] == dur[t]:
                continue
            rest = order[k + 1:]
            biggest = order[k]
            if_pass = pack_cost(rest, dur)
            if_fail = pack_cost(rest + [biggest], dur)
            est = cpu * cpu_scale * 1000 + p_pass * if_pass + (1 - p_pass) * if_fail
            if est < best[0]:
                best = (est, k)
        if os.environ.get("BLAST_FORCE") == "1":
            best = (0, len(order) - 1)
        if best[1] < 0:
            log(f"{c}: {len(cand)} cand, cost {base_cost / 1000:.1f}s, verify not worth it")
            continue
        v = order[: best[1] + 1]
        keep = order[best[1] + 1:]
        plan[c]["verify"] = v
        plan[c]["keep"] = keep
        vsec = sum(secs[t] for t in v)
        job = {
            "commit": c,
            "ids": [ids[t] for t in v],
            "limits": {ids[t]: 15.0 + 6.0 * secs[t] for t in v},
            "patched": a.patched,
            "modules": [m for m in a.sqlglot_modules if m],
            "gain": base_cost - best[0],
            "secs": [secs[t] for t in v],
            "est_wall": 1.5 + vsec * 1.6,
            "max_wall": 20.0 + vsec * 4.0,
        }
        if plan[c]["proxy"]:
            # one child does both: the tests, then the proxy
            pj = jobs.pop()
            job["proxy"] = pj["proxy"]
            job["proxy_base"] = pj["proxy_base"]
            job["gain"] += pj["gain"]
            job["est_wall"] += pj["est_wall"]
            job["max_wall"] += pj["max_wall"]
        jobs.append(job)
        log(f"{c}: {len(cand)} cand, cost {base_cost / 1000:.1f}s, verify {len(v)} ({vsec:.1f}s), "
            f"keep {len(keep)}, est {best[0] / 1000:.1f}s {a.reasons[:2]}")

    results: dict = {}
    if jobs:
        jobs.sort(key=lambda j: -j["gain"] / (j["est_wall"] + 1.0))
        ver = Verifier(deadline_start, deadline_kill)
        ver.preload(set())
        ver.run(jobs)
        results = ver.results

    num_of = {v: k for k, v in ids.items()}
    total = 0
    for c in commits:
        a = analyses[c]
        p = plan[c]
        out = set(p["keep"])
        res = results.get(c)
        if p["proxy"]:
            ok = set(res.get("proxy_ok", [])) if res and "failed" in res else set()
            for t in p["proxy"]:
                if ids[t] not in ok:
                    out.add(t)
            log(f"{c}: proxy cleared {len(ok)} of {len(p['proxy'])}")
            if res and "import_error" in res:
                log(f"{c}: patched sqlglot does not import: {res['import_error']}")
                out = set(range(n))
        if p["verify"]:
            if res is None or "failed" not in res:
                if res and "import_error" in res:
                    log(f"{c}: patched sqlglot does not import: {res['import_error']}")
                    out = set(range(n))
                else:
                    out |= set(p["verify"])
            else:
                for nid in res["failed"]:
                    if nid in num_of:
                        out.add(num_of[nid])
        sel = sorted(out)
        workers, loads, cost = best_pack(sel, dur)
        total += cost
        if workers:
            extra = [t for t in sorted(bits_list(a.cand | a.broad), key=lambda t: dur[t]) if t not in out]
            rest = [t for t in sorted(range(n), key=lambda t: dur[t]) if t not in out and not (a.cand | a.broad) >> t & 1]
            workers = fill_slack(workers, loads, extra + rest, dur)
        workers = [w for w in workers if w]
        print(json.dumps({"commit": c, "workers": workers}), flush=True)
    log(f"done, simulated estimate {total / 1000:.1f}s")


if __name__ == "__main__":
    main()
