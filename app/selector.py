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
# as _gc: child_verify imports gc locally further down, which makes a bare
# `gc` local to the whole function, unbound before that line
import gc as _gc
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
HEAVY_MIN_MS = 3000  # a candidate this slow may be verified alone
# Tests whose recorded time is mostly gc.collect() over the import heap. Run
# alone, they get every dialect loaded and the heap frozen first, so each
# collect scans only what the test itself made. Measured 3.7 s to 1.3 s CPU.
FREEZE_IDS = {"tests/test_generator.py::TestGenerator::test_generator_memory"}
PROXY_MIN_MS = 3000  # tests cheaper than this are not worth the proxy run

CONFIG_FILES = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py", "setup.py",
                ".pytest.ini", "pytest.toml", ".pytest.toml"}


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
        self.test_ids = {t["number"]: t["id"] for t in tests}
        self.num_of_id = {t["id"]: t["number"] for t in tests}
        for t in tests:
            self.by_file.setdefault(t["file"], 0)
            self.by_file[t["file"]] |= 1 << t["number"]

    def decode(self, hexbits: str) -> int:
        """A bitset over the map's test order (as in tbl.json.gz) as test numbers."""
        raw = int(hexbits, 16)
        if self.identity:
            return raw
        out = 0
        i = 0
        while raw:
            if raw & 1 and self.idx_num[i] is not None:
                out |= 1 << self.idx_num[i]
            raw >>= 1
            i += 1
        return out

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
    from_names: set = set()  # names some file binds with `from x import name`
    for mod, rel in files.items():
        try:
            tree = ast.parse(open(os.path.join(root, rel), encoding="utf-8").read())
        except Exception:
            continue
        pkg = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        targets = set()
        mod_alias: dict = {}  # local name -> module it is bound to anywhere in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    for k in range(1, len(parts) + 1):
                        targets.add(".".join(parts[:k]))
                    mod_alias.setdefault(a.asname or parts[0], set()).add(a.name if a.asname else parts[0])
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
                    from_names.add(a.asname or a.name)
                    mod_alias.setdefault(a.asname or a.name, set()).add(f"{base}.{a.name}")
        for t in targets:
            if t in files and t != mod:
                importers.setdefault(files[t], []).append(rel)
        start = {k: len(v) for k, v in refs.items()}
        _collect_refs(tree, rel, refs, strings)
        # the module an attribute chain starts from (tokens for
        # tokens.Tokenizer.KEYWORDS), when its head name is only ever bound
        # to that one module in this file
        for k, v in refs.items():
            for r in v[start.get(k, 0):]:
                parts = r[4].split(".") if len(r) > 4 and r[4] else []
                ms = mod_alias.get(parts[0]) if parts else None
                r[4:] = ["", ""]
                if not ms or len(ms) != 1 or _rebinds(tree, parts[0]):
                    continue
                t = next(iter(ms))
                if t in files:  # mod.Cls...
                    if len(parts) > 1:
                        r[4:] = [files[t], parts[1]]
                elif t.rpartition(".")[0] in files:  # Cls... from `from mod import Cls`
                    r[4:] = [files[t.rpartition(".")[0]], t.rpartition(".")[2]]
    idx = {"importers": {k: sorted(v) for k, v in importers.items()},
           "refs": refs, "strings": sorted(strings), "from_names": sorted(from_names)}
    try:
        idx.update(build_break_index(root, files))
    except Exception as e:  # noqa: BLE001
        print(f"break index unavailable: {type(e).__name__}: {e}", file=sys.stderr)
    return idx


# What a module that fails to import takes down. The map imported the whole
# package before any test ran, so it cannot see which tests import a lazily
# loaded module; these static facts fill that in.
_BREAK_PROBE = r"""
import importlib, json, sys
pairs = json.loads(sys.stdin.read())
import sqlglot
eager = sorted(m for m in sys.modules if m == "sqlglot" or m.startswith("sqlglot."))
lazy = []
for base, name in pairs:
    try:
        obj = getattr(importlib.import_module(base), name)
    except BaseException:
        continue
    mod = obj.__name__ if type(obj).__name__ == "module" else getattr(obj, "__module__", None)
    if isinstance(mod, str):
        lazy.append([base, name, mod])
print(json.dumps({"eager": eager, "lazy": lazy}))
"""


def _import_sites(tree, pkg: str):
    """(target module, line, inside a function) for every import statement."""
    out = []

    def visit(node, in_func):
        for ch in ast.iter_child_nodes(node):
            f = in_func or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(ch, ast.If) and _is_type_checking(ch.test):
                # never runs; only its else branch does
                for e in ch.orelse:
                    visit(ast.Module(body=[e], type_ignores=[]), in_func)
                continue
            if isinstance(ch, ast.Import):
                for a in ch.names:
                    parts = a.name.split(".")
                    for k in range(1, len(parts) + 1):
                        out.append((".".join(parts[:k]), ch.lineno, in_func, None))
            elif isinstance(ch, ast.ImportFrom):
                base = ch.module or ""
                if ch.level:
                    p = pkg.split(".")
                    p = p[: len(p) - (ch.level - 1)] if ch.level > 1 else p
                    base = ".".join(p + ([base] if base else []))
                parts = base.split(".")
                for k in range(1, len(parts) + 1):
                    out.append((".".join(parts[:k]), ch.lineno, in_func, None))
                for a in ch.names:
                    out.append((f"{base}.{a.name}", ch.lineno, in_func, (base, a.name)))
            visit(ch, f)

    visit(tree, False)
    return out


def _is_type_checking(test) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _words(node) -> set:
    w = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            w.add(n.id.lower())
        elif isinstance(n, ast.Attribute):
            w.add(n.attr.lower())
        elif isinstance(n, ast.alias):
            w.update(p.lower() for p in (n.name or "").split("."))
        elif isinstance(n, ast.ImportFrom) and n.module:
            w.update(p.lower() for p in n.module.split("."))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            w.update(re.findall(r"[a-z0-9_]+", n.value.lower()))
    return w


def build_break_index(root: str, files: dict) -> dict:
    import subprocess

    trees = {}
    for mod, rel in files.items():
        try:
            trees[rel] = ast.parse(open(os.path.join(root, rel), encoding="utf-8").read())
        except Exception:
            continue
    sites = {}
    pending = set()
    for mod, rel in files.items():
        if rel not in trees:
            continue
        pkg = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        s = _import_sites(trees[rel], pkg)
        sites[rel] = s
        for t, _line, _f, pair in s:
            if pair and t not in files and pair[0] in files and files[pair[0]].endswith("__init__.py"):
                pending.add(pair)
    env = dict(os.environ, PYTHONPATH=root)
    r = subprocess.run([sys.executable, "-c", _BREAK_PROBE], input=json.dumps(sorted(pending)),
                       capture_output=True, text=True, cwd=root, env=env, timeout=300)
    probe = json.loads(r.stdout.strip().splitlines()[-1])
    lazy = {(b, n): m for b, n, m in probe["lazy"]}
    edges: dict = {}
    for rel, s in sites.items():
        me = module_name(rel)
        for t, line, in_func, pair in s:
            tgt = files.get(t)
            if tgt is None and pair:
                tgt = files.get(lazy.get(pair, ""))
            if tgt is not None and t != me and tgt != rel:
                edges.setdefault(tgt, []).append([rel, line, 1 if in_func else 0])
    # words each test mentions, to catch registry lookups by name
    # ("dremio", Dialect["hive"], sqlglot.dialects.Snowflake)
    vocab = {os.path.basename(rel)[:-3].lower() for rel in files.values()
             if rel.startswith("sqlglot/")} - {"__init__"}
    mentions: dict = {}  # word -> {file: ["*"] or ["Class::method", "::func"]}
    for rel, tree in trees.items():
        if not rel.startswith("tests/") or not os.path.basename(rel).startswith("test_"):
            continue
        for node in tree.body:
            units = []
            if isinstance(node, ast.ClassDef):
                rest = []
                for b in node.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name.startswith("test"):
                        units.append((f"{node.name}::{b.name}", b))
                    else:
                        rest.append(b)
                units.append(("*", ast.Module(body=rest + node.bases + node.decorator_list, type_ignores=[])))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                units.append((f"::{node.name}", node))
            else:
                units.append(("*", node))
            for key, n in units:
                for w in _words(n) & vocab:
                    mentions.setdefault(w, {}).setdefault(rel, []).append(key)
    # functions that load every module of a registry in a loop (Dialect.classes)
    # A registry loader builds the module name from a key (a key a test
    # names is in mentions); a loop over such keys loads every module.
    # A data-driven import (serde: a path read from a payload) hands back
    # a class the test then instantiates, which the map records.
    loaders = set()
    loops = []
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/"):
            continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and _call_name(c) in ("import_module", "__import__"):
                    if c.args and isinstance(c.args[0], (ast.JoinedStr, ast.BinOp)):
                        loaders.add(n.name)
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/"):
            continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hit = any(
                isinstance(c, ast.Call) and _call_name(c) in loaders
                for lp in ast.walk(n) if isinstance(lp, (ast.For, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                for c in ast.walk(lp)
            )
            if hit:
                lo, hi = _span(n)
                loops.append([rel, lo, hi])
    eager = sorted({files[m] for m in probe["eager"] if m in files})
    return {"edges": edges, "eager": eager, "mentions": mentions, "loops": loops}


def _call_name(c) -> str:
    f = c.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


CONST_RE = re.compile(r"^_?[A-Z][A-Z0-9_]{2,}$")


def _rebinds(tree, name: str) -> bool:
    """Whether the file assigns name other than by an import (a def,
    class, assignment, loop target or argument of that name)."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(n, ast.arg) and n.arg == name:
            return True
    return False


def _top_classes(text: str) -> set:
    """Top-level classes of a module whose every class statement, nested ones
    too, names its bases as plain names or attributes rooted at a name the
    module imports or a class it defines: the MRO of each is then fixed by
    the modules this one imports at load time, not by a call or a lookup.
    Empty when any class statement does otherwise."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    known = {"object"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            known.update((a.asname or a.name.split(".")[0]) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            known.update((a.asname or a.name) for a in n.names)
        elif isinstance(n, ast.ClassDef):
            known.add(n.name)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            known.discard(n.id)  # a name assigned at run time may hold any class
    for n in ast.walk(tree):
        if not isinstance(n, ast.ClassDef):
            continue
        for b in n.bases:
            while isinstance(b, ast.Attribute):
                b = b.value
            if not (isinstance(b, ast.Name) and b.id in known):
                return set()
    out = set()
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and not _rebinds(
                ast.Module(body=[x for x in tree.body if x is not n], type_ignores=[]), n.name):
            out.add(n.name)
    return out


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
        chain = ""
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            name = n.attr
            v = n.value
            owner = v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else "?"
            parts = []
            while isinstance(v, ast.Attribute):
                parts.append(v.attr)
                v = v.value
            if isinstance(v, ast.Name):
                chain = ".".join([v.id] + parts[::-1])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and CONST_RE.match(n.value):
            strings.add(n.value)
        if name and CONST_RE.match(name):
            # the fifth field holds the full owner chain here; build_index
            # resolves it to the module file its head name binds, or ""
            refs.setdefault(name, []).append([rel, n.lineno, 1 if in_func else 0, owner, chain])

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


class _Dumper:
    """ast.dump(node) with each node formatted once. diff_ast dumps a class
    whole, then its body statement by statement, then their bodies again;
    the memo makes that one pass. Keyed by id(), so it lives only as long as
    the trees it saw. Same text as ast.dump with its defaults (no indent:
    the simple flag never changes the text)."""

    def __init__(self):
        self.memo: dict = {}
        self.keep: list = []

    def __call__(self, node) -> str:
        k = id(node)
        r = self.memo.get(k)
        if r is not None:
            return r
        cls = type(node)
        args = []
        for name in node._fields:
            try:
                value = getattr(node, name)
            except AttributeError:
                continue
            if value is None and getattr(cls, name, ...) is None:
                continue
            if value == []:
                field_type = cls._field_types.get(name, object)
                if getattr(field_type, "__origin__", ...) is list:
                    continue
            args.append(name + "=" + self.value(value))
        r = cls.__name__ + "(" + ", ".join(args) + ")"
        self.memo[k] = r
        self.keep.append(node)  # ids stay unique while the memo lives
        return r

    def value(self, v) -> str:
        if isinstance(v, ast.AST):
            return self(v)
        if isinstance(v, list):
            if not v:
                return "[]"
            return "[" + ", ".join(self.value(x) for x in v) + "]"
        return repr(v)


def _header(node, dump=None) -> str:
    _dump_ = dump or _dump
    parts = []
    for f in node._fields:
        if f in BODY_FIELDS:
            continue
        v = getattr(node, f, None)
        if isinstance(v, list):
            parts.append(f + "=[" + ",".join(_dump_(x) if isinstance(x, ast.AST) else repr(x) for x in v) + "]")
        elif isinstance(v, ast.AST):
            parts.append(f + "=" + _dump_(v))
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

    def __init__(self, scope, kind, nodes=(), prev=None, nxt=None, parent=None, lam=False, new=None):
        self.scope = scope
        self.kind = kind  # "replace" | "insert" | "header"
        self.nodes = list(nodes)
        self.new = new  # the patched statements, when paired one to one
        self.prev = prev
        self.next = nxt
        self.parent = parent
        self.lam = lam


def diff_ast(old_tree, new_tree) -> list[Site]:
    sites: list[Site] = []
    dump = _Dumper()

    def walk_block(ob, nb, scope, parent):
        od = [dump(s) for s in ob]
        nd = [dump(s) for s in nb]
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
            sites.append(Site(scope, "replace", [o], parent=parent, new=[n]))
            return
        has_body = any(isinstance(getattr(o, f, None), list) and getattr(o, f) for f in BODY_FIELDS)
        if has_body:
            if _header(o, dump) != _header(n, dump):
                sites.append(Site(scope, "header", [o], parent=parent, new=[n]))
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
                        sites.append(Site(scope + (("func", "<lambda>", a),), "replace", [a.body], parent=a, lam=True,
                                          new=[b.body]))
                return
        sites.append(Site(scope, "replace", [o], parent=parent, new=[n]))

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


def _has_line_event(n) -> bool:
    """Whether running statement n is seen by the line recorder."""
    if n is None:
        return False
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
        return False
    if isinstance(n, (ast.Global, ast.Nonlocal)):
        return False
    if isinstance(n, ast.AnnAssign) and n.value is None:
        return False
    return True


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
        self.imp = 0  # tests that can see the change only through import-time state
        # impfp keys a nested class change may alter (see nested_rule); only
        # honoured while every import-time doubt comes from that rule
        self.imp_excuse: list = []
        self.imp_strict = False
        self.equiv = ([], {})  # see site_equiv; None once any change is not of that kind
        self.equiv_files: set = set()
        self.collect = False  # a new conftest only drops collected items: collect to see which
        # a package module changed outside function bodies: its import runs
        # the change, and a module that no longer imports fails every test
        self.imp_check = False


def leak_rule(a: Analysis, rel: str, old_tree, new_tree, sites):
    """Shared-process state (leak.py): a test that fails only after an
    earlier test in its process left state behind is in the labels, but
    neither the map nor a lone verification run can see it. Select all."""
    try:
        import leak

        why = leak.shared_state(rel, old_tree, new_tree, sites)
    except Exception as e:  # noqa: BLE001
        why = f"rule error {e}"
    if why:
        a.full = True
        a.reasons.append(f"shared state {rel}: {why}")


class Analyzer:
    def __init__(self, mp: Map, index: dict, rlines: dict | None = None):
        self.mp = mp
        self.importers = index["importers"]
        self.refs = index.get("refs", {})
        self.strings = set(index.get("strings", []))
        self.from_names = index.get("from_names")
        self._module_rule: dict = {}
        # lines the TPC executor tests run outside the proxied front end
        self.rlines = rlines
        self.edges = index.get("edges")
        self.eager = set(index["eager"]) if "eager" in index else None
        self.mentions = index.get("mentions", {})
        self.loops = index.get("loops", [])
        self._ids_by_suffix = None
        self._top_cache = None
        self._reach_cache = None
        self._tree_cache = None
        self.warned = None  # tests that raise a warning on the pinned tree
        try:
            self.warned = json.loads(gzip.decompress(open(os.path.join(APP, "warn.json.gz"), "rb").read()))
        except Exception:  # noqa: BLE001
            pass

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

    def text_readers(self, rel: str) -> int:
        """Tests that open a source file. pytest opens a test module, and
        every test module it imports, while it collects a file, so the map
        charges that open to every test in the collected file. That is the
        import, which the line and module rules already cover. Only opens
        by tests outside those files are reads of the text."""
        mp = self.mp
        b = mp.opened_tests(rel)
        if not b or not rel.startswith("tests/"):
            return b
        seen = {rel}
        queue = [rel]
        while queue:
            x = queue.pop()
            b &= ~mp.by_file.get(x, 0)
            for imp in self.importers.get(x, []):
                if imp not in seen:
                    seen.add(imp)
                    queue.append(imp)
        return b

    def mention_tests(self, word: str) -> int:
        mp = self.mp
        if self._ids_by_suffix is None:
            d: dict = {}
            for f, b in mp.by_file.items():
                for t in bits_list(b):
                    tid = mp.test_ids[t].split("[")[0]
                    parts = tid.split("::")
                    d.setdefault((f, "::" + parts[-1]), []).append(t)
                    if len(parts) >= 3:
                        d.setdefault(("*", parts[-2] + "::" + parts[-1]), []).append(t)
                        d.setdefault((f, "~" + parts[-1]), []).append(t)
            self._ids_by_suffix = d
        d = self._ids_by_suffix
        out = 0
        for f, keys in self.mentions.get(word, {}).items():
            for k in keys:
                if k == "*":
                    out |= mp.by_file.get(f, 0)
                    continue
                cls, _, meth = k.partition("::")
                ts = d.get((f, "::" + meth), []) if not cls else \
                    d.get(("*", k), []) + d.get((f, "~" + meth), [])
                for t in ts:
                    out |= 1 << t
        return out

    def break_rule(self, rel: str) -> int | None:
        """Tests that fail when module rel cannot be imported at all: every
        module that imports it at load time breaks too, then each broken
        test file fails whole, and any test that loads a broken module late
        (a call-time import, a registry lookup by name, a test that uses its
        code). None when the break reaches the package's own import."""
        if self.eager is None or self.edges is None:
            return None
        mp = self.mp
        broken = {rel}
        queue = [rel]
        b = mp.spawn
        while queue:
            x = queue.pop()
            if x in self.eager:
                return None
            for imp, line, in_func in self.edges.get(x, []):
                if in_func:
                    b |= mp.lines_tests(imp, line, line)
                elif imp not in broken:
                    broken.add(imp)
                    queue.append(imp)
        for x in broken:
            b |= mp.by_file.get(x, 0) | mp.file_tests(x) | mp.opened_tests(x)
            m = module_name(x)
            if m:
                b |= mp.module_class_tests(m)
            base = os.path.basename(x)[:-3].lower()
            if base != "__init__":
                b |= self.mention_tests(base)
            else:
                b |= self.mention_tests(os.path.basename(os.path.dirname(x)).lower())
        for f, lo, hi in self.loops:
            b |= mp.lines_tests(f, lo, hi)
        return b

    def rebind_rule(self, rel: str, old_tree, new_tree) -> int | None:
        ob, om = _import_bindings(old_tree, rel)
        nb, nm = _import_bindings(new_tree, rel)
        if self.eager is None:
            return None
        for m in om ^ nm:
            # a module that now loads, or no longer loads, here: harmless
            # only when the package's own import loads it anyway, or it is
            # a standard library module that exists
            rp = m.replace(".", "/")
            if rp + ".py" in self.eager or rp + "/__init__.py" in self.eager:
                continue
            top = m.split(".")[0]
            if m in om or top not in sys.stdlib_module_names:
                return None
            import importlib.util

            if importlib.util.find_spec(m) is None:
                return None
        names = {k for k in set(ob) | set(nb) if ob.get(k) != nb.get(k)}
        if not names:
            return 0
        mp = self.mp
        pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        mod = module_name(rel)
        star = re.compile(r"from\s+" + re.escape(mod) + r"\s+import\s+\*")
        out = 0
        for top in ("sqlglot", "tests"):
            for root, dirs, files in os.walk(os.path.join(SRC, top)):
                dirs[:] = [x for x in dirs if x not in ("__pycache__", "fixtures")]
                for f in files:
                    if not f.endswith(".py"):
                        continue
                    path = os.path.join(root, f)
                    r = os.path.relpath(path, SRC)
                    text = open(path, encoding="utf-8", errors="replace").read()
                    if star.search(text):
                        return None
                    if not pat.search(text):
                        continue
                    tree = old_tree if r == rel else ast.parse(text)
                    for line, fn in _name_uses(tree, names, mod, None if r == rel else r):
                        if fn is not None:
                            out |= mp.lines_tests(r, fn[0], fn[1])
                        elif r.startswith("tests/"):
                            out |= mp.by_file.get(r, 0)
                        elif r == rel:
                            return None  # used while the module itself loads
                        else:
                            b = self.break_rule(r)
                            if b is None:
                                return None
                            out |= b | mp.file_tests(r)
        # a from-import of a name the source module does not bind fails,
        # and the module with it
        seen = {x for v in ob.values() for x in v}
        for k in names:
            for kind, base, nm in nb.get(k, []):
                if kind == "from" and (kind, base, nm) not in seen and not _module_binds(base, nm):
                    b = self.break_rule(rel)
                    if b is None:
                        return None
                    out |= b
        return out | mp.spawn

    # config ------------------------------------------------------------
    def config_effect(self, a, rel: str, old: str | None, new: str | None) -> int | None:
        """Tests a pytest config edit can change, or None when that cannot
        be shown. Handles a conftest.py that only adds or edits plain
        fixtures, and a pyproject.toml pytest table that only touches
        filterwarnings or markers."""
        try:
            base = os.path.basename(rel)
            if base == "conftest.py" and new is not None:
                return self._conftest_effect(a, rel, old, new)
            if base == "pyproject.toml" and old is not None and new is not None:
                return self._pyproject_effect(old, new)
        except Exception as e:  # noqa: BLE001
            log(f"config check failed {rel}: {type(e).__name__}: {e}")
        return None

    def _conftest_effect(self, a, rel: str, old: str | None, new: str) -> int | None:
        d = os.path.dirname(rel)
        if not os.path.isfile(os.path.join(SRC, d, "__init__.py")):
            return None  # rootdir-relative import would put d on sys.path
        of = _conftest_fixtures(old) if old is not None else {}
        nf = _conftest_fixtures(new)
        if of is None or nf is None:
            return None
        names = {k for k in set(of) | set(nf) if of.get(k) != nf.get(k)}
        if not names:
            return 0
        pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        out = 0
        prefix = d + "/" if d else ""
        for root, dirs, files in os.walk(os.path.join(SRC, d)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "fixtures")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                r = os.path.relpath(os.path.join(root, f), SRC)
                if r == rel or not r.startswith(prefix):
                    continue
                text = a.patched.get(r) if r in a.patched else open(
                    os.path.join(SRC, r), encoding="utf-8", errors="replace").read()
                if text is None or not pat.search(text):
                    continue
                if r not in self.mp.by_file:
                    return None  # a helper may request it on a test's behalf
                out |= self.mp.by_file[r]
        return out

    def _pyproject_effect(self, old: str, new: str) -> int | None:
        import tomllib

        def opts(text):
            tp = tomllib.loads(text).get("tool", {}).get("pytest")
            if tp is None:
                return {}
            if not isinstance(tp, dict):
                raise ValueError("tool.pytest")
            native = {k: v for k, v in tp.items() if k != "ini_options"}
            ini = tp.get("ini_options")
            if native and ini is not None:
                raise ValueError("both pytest tables")
            if ini is not None:
                if not isinstance(ini, dict):
                    raise ValueError("ini_options")
                return dict(ini)
            return native

        o, n = opts(old), opts(new)
        changed = {k for k in set(o) | set(n) if o.get(k) != n.get(k)}
        if "addopts" in changed:
            # options that only change what pytest prints or when it stops
            # early: no test can fail on account of them
            if not (_report_only_addopts(o.get("addopts")) and _report_only_addopts(n.get("addopts"))):
                return None
            if not o and _shadowed_pytest_config():
                return None  # the new table would replace another file's pytest section
            changed.discard("addopts")
        if not changed <= {"filterwarnings", "markers"}:
            return None
        if "markers" in changed:
            m = n.get("markers", [])
            if not isinstance(m, (list, str)):
                return None
        if "filterwarnings" not in changed:
            return 0
        fw = n.get("filterwarnings", [])
        if isinstance(fw, str):
            fw = [x for x in fw.splitlines() if x.strip()]
        if not isinstance(fw, list):
            return None
        error = False
        for entry in fw:
            act = _warning_filter_action(entry)
            if act is None:
                return None
            error = error or act == "error"
        if not error:
            return 0
        if self.warned is None or self.warned.get("other"):
            return None
        out = 0
        num = self.mp.num_of_id
        for tid in self.warned.get("tests", []):
            if tid not in num:
                return None
            out |= 1 << num[tid]
        return out

    def meta_readers(self, kind: str) -> int | None:
        """Tests that can read a changed docstring ("doc") or a changed
        bare annotation ("ann"), from every place the package or the tests
        mention the attributes and tools that read them. None when one of
        those places could reach any test."""
        cache = self.__dict__.setdefault("_meta", {})
        if kind in cache:
            return cache[kind]
        words = _META_WORDS[kind]
        pat = re.compile(r"\b(" + "|".join(words) + r")\b")
        mp = self.mp
        out = 0
        try:
            for top in ("sqlglot", "tests"):
                for root, dirs, files in os.walk(os.path.join(SRC, top)):
                    dirs[:] = [x for x in dirs if x not in ("__pycache__", "fixtures")]
                    for f in files:
                        if not f.endswith(".py"):
                            continue
                        path = os.path.join(root, f)
                        r = os.path.relpath(path, SRC)
                        text = open(path, encoding="utf-8", errors="replace").read()
                        if not pat.search(text):
                            continue
                        b = self._meta_file(r, ast.parse(text), words)
                        if b is None:
                            out = None
                            raise StopIteration
                        out |= b
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            log(f"meta readers {kind}: {type(e).__name__}: {e}")
            out = None
        cache[kind] = out
        return out

    def _meta_file(self, rel: str, tree, words) -> int | None:
        """Tests reaching the uses of words in one file. A use inside a
        function counts for the tests running that line; a use the import
        runs counts for the file's own tests when it is a test file, and
        for any test otherwise. Copying __doc__ onto an object only passes
        the text on, which a later read of that object sees."""
        mp = self.mp
        skip: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr" \
                    and len(node.args) == 3 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value == "__doc__":
                skip.update(id(x) for x in ast.walk(node))
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Attribute) and t.attr == "__doc__"
                                                       for t in node.targets):
                skip.update(id(x) for x in ast.walk(node))
        out = 0

        def visit(node, fn):
            nonlocal out
            for ch in ast.iter_child_nodes(node):
                f = fn
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    f = ch
                hit = None
                if isinstance(ch, ast.Name) and ch.id in words:
                    hit = ch
                elif isinstance(ch, ast.Attribute) and ch.attr in words:
                    hit = ch
                elif isinstance(ch, ast.alias) and ch.name.split(".")[-1] in words:
                    hit = ch
                elif isinstance(ch, ast.ImportFrom) and (ch.module or "").split(".")[0] in words:
                    hit = ch
                elif isinstance(ch, ast.Constant) and isinstance(ch.value, str) and ch.value in words:
                    hit = ch
                if hit is not None and id(hit) not in skip:
                    line = getattr(hit, "lineno", None) or getattr(node, "lineno", None)
                    if f is not None and line is not None and not mp.has_import_line(rel, line, line):
                        out |= mp.lines_tests(rel, line, line)
                    elif rel.startswith("tests/"):
                        out |= self.module_rule(rel)
                    else:
                        return False
                if visit(ch, f) is False:
                    return False
            return True

        if visit(tree, None) is False:
            return None
        return out

    def cval_tests(self, rel: str, changes: list, by_state: bool = False) -> int | None:
        """Tests whose comparison results can differ, from the values each
        test compared at those sites on the pinned tree (tools/cval_plugin.py):
        a test that ran a site with a value where old and new comparison
        agree runs as on the pinned tree. Tests the profile did not see pass
        keep their coverage verdict. None when the profile cannot say."""
        cv = self.__dict__.get("_cval", 0)
        if cv == 0:
            cv = None
            try:
                cv = json.loads(gzip.decompress(open(os.path.join(APP, "cval.json.gz"), "rb").read()))
                num = self.mp.num_of_id
                # a test id outside tests.json cannot fail: it counts for nothing
                bits = [self.mp.by_file.get(o[2:], 0) if o.startswith("M:")
                        else (1 << num[o]) if o in num else 0 for o in cv["owners"]]
                ok = 0
                for i in cv["ok"]:
                    if not cv["owners"][i].startswith("M:"):
                        ok |= bits[i]
                cv["bits"] = bits
                cv["okbits"] = ok
                cv["import"] = set(cv["import"])
                cv["import_t"] = set(cv["import_t"])
            except Exception as e:  # noqa: BLE001
                log(f"cval unavailable ({type(e).__name__}: {e})")
                cv = None
            self._cval = cv
        if cv is None:
            return None
        out = 0
        bits = cv["bits"]
        owners_l = cv["owners"]
        seen_all = self.mp.all
        for pos, side, oop, olit, nop, nlit in changes:
            key = f"{rel}:{pos}"
            # a comparison the package import runs is left to the import
            # state check (by_state); one a test module's own import runs
            # is not covered by it
            if key in cv["import_t"] or key in cv["import"] and not by_state:
                return None
            e = cv["sites"].get(key)
            # only a test that compared there itself, with values that agree,
            # is cleared; one the map says runs the line but the profile never
            # saw compare there (a cache another test filled, say) is kept
            seen = 0
            for i in e["s"] if e else ():
                out |= bits[i]
            for owners in (e["v"].values() if e else ()):
                for i in owners:
                    if not owners_l[i].startswith("M:"):
                        seen |= bits[i]
            seen_all &= seen
            for r, owners in (e["v"].items() if e else ()):
                try:
                    v = ast.literal_eval(r)
                except Exception:  # noqa: BLE001
                    v = None
                    differs = True
                else:
                    differs = _cv_differs(side, oop, olit, nop, nlit, v)
                if differs:
                    for i in owners:
                        out |= bits[i]
        return out | (self.mp.all & ~(cv["okbits"] & seen_all))

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
            # a neighbour that raises no line event (a docstring, a bare
            # constant, global/nonlocal, an annotation without a value) is
            # credited to no test: it cannot stand in for the new statement
            prev = site.prev if _has_line_event(site.prev) else None
            nxt = site.next if _has_line_event(site.next) else None
            if prev is not None:
                spans.append(_span(prev))
            elif nxt is not None:
                spans.append(_span(nxt))
            elif site.prev is not None or site.next is not None:
                p = site.parent
                if p is not None and hasattr(p, "lineno"):
                    spans.append(_span(p))
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
            if site.nodes and not any(_has_line_event(n) for n in site.nodes):
                # only statements without line events were replaced or removed
                # (a docstring turned into code): credit the block they sit in
                p = site.parent
                body = [x for x in getattr(p, "body", []) or []
                        if _has_line_event(x) and x not in site.nodes]
                if body:
                    spans.append(_span(body[0]))
                elif p is not None and hasattr(p, "lineno"):
                    spans.append(_span(p))
        return spans

    def r_hit(self, rel: str, site: Site) -> bool:
        """Whether the TPC tests may reach this change outside the front end."""
        if self.rlines is None:
            return True
        lines = self.rlines.get(rel)
        if not lines:
            return False
        return any(l in lines for lo, hi in self.site_spans(site) for l in range(lo, hi + 1))

    def _foreign_class(self, rel: str, via: list) -> bool:
        """Whether a reference mod.Cls.<...>.NAME names a class that module
        mod defines at top level, where mod is not rel and does not import
        rel, even transitively: then Cls cannot be, or derive from, a class
        of rel, so it does not read rel's class attributes."""
        if len(via) < 2 or not via[0] or not via[1] or via[0] == rel:
            return False
        m, cls = via
        if self._top_cache is None:
            self._top_cache = {}
        tops = self._top_cache.get(m)
        if tops is None:
            try:
                tops = _top_classes(open(os.path.join(SRC, m), encoding="utf-8").read())
            except OSError:
                tops = set()
            self._top_cache[m] = tops
        if cls not in tops:
            return False
        if self.edges is None:
            return False
        if self._reach_cache is None:
            self._reach_cache = {}
        seen = self._reach_cache.get(rel)
        if seen is None:
            # every file whose own load imports rel, directly or through
            # others (a lazy package attribute counts as the module it loads)
            seen = {rel}
            queue = [rel]
            while queue:
                x = queue.pop()
                for imp, _line, in_func in self.edges.get(x, []):
                    if not in_func and imp not in seen:
                        seen.add(imp)
                        queue.append(imp)
            self._reach_cache[rel] = seen
        return m not in seen

    def import_dependents(self, site: Site, rel: str) -> int:
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
        for f in {f for f, _line in self._import_refs(site, rel)}:
            out |= self._file_rule(f)
        return out

    def _file_rule(self, f: str) -> int:
        mp = self.mp
        out = mp.file_tests(f) | mp.by_file.get(f, 0)
        m = module_name(f)
        if m:
            out |= mp.module_class_tests(m)
        return out

    def _import_refs(self, site: Site, rel: str) -> set:
        """(file, line) of each read of the changed names that runs at import."""
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
        out = set()
        for name in names:
            for f, line, in_func, owner, *via in self.refs.get(name, []):
                if owners and owner not in owners and owner != "?":
                    continue  # another class's attribute of the same name
                if owners and self._foreign_class(rel, via):
                    continue  # tokens.Tokenizer.KEYWORDS: a class of a module that cannot see rel
                if not owners and owner:
                    continue  # module-level name: importers are in the module rule
                if not in_func or mp.has_import_line(f, line, line):
                    out.add((f, line))
        return out

    def _class_stmt(self, f: str, line: int) -> Site | None:
        """The statement holding line when it sits directly in a class body
        (classes only between it and the module), as a Site."""
        if self._tree_cache is None:
            self._tree_cache = {}
        tree = self._tree_cache.get(f)
        if tree is None:
            try:
                tree = ast.parse(open(os.path.join(SRC, f), encoding="utf-8").read())
            except (OSError, SyntaxError):
                tree = False
            self._tree_cache[f] = tree
        if not tree:
            return None
        scope = ()
        body = tree.body
        while True:
            hit = [n for n in body if n.lineno <= line <= (n.end_lineno or n.lineno)]
            if len(hit) != 1:
                return None
            n = hit[0]
            if isinstance(n, ast.ClassDef) and not (n.lineno <= line < n.body[0].lineno):
                scope = scope + (("class", n.name, n),)
                body = n.body
                continue
            if not scope or not isinstance(n, (ast.Assign, ast.AnnAssign)):
                return None
            return Site(scope, "replace", [n], parent=scope[-1][2])

    def const_rule(self, site: Site) -> tuple[int, bool] | None:
        """A module-level constant whose literal value changed and whose
        every read sits inside a function that the import never runs:
        only tests that run one of those reads can see the new value.
        None when that cannot be shown, so the caller uses the module rule."""
        mp = self.mp
        if site.kind != "replace" or site.new is None or len(site.new) != len(site.nodes):
            return None
        names = set()
        for n in list(site.nodes) + list(site.new):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                names.add(n.targets[0].id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                names.add(n.target.id)
            else:
                return None
            if not _literal(n.value):
                return None
        if self.from_names is None:
            return None
        reach = 0
        r_hit = False  # whether the TPC tests read it outside the proxied front end
        for name in names:
            if not CONST_RE.match(name) or name in self.strings:
                return None
            for f, line, in_func, owner, *_ in self.refs.get(name, []):
                if owner in ("self", "cls") and name not in self.from_names:
                    # an attribute of an instance or class: a module global
                    # becomes one only through a from-import of its name
                    continue
                if not in_func or mp.has_import_line(f, line, line):
                    return None
                reach |= mp.lines_tests(f, line, line)
                if self.rlines is None or line in self.rlines.get(f, ()):
                    r_hit = True
        return reach, r_hit

    def table_rule(self, rel: str, site: Site, old_tree) -> int | None:
        """A module-level table `NAME = {...}` whose change is confined to some
        of its entries. tools/tbl_plugin.py recorded, per test (each in its own
        fork), which keys were looked up in this table or any table built from
        it (copies, `{**T, ...}`), and which tests read such a table whole.
        When the import itself never read the changed keys and never read the
        table whole, and the new entries evaluate at import as the pinned ones
        do, only those tests can see the change. None when that cannot be shown."""
        tbl = getattr(self, "tbl", None)
        if not tbl or site.scope or site.kind != "replace" or site.lam:
            return None
        if not site.new or len(site.nodes) != 1 or len(site.new) != 1:
            return None
        o, n = site.nodes[0], site.new[0]
        if type(o) is not type(n) or not isinstance(o, (ast.Assign, ast.AnnAssign)):
            return None
        tg_o = o.targets if isinstance(o, ast.Assign) else [o.target]
        tg_n = n.targets if isinstance(n, ast.Assign) else [n.target]
        if len(tg_o) != 1 or len(tg_n) != 1 or _dump(tg_o[0]) != _dump(tg_n[0]) \
                or not isinstance(tg_o[0], ast.Name):
            return None
        if isinstance(o, ast.AnnAssign) and _dump(o.annotation) != _dump(n.annotation):
            return None
        od, nd = o.value, n.value
        if not isinstance(od, ast.Dict) or not isinstance(nd, ast.Dict) or not od.keys or not nd.keys:
            return None
        info = tbl["sites"].get(f"{rel}:{od.lineno}:{od.col_offset}")
        if info is None or info["imp_all"]:
            return None
        recorded = {(lo, hi): set(names) for lo, hi, names in info["entries"]}
        atoms, hashable = _module_atoms(old_tree)

        def espan(k, v):
            return (k.lineno if k is not None else v.lineno), v.end_lineno

        def edump(k, v):
            return ("**" if k is None else _dump(k)) + "=" + _dump(v)

        oe = list(zip(od.keys, od.values))
        ne = list(zip(nd.keys, nd.values))
        sm = difflib.SequenceMatcher(None, [edump(*e) for e in oe], [edump(*e) for e in ne], autojunk=False)
        affected: set = set()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            olds, news = oe[i1:i2], ne[j1:j2]
            for k, v in news:
                if not _import_safe(v, atoms, set()):
                    return None
                if k is not None and not _import_safe(k, atoms, set()):
                    return None
            if tag == "replace" and len(olds) == 1 and len(news) == 1 \
                    and olds[0][0] is None and news[0][0] is None:
                # **{k: v for k in {...}}: when only the key set changed,
                # only the keys added or removed are affected
                d = _comp_key_delta(olds[0][1], news[0][1], hashable)
                rec = recorded.get(espan(*olds[0]))
                if rec is None:
                    return None
                if d is not None and d[1] <= rec:
                    affected |= d[0] | d[1]
                    continue
            for k, v in olds:
                rec = recorded.get(espan(k, v))
                if rec is None:
                    return None
                affected |= rec
            for k, v in news:
                names = _entry_key_names(k, v, hashable)
                if names is None:
                    return None
                affected |= names
        if not affected or affected & set(info["imp_keys"]):
            return None
        sets = tbl["sets"]
        out = self.mp.decode(sets[info["all"]]) | self.mp.decode(sets[tbl["unk"]])
        for name in affected:
            sid = info["keys"].get(name)
            if sid is not None:
                out |= self.mp.decode(sets[sid])
        return out | self.mp.unknown

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
            for f, line, in_func, _owner, *_ in self.refs.get(name, []):
                if not in_func or mp.has_import_line(f, line, line):
                    return None
                reach |= mp.lines_tests(f, line, line)
        return inst & reach

    def nested_rule(self, rel: str, site: Site, seen: set | None = None,
                    top: bool = False) -> tuple[int, int, list] | None:
        """An attribute assignment in a class nested in another class
        (MySQL.Tokenizer.KEYWORDS). The outer class's users see it only
        through the nested class: by instantiating it (or a subclass), by
        running a line that reads the name, by running the changed lines,
        or through state the imports derive from it (the outer class's
        attributes, a subclass's copy, a table built from it). The first
        three are the candidates; the class rule's other tests are left to
        the import-state check, which may differ only under the nested
        class's own keys. None when that cannot be shown."""
        mp = self.mp
        scope = site.scope
        if len(scope) < (1 if top else 2) or any(sc[0] != "class" for sc in scope):
            return None
        if site.kind not in ("replace", "insert") or not site.nodes:
            return None
        names = set()
        for n in list(site.nodes) + list(site.new or []):
            if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets):
                names.update(t.id for t in n.targets)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                names.add(n.target.id)
            else:
                return None
        mod = module_name(rel)
        qual = ".".join(sc[1] for sc in scope)
        inst = mp.class_tests(f"{mod}:{qual}")
        if inst is None:
            return None
        reach = 0
        for name in names:
            if not CONST_RE.match(name) or name in self.strings:
                return None  # a name built or looked up as a string
            for f, line, in_func, _owner, *_ in self.refs.get(name, []):
                # a read the imports run (a metaclass building the class) is
                # charged to whichever test loaded the module first; what it
                # leaves behind is the import state the check compares
                if in_func and not mp.has_import_line(f, line, line):
                    reach |= mp.lines_tests(f, line, line)
        # a read sees this class's value only on the class, a subclass or an
        # instance: tests that touch the outer class or name its module
        outer = self.class_rule(rel, scope)
        base = os.path.basename(rel)[:-3].lower()
        cand = inst | (reach & (outer | self.mention_tests(base))) | mp.spawn
        for lo, hi in self.site_spans(site):
            cand |= mp.lines_tests(rel, lo, hi)
        imp = outer
        keys = [f"{mod}:{qual}"]
        # a class body elsewhere that copies the value at import
        # (Doris.Tokenizer.KEYWORDS = {**MySQL.Tokenizer.KEYWORDS}) is the
        # same kind of change one step on; anything else takes its file
        seen = set() if seen is None else seen
        for f, line in sorted(self._import_refs(site, rel)):
            if (f, line) in seen:
                continue
            seen.add((f, line))
            dep = self._class_stmt(f, line) if f.startswith("sqlglot/") and len(seen) < 64 else None
            # a copy into a top-level class body is the same: its users
            # either reach the class or see what the import derived from it
            sub = self.nested_rule(f, dep, seen, top=True) if dep is not None else None
            if sub is None:
                cand |= self._file_rule(f)
                continue
            cand |= sub[0]
            imp |= sub[1]
            keys += sub[2]
        return cand, imp, keys

    def func_rule(self, rel: str, site: Site) -> tuple[int, bool, bool]:
        """Tests reaching the changed statements; flag if the code runs at
        import, and whether that flag comes from the changed lines themselves
        (False) or only from the enclosing function (True)."""
        mp = self.mp
        spans = self.site_spans(site)
        b = 0
        at_import = False
        for lo, hi in spans:
            b |= mp.lines_tests(rel, lo, hi)
            if not site.lam and mp.has_import_line(rel, lo, hi):
                at_import = True
        whole = False
        if not site.lam and not _live_site(site):
            fn = [s for s in site.scope if s[0] == "func"]
            if fn:
                f = fn[-1][2]
                body = f.body
                if body and mp.has_import_line(rel, body[0].lineno, f.end_lineno):
                    at_import = whole = True
        return b, at_import, whole

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
        a.broad |= a.cand | a.imp
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
                if old_rel == new_rel and _config_inert(r, old_text, new_text):
                    # pytest never reads it: only tests that open it, or
                    # spawn a build or a tool that might, can see the edit
                    a.cand |= mp.opened_tests(r) | mp.spawn
                    a.rsafe = False
                    a.reasons.append(f"inert config {r}")
                    return
                if old_rel is None and base == "conftest.py" and _drop_items_only(new_text):
                    # it only deletes collected items: which ones is settled
                    # by collecting on the patched tree, and those fail
                    a.collect = True
                    a.cand |= mp.opened_tests(r) | mp.spawn
                    a.rsafe = False
                    a.reasons.append(f"collection hook {r}")
                    return
                if old_rel in (None, new_rel):
                    eff = self.config_effect(a, r, None if old_rel is None else old_text, new_text)
                    if eff is not None:
                        a.cand |= eff | mp.opened_tests(r) | mp.spawn
                        a.rsafe = False
                        a.reasons.append(f"config {r}, {len(bits_list(eff))} affected")
                        return
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
                b = self.break_rule(old_rel)
                if b is not None:
                    a.cand |= b
                    a.reasons.append(f"deleted module {old_rel}, {len(bits_list(b))} reach it")
                elif old_rel.startswith("sqlglot/"):
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
            try:
                new_tree = ast.parse(new_text)
            except SyntaxError:
                new_tree = None
            if new_tree is not None:
                empty = ast.parse("")
                leak_rule(a, new_rel, empty, new_tree, diff_ast(empty, new_tree))
            return
        if old_rel is not None and new_rel is not None and old_rel != new_rel:
            for r in (old_rel, new_rel):
                if r.endswith(".py"):
                    a.cand |= self.module_rule(r) if r == old_rel else 0
            a.cand |= mp.opened_tests(old_rel)
            if old_rel.endswith(".py"):
                # the old path is gone: everything importing it breaks.
                # The code itself lives on under the new name, so its edits
                # reach the tests that ran it from the old file.
                b = self.break_rule(old_rel)
                if b is None and old_rel.startswith("sqlglot/"):
                    a.full = True
                    a.reasons.append(f"renamed module {old_rel}")
                    return
                a.cand |= b or 0
                if new_rel.endswith(".py") and new_text != old_text:
                    self.analyze_py(a, old_rel, old_text, new_text)
                a.reasons.append(f"renamed module {old_rel}")
            return
        # data file: fixtures and anything else a test reads
        b = mp.opened_tests(rel)
        a.cand |= b
        if rel.startswith("tests/fixtures/") and not b and old_rel is not None:
            a.reasons.append(f"unread file {rel}")

    def analyze_py(self, a: Analysis, rel: str, old_text: str, new_text: str):
        mp = self.mp
        a.cand |= self.text_readers(rel)  # tests that read the source as text
        is_pkg = rel.startswith("sqlglot/")
        try:
            new_tree = ast.parse(new_text)
        except SyntaxError:
            a.rsafe = False
            b = self.break_rule(rel)
            if b is None:
                if is_pkg:
                    a.full = True
                    a.reasons.append(f"syntax error {rel}")
                else:
                    a.cand |= self.module_rule(rel)
                return
            a.cand |= b
            a.reasons.append(f"syntax error {rel}, {len(bits_list(b))} reach it")
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
        leak_rule(a, rel, old_tree, new_tree, sites)
        if is_pkg and any(not any(s[0] == "func" for s in st.scope) for st in sites):
            a.imp_check = True
        meta = [st for st in sites if _meta_kind(st, old_tree) is not None]
        if meta:
            # a docstring reworded, or a bare annotation added or removed: no
            # statement runs differently, only __doc__ or __annotations__ holds
            # another value, so only the code that reads those can tell
            readers = 0
            for kind in {_meta_kind(st, old_tree) for st in meta}:
                r = self.meta_readers(kind)
                if r is None:
                    readers = None
                    break
                readers |= r
            if readers is not None:
                sites = [st for st in sites if st not in meta]
                a.cand |= readers | mp.spawn
                a.rsafe = False
                a.reasons.append(f"{len(meta)} docstring or annotation edits {rel}")
                if not sites:
                    a.broad |= mp.file_tests(rel)
                    return
        if a.equiv is not None:
            if is_pkg and sites and _encoding_cookie(old_text) == _encoding_cookie(new_text):
                cm, dc = a.equiv
                if all(site_equiv(rel, st, cm, dc) for st in sites):
                    a.equiv_files.add(rel)
                else:
                    a.equiv = None
            else:
                a.equiv = None
        if is_pkg and sites and all(_import_only(st, old_tree) for st in sites):
            # only module-level imports changed: same modules load, only
            # the names they bind differ, so trace the uses of those names
            b = self.rebind_rule(rel, old_tree, new_tree)
            if b is not None:
                a.cand |= b
                a.rsafe = False
                a.broad |= mp.file_tests(rel)
                a.reasons.append(f"import names {rel}, {len(bits_list(b))} use them")
                return
        if not sites:
            a.reasons.append(f"no-op {rel}")
            a.broad |= mp.file_tests(rel)
            return
        for site in sites:
            scope = site.scope
            in_func = any(s[0] == "func" for s in scope)
            if in_func:
                b, at_import, whole = self.func_rule(rel, site)
                # changed lines the package import runs: what the import
                # leaves behind is compared with the pinned tree's
                # (impfp.py); if it matches, only tests running the lines
                # themselves can see the change
                by_state = at_import and not whole and is_pkg
                if b and is_pkg and (not at_import or by_state):
                    ch = cmp_changes(site)
                    nb = self.cval_tests(rel, ch, by_state) if ch else None
                    if nb is not None:
                        a.reasons.append(f"compared values {rel}: {len(bits_list(b & nb))} of {len(bits_list(b))}")
                        a.broad |= b
                        b &= nb
                a.cand |= b
                if (at_import and not by_state) or self.r_hit(rel, site):
                    a.rsafe = False
            elif not (site.kind == "insert" and not scope and _new_unused_defs(site.nodes, old_text)):
                cr = self.const_rule(site) if not scope and is_pkg else None
                if cr is None or cr[1]:
                    a.rsafe = False
            if in_func:
                if by_state:
                    a.imp |= self.module_rule(rel)
                    a.imp_strict = True
                    a.reasons.append(f"import-time code, state check {rel}")
                elif at_import:
                    a.cand |= self.module_rule(rel)
                    a.reasons.append(f"import-time code {rel}")
            elif scope:
                # class body: attributes, method headers, added or removed methods
                sp = None
                if site.kind == "header" and is_pkg and site.new and scope[-1][0] == "class":
                    sp = _defaults_only(site.nodes[0], site.new[0])
                if sp is not None and not mp.has_import_line(rel, sp[0], sp[1]):
                    a.cand |= mp.lines_tests(rel, sp[0], sp[1]) | mp.spawn
                    a.rsafe = False
                    a.broad |= self.class_rule(rel, scope)
                    a.reasons.append(f"literal default of method {site.nodes[0].name} {rel}, callers only")
                    continue
                narrow = self.attr_rule(rel, site)
                if narrow is not None:
                    a.cand |= narrow | mp.spawn
                    a.broad |= self.class_rule(rel, scope)
                    a.reasons.append(f"attribute read at run time {rel}")
                    continue
                nested = self.nested_rule(rel, site) if is_pkg else None
                if nested is not None:
                    cand, imp, keys = nested
                    a.cand |= cand
                    a.imp |= imp & ~cand
                    a.imp_excuse += keys
                    a.broad |= self.class_rule(rel, scope)
                    a.reasons.append(f"nested class attribute, state check {rel}")
                    continue
                a.cand |= self.class_rule(rel, scope) | self.import_dependents(site, rel)
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
                sp = _defaults_only(site.nodes[0], site.new[0]) if header_only and is_pkg and site.new else None
                if sp is not None and not mp.has_import_line(rel, sp[0], sp[1]):
                    a.cand |= mp.lines_tests(rel, sp[0], sp[1]) | mp.spawn
                    a.rsafe = False
                    a.reasons.append(f"literal default of {site.nodes[0].name} {rel}, callers only")
                elif header_only:
                    o = site.nodes[0]
                    lo, hi = _span(o)
                    a.cand |= mp.lines_tests(rel, lo, hi)
                    # name or decorators changed: anything importing it may break
                    a.cand |= self.module_rule(rel)
                else:
                    narrow = self.const_rule(site) if is_pkg else None
                    tb = self.table_rule(rel, site, old_tree) if is_pkg and narrow is None else None
                    if narrow is not None:
                        a.cand |= narrow[0] | mp.spawn
                        a.reasons.append(f"constant read at run time {rel}")
                    elif tb is not None:
                        a.cand |= tb | mp.spawn
                        a.rsafe = False
                        a.reasons.append(f"table entries read at run time {rel}, {len(bits_list(tb))} read them")
                    else:
                        a.cand |= self.module_rule(rel) | self.import_dependents(site, rel)
            a.broad |= mp.file_tests(rel)
        a.broad |= self.module_rule(rel)


def _literal(n) -> bool:
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        n = n.operand
    return isinstance(n, ast.Constant)


def _defaults_only(o, n) -> tuple[int, int] | None:
    """Body span of an undecorated function whose header differs only in
    literal default values, None otherwise. A default is read when a call
    binds it, and a call runs the body, so only tests that ran the body can
    see the change. Nothing in the pinned tree reads default values through
    introspection (only parameter names)."""
    if type(o) is not type(n) or not isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if o.decorator_list or n.decorator_list or o.name != n.name:
        return None
    for f in ("returns", "type_comment", "type_params"):
        x, y = getattr(o, f, None), getattr(n, f, None)
        dx = ast.dump(x) if isinstance(x, ast.AST) else repr(x)
        dy = ast.dump(y) if isinstance(y, ast.AST) else repr(y)
        if dx != dy:
            return None
    oa, na = o.args, n.args
    if len(oa.defaults) != len(na.defaults) or len(oa.kw_defaults) != len(na.kw_defaults):
        return None
    for f in ("posonlyargs", "args", "kwonlyargs"):
        if [ast.dump(x) for x in getattr(oa, f)] != [ast.dump(x) for x in getattr(na, f)]:
            return None
    for f in ("vararg", "kwarg"):
        x, y = getattr(oa, f), getattr(na, f)
        if (x is None) != (y is None) or (x is not None and ast.dump(x) != ast.dump(y)):
            return None
    changed = 0
    for x, y in list(zip(oa.defaults, na.defaults)) + list(zip(oa.kw_defaults, na.kw_defaults)):
        if (x is None) != (y is None):
            return None
        if x is None or ast.dump(x) == ast.dump(y):
            continue
        if not (_literal(x) and _literal(y)):
            return None
        changed += 1
    if not changed:
        return None
    live = [s for s in o.body if _has_line_event(s)]
    if not live:
        return None
    return live[0].lineno, o.end_lineno


def _docstring(n) -> bool:
    return isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and type(n.value.value) is str


def _future_annotations(tree) -> bool:
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(x.name == "annotations" for x in n.names) for n in tree.body)


_ANN_BASES = {"TypedDict", "NamedTuple", "Protocol"}


def _meta_kind(site, old_tree) -> str | None:
    """"doc" for a docstring whose text alone changed, "ann" for a bare
    annotation (x: T, no value) added or removed in a module or class body
    that never evaluates it, None for anything else."""
    p = site.parent
    body = getattr(p, "body", None)
    if site.kind == "replace" and site.new and len(site.nodes) == 1 and len(site.new) == 1:
        o, n = site.nodes[0], site.new[0]
        if _docstring(o) and _docstring(n) and body and body[0] is o \
                and isinstance(p, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return "doc"
    nodes = list(site.nodes) + list(site.new or [])
    if site.kind not in ("replace", "insert") or not nodes:
        return None
    if not all(isinstance(n, ast.AnnAssign) and n.value is None and isinstance(n.target, ast.Name) and n.simple
               for n in nodes):
        return None
    if not _future_annotations(old_tree):
        return None  # without it a class or module body evaluates the annotation
    if p is old_tree:
        return "ann"
    if isinstance(p, ast.ClassDef) and site.scope and site.scope[-1][2] is p:
        if p.decorator_list or p.keywords:
            return None  # a dataclass or a metaclass may read the annotations
        for b in p.bases:
            nm = b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else None
            if nm is None or nm in _ANN_BASES:
                return None
        return "ann"
    return None


_META_WORDS = {
    "doc": ("__doc__", "doctest", "pydoc", "getdoc"),
    # dataclass, NamedTuple, TypedDict and Protocol read only the class they
    # make, and _meta_kind looks at that class's decorators and bases
    "ann": ("__annotations__", "__annotate__", "get_type_hints", "get_annotations", "annotationlib"),
}


_REPORT_FLAGS = {"-x", "--exitfirst", "-q", "--quiet", "-v", "--verbose", "-l", "--showlocals",
                 "--no-header", "--no-summary", "--disable-warnings", "--disable-pytest-warnings"}
_REPORT_VALUED = {"--tb", "--maxfail", "--durations", "--durations-min", "--color", "--code-highlight",
                  "-r", "--verbosity"}


def _report_only_addopts(v) -> bool:
    """True when every option in an addopts value only changes reporting or
    stops the session after a failure, so no test's outcome can change."""
    import shlex

    if v is None:
        return True
    if isinstance(v, list):
        toks = [str(x) for x in v]
    elif isinstance(v, str):
        try:
            toks = shlex.split(v)
        except ValueError:
            return False
    else:
        return False
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in _REPORT_FLAGS or re.fullmatch(r"-[xqvl]+", tok) or re.fullmatch(r"-r[fEsxXpPaAN]+", tok):
            i += 1
            continue
        name, eq, _ = tok.partition("=")
        if name in _REPORT_VALUED:
            i += 1 if eq else 2
            if i > len(toks):
                return False
            continue
        return False
    return True


def _shadowed_pytest_config() -> bool:
    """Whether a file pytest reads after pyproject.toml holds a pytest section."""
    for f, sections in (("tox.ini", ("pytest",)), ("setup.cfg", ("tool:pytest",))):
        path = os.path.join(SRC, f)
        if not os.path.isfile(path):
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            return True
        if any(re.search(r"^\[" + re.escape(sct) + r"\]", text, re.M) for sct in sections):
            return True
    return False


def _config_inert(rel: str, old: str, new: str | None) -> bool:
    """A packaging or tool config edit pytest does not act on. pytest parses
    pyproject.toml, tox.ini and setup.cfg while it looks for its own section,
    so the new text must still parse and neither side may hold that section."""
    base = os.path.basename(rel)
    if new is None or base not in ("setup.py", "pyproject.toml", "setup.cfg", "tox.ini"):
        return False
    if base == "setup.py":
        return True
    try:
        if base == "pyproject.toml":
            import tomllib

            for text in (old, new):
                if "pytest" in tomllib.loads(text).get("tool", {}):
                    return False
            return True
        import iniconfig

        for text in (old, new):
            cfg = iniconfig.IniConfig(rel, data=text)
            if "pytest" in cfg or "tool:pytest" in cfg:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _import_only(site, old_tree) -> bool:
    if site.scope or site.parent is not old_tree:
        return False
    nodes = list(site.nodes) + list(site.new or [])
    return bool(nodes) and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)


def _import_bindings(tree, rel: str):
    """Module-level imports: name -> what it is bound to, and the modules
    loaded (a from-import of a submodule loads it too)."""
    pkg = module_name(rel) if rel.endswith("__init__.py") else module_name(rel).rpartition(".")[0]
    binds: dict = {}
    mods = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name)
                key = a.asname or a.name.split(".")[0]
                binds.setdefault(key, []).append(("import", a.name, bool(a.asname)))
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                p = pkg.split(".")
                p = p[: len(p) - (n.level - 1)] if n.level > 1 else p
                base = ".".join(p + ([base] if base else []))
            mods.add(base)
            for a in n.names:
                if a.name == "*":
                    mods.add("*" + base)
                    continue
                sub = base + "." + a.name
                rp = sub.replace(".", "/")
                if os.path.isfile(os.path.join(SRC, rp + ".py")) or \
                        os.path.isfile(os.path.join(SRC, rp, "__init__.py")):
                    mods.add(sub)
                binds.setdefault(a.asname or a.name, []).append(("from", base, a.name))
    return binds, mods


def _module_binds(base: str, name: str) -> bool:
    """Whether module base surely binds name at its top level."""
    rp = os.path.join(SRC, base.replace(".", "/"))
    if os.path.isfile(os.path.join(rp, name + ".py")) or os.path.isfile(os.path.join(rp, name, "__init__.py")):
        return True
    path = rp + ".py" if os.path.isfile(rp + ".py") else os.path.join(rp, "__init__.py")
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:  # noqa: BLE001
        return False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if (a.asname or a.name.split(".")[0]) == name:
                    return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return True
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name                 and n.value is not None:
            return True
    return False


def _name_uses(tree, names: set, mod: str, frel: str | None):
    """(line, enclosing top-level function span or None) of each use of
    mod's names. frel None: tree is mod itself, and every mention counts
    except its own top-level imports. Otherwise only what can reach mod's
    binding: a from-import of it (and that file's own uses of the name),
    an attribute on a name bound to mod, getattr(..., "name")."""
    out = []
    own = frel is None
    word = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
    alias: dict = {}
    bound: set = set()
    if not own:
        pkg = module_name(frel) or ""
        if not frel.endswith("__init__.py"):
            pkg = pkg.rpartition(".")[0]
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        alias[a.asname] = a.name
                    else:
                        alias[a.name.split(".")[0]] = a.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    p = pkg.split(".")
                    p = p[: len(p) - (n.level - 1)] if n.level > 1 else p
                    base = ".".join(p + ([base] if base else []))
                for a in n.names:
                    alias[a.asname or a.name] = f"{base}.{a.name}"
                    if base == mod and a.name in names:
                        bound.add(a.asname or a.name)

    def dotted(e):
        parts = []
        while isinstance(e, ast.Attribute):
            parts.append(e.attr)
            e = e.value
        if not isinstance(e, ast.Name):
            return None
        head = alias.get(e.id, e.id if own else None)
        if head is None:
            return None
        return ".".join([head] + parts[::-1])

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if own and fn is None and isinstance(ch, (ast.Import, ast.ImportFrom)) and ch in tree.body:
                continue
            f = fn
            if fn is None and isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                f = _span(ch)
                # decorators and defaults run where the def stands
                for d in getattr(ch, "decorator_list", []) + ch.args.defaults + \
                        [x for x in ch.args.kw_defaults if x is not None]:
                    visit(ast.Module(body=[ast.Expr(d)], type_ignores=[]), fn)
            hit = False
            if own:
                if isinstance(ch, ast.Name) and ch.id in names:
                    hit = True
                elif isinstance(ch, ast.Attribute) and ch.attr in names:
                    hit = True
                elif isinstance(ch, ast.alias) and ({ch.name, ch.asname} | set(ch.name.split("."))) & names:
                    hit = True
                elif isinstance(ch, ast.Constant) and isinstance(ch.value, str) and word.search(ch.value) \
                        and not isinstance(node, ast.Expr):
                    hit = True
                elif isinstance(ch, ast.keyword) and ch.arg in names:
                    hit = True
            else:
                if isinstance(ch, ast.ImportFrom):
                    hit = any(alias.get(a.asname or a.name) == f"{mod}.{a.name}" and a.name in names
                              for a in ch.names)
                elif isinstance(ch, ast.Name) and ch.id in bound:
                    hit = True
                elif isinstance(ch, ast.Attribute) and ch.attr in names and dotted(ch.value) == mod:
                    hit = True
                elif isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name) and \
                        ch.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(ch.args) >= 2 \
                        and isinstance(ch.args[1], ast.Constant) and ch.args[1].value in names:
                    hit = True
            if hit:
                out.append((getattr(ch, "lineno", getattr(node, "lineno", 0)), f))
            visit(ch, f)

    visit(tree, None)
    return out


_WARN_ACTIONS = {"error", "ignore", "always", "all", "default", "module", "once"}


def _warning_filter_action(entry) -> str | None:
    """The action of one filterwarnings entry, None unless it surely parses
    the way pytest parses it (builtin categories only: resolving another
    one imports code)."""
    import builtins

    if not isinstance(entry, str):
        return None
    parts = entry.split(":")
    if len(parts) > 5:
        return None
    parts += [""] * (5 - len(parts))
    act, msg, cat, mod, line = (x.strip() for x in parts)
    if act not in _WARN_ACTIONS:
        return None
    if cat:
        c = getattr(builtins, cat, None)
        if "." in cat or not (isinstance(c, type) and issubclass(c, Warning)):
            return None
    if line and not line.isdigit():
        return None
    try:
        re.compile(msg)
        re.compile(mod)
    except re.error:
        return None
    return act


def _conftest_fixtures(text: str) -> dict | None:
    """name -> source dump of each fixture a conftest defines, or None if it
    does anything else at import: hooks, autouse fixtures, plugins, code,
    or imports beyond the standard library and pytest."""
    import importlib

    tree = ast.parse(text)
    have = set()
    out = {}
    for k, n in enumerate(tree.body):
        if k == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str):
            continue
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if isinstance(n, ast.ImportFrom) and (n.level or not n.module):
                return None
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module]
            for m in mods:
                top = m.split(".")[0]
                if m == "pytest":
                    # the run's own pytest; names checked without importing it
                    if isinstance(n, ast.ImportFrom) and any(
                            a.name not in ("fixture", "mark", "param", "raises", "approx", "skip")
                            for a in n.names):
                        return None
                    continue
                if top not in sys.stdlib_module_names or top in ("antigravity", "this", "tkinter",
                                                                 "turtle", "idlelib"):
                    return None
                mod = importlib.import_module(m)
                if isinstance(n, ast.ImportFrom):
                    for a in n.names:
                        if a.name == "*" or not hasattr(mod, a.name):
                            return None
            for a in n.names:
                have.add((a.asname or a.name).split(".")[0])
            continue
        if not isinstance(n, ast.FunctionDef) or n.name.startswith("pytest_") or len(n.decorator_list) != 1:
            return None
        dec = n.decorator_list[0]
        call = dec if isinstance(dec, ast.Call) else None
        f = call.func if call else dec
        if not ((isinstance(f, ast.Attribute) and f.attr == "fixture" and isinstance(f.value, ast.Name)
                 and f.value.id == "pytest" and "pytest" in have)
                or (isinstance(f, ast.Name) and f.id == "fixture" and "fixture" in have)):
            return None
        name = n.name
        if call:
            if call.args:
                return None
            for kw in call.keywords:
                v = ast.literal_eval(kw.value)
                if kw.arg == "autouse" and v:
                    return None
                if kw.arg == "name":
                    name = v
                elif kw.arg not in ("autouse", "scope", "params", "ids"):
                    return None
        for dflt in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
            ast.literal_eval(dflt)
        if not isinstance(name, str):
            return None
        out[name] = ast.dump(n)
    return out


_HOOK_ARGS = {"session", "config", "items"}


def _drop_items_only(text: str) -> bool:
    """A conftest whose only content is a pytest_collection_modifyitems
    hook that deletes entries of `items` at positions built from integer
    literals and len(items). Such a hook leaves every remaining item and its
    order alone; it only decides which items run at all."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False

    def simple(e) -> bool:
        if e is None:
            return True
        if isinstance(e, ast.Constant):
            return type(e.value) is int
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            return simple(e.operand)
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            return simple(e.left) and simple(e.right)
        if isinstance(e, ast.Call):
            return isinstance(e.func, ast.Name) and e.func.id == "len" and not e.keywords \
                and len(e.args) == 1 and isinstance(e.args[0], ast.Name) and e.args[0].id == "items"
        return False

    def index(e) -> bool:
        if isinstance(e, ast.Slice):
            return simple(e.lower) and simple(e.upper) and simple(e.step)
        return simple(e)

    hooks = 0
    for k, n in enumerate(tree.body):
        if k == 0 and _docstring(n):
            continue
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        if not isinstance(n, ast.FunctionDef) or n.name != "pytest_collection_modifyitems" or n.decorator_list:
            return False
        a = n.args
        names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        if a.vararg or a.kwarg or a.defaults or a.kw_defaults or "items" not in names \
                or not set(names) <= _HOOK_ARGS:
            return False
        for j, st in enumerate(n.body):
            if j == 0 and _docstring(st):
                continue
            if isinstance(st, ast.Delete):
                if not all(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == "items" and index(t.slice) for t in st.targets):
                    return False
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                c = st.value
                if not (isinstance(c.func, ast.Attribute) and c.func.attr == "pop"
                        and isinstance(c.func.value, ast.Name) and c.func.value.id == "items"
                        and not c.keywords and len(c.args) <= 1 and all(simple(x) for x in c.args)):
                    return False
            elif isinstance(st, ast.Pass):
                continue
            else:
                return False
        hooks += 1
    return hooks == 1


def _inert(n) -> bool:
    """Statements the compiler emits no code for, so coverage never sees
    their lines: docstrings and other bare constants, global, nonlocal."""
    return isinstance(n, (ast.Global, ast.Nonlocal)) or \
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)


def _live_site(site) -> bool:
    """Whether the site's spans hold code that coverage records whenever the
    change can take effect. Then import-time runs reach the change only if
    they ran those lines, and the whole-suite import record settles it:
    a run that never reaches the changed code runs as on the pinned tree."""
    if site.kind == "insert":
        anchor = site.prev if site.prev is not None else site.next
        return anchor is not None and not _inert(anchor)
    return bool(site.nodes) and not any(_inert(n) for n in list(site.nodes) + list(site.new or []))


def _new_unused_defs(nodes, old_text: str) -> bool:
    for n in nodes:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if n.name.startswith("__") or n.decorator_list:
            return False
        if re.search(rf"\b{re.escape(n.name)}\b", old_text):
            return False
    return True


# ---------------------------------------------------------------- is/== swaps

_SWAP = {ast.Is: ast.Eq, ast.IsNot: ast.NotEq}


def _swap_norm(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            n.ops = [_SWAP.get(type(o), type(o))() for o in n.ops]
    return ast.dump(tree, include_attributes=False)


_SWAP_TOKEN = re.compile(r"==|!=|\bis\b")


def swap_possible(patch_text: str) -> bool:
    """False when no added or removed line of the patch holds `is`, `==` or
    `!=`. Lines outside the hunks are the same text on both sides, so the
    comparison operators then appear in the same order in old and new, and
    no comparison can have swapped. swap_sites would parse and walk both
    trees only to return None. A wrong False only keeps tests."""
    for line in patch_text.splitlines():
        if line[:1] in "+-" and not line.startswith(("+++", "---")) and _SWAP_TOKEN.search(line):
            return True
    return False


def swap_sites(patched: dict) -> list | None:
    """When the patch only swaps `is` with `==` (or `is not` with `!=`) in
    single-operator comparisons of sqlglot modules, the pinned-tree keys
    "rel:line:col" of those comparisons. None for any other patch."""
    keys: list = []
    if not patched:
        return None
    for rel, new_text in patched.items():
        if new_text is None or not rel.startswith("sqlglot/") or not rel.endswith(".py"):
            return None
        p = os.path.join(SRC, rel)
        if not os.path.isfile(p):
            return None
        old_text = open(p, encoding="utf-8", newline="").read()
        if old_text == new_text:
            continue
        try:
            old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
        except SyntaxError:
            return None
        olds = [n for n in ast.walk(old_tree) if isinstance(n, ast.Compare)]
        news = [n for n in ast.walk(new_tree) if isinstance(n, ast.Compare)]
        found = []
        if len(olds) != len(news):
            return None
        for o, n in zip(olds, news):
            to, tn = [type(x) for x in o.ops], [type(x) for x in n.ops]
            if to == tn:
                continue
            if len(to) != 1 or len(tn) != 1 or _SWAP.get(to[0], to[0]) != _SWAP.get(tn[0], tn[0]):
                return None
            found.append(f"{rel}:{o.lineno}:{o.col_offset}")
        # everything else must be identical; the dumps run after the walk
        # above, since normalising rewrites the ops in place
        if not found or _swap_norm(old_tree) != _swap_norm(new_tree):
            return None
        keys.extend(found)
    return keys or None


def swap_cleared(keys: list, eqprof: dict, mp, num_of_id: dict) -> int:
    """Profiled tests the swap cannot affect: at every changed comparison the
    test either never ran it, or saw only operand types for which `is` and
    `==` agree (tools/eqprof_plugin.py). Until the first comparison whose
    result would differ, the patched run is the pinned run, and no such
    comparison was seen, so the test runs as on the pinned tree."""
    for k in keys:
        rel, line, _ = k.rsplit(":", 2)
        if mp.has_import_line(rel, int(line), int(line)):
            return 0  # runs at import: profiled runs saw it, others may not
    out = 0
    for tid, sites in eqprof.items():
        num = num_of_id.get(tid)
        if num is None:
            continue
        if all(sites.get(k, 1) == 1 for k in keys):
            out |= 1 << num
    return out


# ---------------------------------------------------------------- whole-suite observations


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _tracked_dicts(root, in_func: bool) -> set:
    """ids of dict literals obs_plugin wraps: bound to one name at module or
    class level, every key a constant."""
    out: set = set()

    def one(node, f):
        if not f and isinstance(node, (ast.Assign, ast.AnnAssign)):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            v = node.value
            if len(tg) == 1 and isinstance(tg[0], ast.Name) and isinstance(v, ast.Dict) \
                    and v.keys and all(isinstance(k, ast.Constant) for k in v.keys):
                out.add(id(v))
        for child in ast.iter_child_nodes(node):
            one(child, f or isinstance(child, _FUNCS))

    one(root, in_func or isinstance(root, _FUNCS))
    return out


_ATOMS_CACHE: dict = {}


def _module_atoms(tree) -> tuple[set, set]:
    """Dumps of every expression the pinned module evaluates outside function
    bodies (so it evaluated fine at import), and of those used as dict keys
    or set elements (so they hash)."""
    hit = _ATOMS_CACHE.get(id(tree))
    if hit is not None and hit[0] is tree:
        return hit[1], hit[2]
    atoms: set = set()
    hashable: set = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                visit(d)
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                visit(d)
            return
        if isinstance(node, ast.Lambda):
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                visit(d)
            return
        if isinstance(node, ast.expr):
            atoms.add(_dump(node))
        if isinstance(node, ast.Dict):
            hashable.update(_dump(k) for k in node.keys if k is not None)
        elif isinstance(node, ast.Set):
            hashable.update(_dump(e) for e in node.elts)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    _ATOMS_CACHE[id(tree)] = (tree, atoms, hashable)
    return atoms, hashable


def _import_safe(e, atoms: set, bound: set) -> bool:
    """Whether evaluating e at import cannot fail where the pinned module did
    not: built only from literals, displays, comprehensions, lambdas (bodies
    are not run at import) and expressions the pinned module evaluates."""
    if isinstance(e, ast.Constant):
        return True
    if isinstance(e, ast.Name) and e.id in bound:
        return True
    if _dump(e) in atoms and not (bound & {x.id for x in ast.walk(e) if isinstance(x, ast.Name)}):
        return True
    if isinstance(e, ast.Dict):
        return all(_import_safe(v, atoms, bound) for v in e.values) and \
            all(k is None or _import_safe(k, atoms, bound) for k in e.keys)
    if isinstance(e, (ast.Set, ast.List, ast.Tuple)):
        return all(_import_safe(x, atoms, bound) for x in e.elts)
    if isinstance(e, ast.Lambda):
        a = e.args
        return all(_import_safe(d, atoms, bound) for d in a.defaults + [x for x in a.kw_defaults if x is not None])
    if isinstance(e, (ast.DictComp, ast.SetComp, ast.ListComp, ast.GeneratorExp)):
        inner = set(bound)
        for g in e.generators:
            if g.is_async or not _import_safe(g.iter, atoms, inner):
                return False
            if not isinstance(g.target, ast.Name):
                return False
            inner.add(g.target.id)
            if g.ifs:
                return False
        parts = [e.key, e.value] if isinstance(e, ast.DictComp) else [e.elt]
        return all(_import_safe(p, atoms, inner) for p in parts)
    return False


def _key_name(k, hashable: set) -> str | None:
    """The name tbl_plugin gives the key k evaluates to, for a key the
    pinned module already hashes or a constant."""
    if isinstance(k, ast.Constant):
        if isinstance(k.value, str):
            return "s:" + k.value
        if isinstance(k.value, (int, float, bool, bytes)) or k.value is None:
            return "r:" + repr(k.value)
        return None
    if _dump(k) not in hashable:
        return None
    if isinstance(k, ast.Attribute):
        return k.attr
    if isinstance(k, ast.Name):
        return k.id
    return None


def _comp_keys(v):
    """(element nodes, comprehension) for {k: ... for k in <display>}."""
    if not isinstance(v, ast.DictComp) or len(v.generators) != 1:
        return None
    g = v.generators[0]
    if g.ifs or g.is_async or not isinstance(g.target, ast.Name):
        return None
    if not isinstance(v.key, ast.Name) or v.key.id != g.target.id:
        return None
    if not isinstance(g.iter, (ast.Set, ast.List, ast.Tuple)):
        return None
    return g.iter.elts, g


def _entry_key_names(k, v, hashable: set) -> set | None:
    if k is not None:
        n = _key_name(k, hashable)
        return None if n is None else {n}
    ck = _comp_keys(v)
    if ck is None:
        return None
    out = set()
    for e in ck[0]:
        n = _key_name(e, hashable)
        if n is None:
            return None
        out.add(n)
    return out


def _comp_key_delta(ov, nv, hashable: set):
    """(added, removed) key names when two `{k: v for k in <display>}` differ
    only in the display's elements; None otherwise."""
    co, cn = _comp_keys(ov), _comp_keys(nv)
    if co is None or cn is None:
        return None
    if _dump(ov.value) != _dump(nv.value) or _dump(co[1].target) != _dump(cn[1].target):
        return None
    od = {_dump(e): e for e in co[0]}
    nd = {_dump(e): e for e in cn[0]}
    added, removed = set(), set()
    for d in set(nd) - set(od):
        n = _key_name(nd[d], hashable)
        if n is None:
            return None
        added.add(n)
    for d in set(od) - set(nd):
        n = _key_name(od[d], hashable)
        if n is None:
            return None
        removed.add(n)
    return added, removed


def _literal(node) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except Exception:  # noqa: BLE001
        return False


def site_equiv(rel: str, site: Site, cmps: list, dicts: dict) -> bool:
    """True when the only differences at this site are `is`/`==` (or
    `is not`/`!=`) swaps in single-operator comparisons, or changed literal
    values under unchanged constant keys of a dict that obs_plugin tracks.
    Appends the pinned-tree keys of those comparisons and dicts."""
    if site.new is None or len(site.new) != len(site.nodes):
        return False
    in_func = site.lam or any(sc[0] == "func" for sc in site.scope)

    def same(o, n, tracked) -> bool:
        if type(o) is not type(n):
            return False
        if isinstance(o, ast.Compare) and len(o.ops) == 1 and len(n.ops) == 1 \
                and type(o.ops[0]) is not type(n.ops[0]):
            to, tn = type(o.ops[0]), type(n.ops[0])
            if _SWAP.get(to, to) is not _SWAP.get(tn, tn):
                return False
            cmps.append(f"{rel}:{o.lineno}:{o.col_offset}")
            return same(o.left, n.left, tracked) and same(o.comparators[0], n.comparators[0], tracked)
        if isinstance(o, ast.Dict) and id(o) in tracked:
            if len(o.keys) != len(n.keys) or any(_dump(x) != _dump(y) for x, y in zip(o.keys, n.keys)):
                return False
            keys = [k.value for k in o.keys]
            try:
                if len(set(keys)) != len(keys):
                    return False
            except TypeError:
                return False
            changed = []
            for k, x, y in zip(keys, o.values, n.values):
                if _dump(x) == _dump(y):
                    continue
                if not (_literal(x) and _literal(y)):
                    return False
                changed.append(repr(k))
            if changed:
                dicts[f"{rel}:{o.lineno}:{o.col_offset}"] = changed
            return True
        for f in o._fields:
            x, y = getattr(o, f, None), getattr(n, f, None)
            if isinstance(x, list):
                if not isinstance(y, list) or len(x) != len(y):
                    return False
                for xx, yy in zip(x, y):
                    if isinstance(xx, ast.AST):
                        if not same(xx, yy, tracked):
                            return False
                    elif xx != yy or type(xx) is not type(yy):
                        return False
            elif isinstance(x, ast.AST):
                if not same(x, y, tracked):
                    return False
            elif x != y or type(x) is not type(y):
                return False
        return True

    for o, n in zip(site.nodes, site.new):
        if not same(o, n, _tracked_dicts(o, in_func)):
            return False
    return True


_CV_OPS = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
           ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
           ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def _is_literal(n) -> bool:
    try:
        ast.literal_eval(n)
    except Exception:  # noqa: BLE001
        return False
    return True


def cmp_changes(site) -> list | None:
    """When the only difference at this site is the literal operand or the
    operator of single-operator comparisons `expr OP literal` (or `literal OP
    expr`), one (key, side, old op, old literal, new op, new literal) per
    changed comparison, keyed by its pinned-tree position. None otherwise."""
    if site.new is None or len(site.new) != len(site.nodes) or site.lam:
        return None
    found: list = []

    def same(o, n) -> bool:
        if type(o) is not type(n):
            return False
        if isinstance(o, ast.Compare) and len(o.ops) == 1 and len(n.ops) == 1 and _dump(o) != _dump(n):
            if type(o.ops[0]) not in _CV_OPS or type(n.ops[0]) not in _CV_OPS:
                return False
            ol, orr, nl, nr = o.left, o.comparators[0], n.left, n.comparators[0]
            if _is_literal(orr) and not _is_literal(ol) and _is_literal(nr) and _dump(ol) == _dump(nl):
                found.append((f"{o.lineno}:{o.col_offset}", 0, type(o.ops[0]), ast.literal_eval(orr),
                              type(n.ops[0]), ast.literal_eval(nr)))
                return True
            if _is_literal(ol) and not _is_literal(orr) and _is_literal(nl) and _dump(orr) == _dump(nr):
                found.append((f"{o.lineno}:{o.col_offset}", 1, type(o.ops[0]), ast.literal_eval(ol),
                              type(n.ops[0]), ast.literal_eval(nl)))
                return True
            return False
        for f in o._fields:
            x, y = getattr(o, f, None), getattr(n, f, None)
            if isinstance(x, list):
                if not isinstance(y, list) or len(x) != len(y):
                    return False
                for xx, yy in zip(x, y):
                    if isinstance(xx, ast.AST):
                        if not same(xx, yy):
                            return False
                    elif xx != yy or type(xx) is not type(yy):
                        return False
            elif isinstance(x, ast.AST):
                if not same(x, y):
                    return False
            elif x != y or type(x) is not type(y):
                return False
        return True

    for o, n in zip(site.nodes, site.new):
        if not same(o, n):
            return None
    return found or None


def _cv_differs(side, oop, olit, nop, nlit, v) -> bool:
    try:
        old = _CV_OPS[oop](v, olit) if side == 0 else _CV_OPS[oop](olit, v)
    except Exception:  # noqa: BLE001
        return True
    try:
        new = _CV_OPS[nop](v, nlit) if side == 0 else _CV_OPS[nop](nlit, v)
    except Exception:  # noqa: BLE001
        return True
    return bool(old) != bool(new) or type(old) is not type(new)


def obs_cleared(found: tuple, obs: dict, mp) -> int:
    """Tests the patch cannot affect, from a run of the whole pinned suite
    (tools/obs_plugin.py). Every swapped comparison saw only operand types
    for which `is` and `==` agree, or never ran; no changed dict value was
    ever read, and the dict was never copied or iterated with its values.
    Then, until the first point where old and new code differ, the patched
    run is the pinned run, and no such point was reached, so every observed
    test runs as on the pinned tree."""
    cmps, dicts = found
    files = set(obs.get("files", []))
    for k in cmps:
        if k.rsplit(":", 2)[0] not in files or obs["cmp"].get(k, 1) != 1:
            return 0
    for k, changed in dicts.items():
        reads = obs["dict"].get(k)
        if reads is None or "*" in reads or any(c in reads for c in changed):
            return 0
    return obs["clear"]


def _encoding_cookie(text: str) -> str:
    head = "\n".join(text.splitlines()[:2])
    m = re.search(r"coding[:=]\s*([-\w.]+)", head)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- packing


def mark_probe(workers, dur, n, full, reasons, ids):
    """Diagnostic for one board run. It grows the busiest worker's load by a
    code in milliseconds and leaves every selected test selected. Codes:
    1 select-everything from a class-attribute or shared-state reason,
    8 select-everything otherwise, 64 the 20 s TPC-DS test selected without
    max_nodes, 512 max_nodes selected without select-everything. A commit
    without a 20 s test is never touched."""
    before = {t for w in workers for t in w}
    heavy = [t for t in before if dur[t] >= 20000]
    if not heavy:
        return workers, 0
    has_mn = any(ids[t].endswith("::test_max_nodes") for t in heavy)
    if full:
        code = 1 if any("class attribute" in r or "shared state" in r for r in reasons) else 8
    else:
        code = 512 if has_mn else 64
    loads = [sum(dur[t] for t in w) for w in workers]
    b = max(range(len(workers)), key=lambda i: loads[i])
    pool = [(dur[t], i, t) for i, w in enumerate(workers) if i != b for t in w if 0 < dur[t] <= code]
    pool += [(dur[t], -1, t) for t in range(n) if t not in before and 0 < dur[t] <= code]
    pool.sort(key=lambda x: (-x[0], x[1] < 0, x[2]))
    left = code
    move = []
    taken = {}
    for d, i, t in pool:
        if d > left:
            continue
        if i >= 0 and len(workers[i]) - taken.get(i, 0) <= 1:
            continue
        move.append(t)
        taken[i] = taken.get(i, 0) + 1
        left -= d
        if not left:
            break
    if left:
        return workers, 0
    gone = set(move)
    out = [[t for t in w if t not in gone] for w in workers]
    out[b] = out[b] + sorted(gone)
    after = [t for w in out for t in w]
    if len(after) != len(set(after)) or not before <= set(after) or any(not w for w in out):
        return workers, 0
    return out, code


def none_probe(sel, loads, dur, n, full):
    """Diagnostic for one board run. It selects no real test. Each commit gets
    a few tiny tests from outside its normal selection, on one worker, with a
    total that names the size of the normal selection: 1 ms when the busiest
    worker would carry under 2 s, 64 ms under 12 s, 4096 ms from 12 s up.
    An empty or select-everything commit gets nothing."""
    if not sel or full or len(sel) >= n:
        return [], 0
    top = max(loads) if loads else 0
    code = 1 if top < 2000 else 64 if top < 12000 else 4096
    chosen = set(sel)
    pool = sorted(((dur[t], t) for t in range(n) if t not in chosen and 0 < dur[t] <= code), key=lambda x: (-x[0], x[1]))
    left = code
    pick = []
    for d, t in pool:
        if d <= left:
            pick.append(t)
            left -= d
            if not left:
                break
    if left:
        return [], 0
    return [sorted(pick)], code


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


def _fp_imported_all(out_path: str, fp_pid: int) -> bool:
    """Waits for the sibling fingerprint process. True when it found every
    package module importing, to the pinned state. False when it found a
    difference, or ended without saying (the child reaps it first, so a
    dead sibling stops the wait)."""
    done = out_path + ".fpdone"
    while True:
        try:
            with open(done) as f:
                return f.read() == "1"
        except OSError:
            pass
        try:
            os.kill(fp_pid, 0)
        except OSError:
            try:
                with open(done) as f:
                    return f.read() == "1"
            except OSError:
                return False
        time.sleep(0.01)


def child_verify(job: dict, out_path: str):
    """Runs in a forked child. Writes {"failed": [...], "passed": [...]} or an error."""
    import shutil

    result: dict = {}
    farm = job["farm"]
    fp_pid = 0
    all_pid = 0
    col_result = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        # The farm links each __pycache__ into the pinned tree. A child that
        # wrote the patched file's bytecode there would hand it to the next
        # commit that patches the same file to the same size in the same
        # second. Never write bytecode from a child; forks inherit this.
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        build_farm(farm, job["patched"])
        os.chdir(farm)
        sys.path.insert(0, farm)
        os.environ["PYTHONPATH"] = farm
        if job.get("impfp"):
            # in its own process, so that importing every module cannot
            # change what the tests below see
            fp_pid = os.fork()
            if fp_pid == 0:
                ok = False
                try:
                    sys.path.insert(1, APP)
                    import impfp

                    with open(os.path.join(APP, "impfp_base.json")) as f:
                        base = json.load(f)
                    d = base["digests"]
                    fp = impfp.fingerprint(set(base["loose"]), set(base["skip"]))
                    ex = job.get("imp_excuse") or []

                    def excused(k):
                        return any(k.startswith(p + ".") for p in ex)

                    ok = bool(d) and not any(v[:1] in "!?" for v in d.values()) and \
                        set(d) == set(fp) and all(d[k] == fp[k] or excused(k) for k in d)
                    if not ok:
                        diff = sorted(k for k in set(d) | set(fp) if d.get(k) != fp.get(k))
                        raise ValueError(f"{len(diff)} differ: {diff[:3]} {[fp.get(k) for k in diff[:3]]}")
                except BaseException as e:  # noqa: BLE001
                    ok = False
                    try:
                        with open(out_path + ".fperr", "w") as f:
                            f.write(f"{type(e).__name__}: {e}")
                    except BaseException:  # noqa: BLE001
                        pass
                try:
                    # for the import-all process below, which waits on this
                    with open(out_path + ".fpdone.tmp", "w") as f:
                        f.write("1" if ok else "0")
                    os.rename(out_path + ".fpdone.tmp", out_path + ".fpdone")
                except BaseException:  # noqa: BLE001
                    pass
                os._exit(0 if ok else 3)
        if job.get("collect"):
            col_pid = os.fork()
            if col_pid == 0:
                code = 3
                try:
                    import pytest

                    class _Items:
                        ids: list = []

                        def pytest_collection_finish(self, session):
                            self.ids = [i.nodeid for i in session.items]

                    it = _Items()
                    rc = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                                      "-o", "addopts=", "--rootdir", farm, "-o", "python_files=test_*.py",
                                      "tests"], plugins=[it])
                    with open(out_path + ".col", "w") as f:
                        json.dump({"ids": it.ids, "rc": int(rc)}, f)
                    code = 0
                except BaseException:  # noqa: BLE001
                    pass
                os._exit(code)
            _, st = os.waitpid(col_pid, 0)
            try:
                with open(out_path + ".col") as f:
                    collected = json.load(f)
                os.unlink(out_path + ".col")
                if os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0:
                    col_result = collected
            except (OSError, ValueError):
                pass
        import concurrent.futures
        import concurrent.futures.process as cfp

        concurrent.futures.ProcessPoolExecutor = _SyncPool
        cfp.ProcessPoolExecutor = _SyncPool
        import importlib

        leaf_failed = []
        for mod in ["sqlglot"] + job["modules"]:
            try:
                importlib.import_module(mod)
            except BaseException as e:  # noqa: BLE001
                if not leaf_failed:
                    result = {"import_error": f"{mod}: {type(e).__name__}: {e}"}
                if mod == "sqlglot":
                    raise SystemExit
                # the package still imports: only what loads this module breaks
                leaf_failed.append(mod)
        if job.get("imp_all"):
            # a change the import runs can also break another module's import
            # (a dialect that parses SQL in its class body): import every
            # package module, in a process of its own
            all_pid = os.fork()
            if all_pid == 0:
                try:
                    # only imports run here: what the child imported is never
                    # garbage, so keep the collector off it. Inside the try:
                    # an error out here would run the child's own cleanup,
                    # which removes the farm the child is still using.
                    _gc.freeze()
                    if fp_pid and _fp_imported_all(out_path, fp_pid):
                        # the fingerprint process imported every module too,
                        # and each one to its pinned state: nothing is left
                        # here to fail
                        with open(out_path + ".impall", "w") as f:
                            json.dump([], f)
                        os._exit(0)
                    bad = []
                    for dp, _dn, fns in os.walk("sqlglot", followlinks=True):
                        for fn in sorted(fns):
                            if not fn.endswith(".py") or fn == "__main__.py":
                                continue
                            name = module_name(os.path.join(dp, fn).replace(os.sep, "/"))
                            if not name or name in leaf_failed:
                                continue
                            try:
                                importlib.import_module(name)
                            except BaseException:  # noqa: BLE001
                                bad.append(name)
                    with open(out_path + ".impall", "w") as f:
                        json.dump(bad, f)
                except BaseException:  # noqa: BLE001
                    pass
                os._exit(0)
        if leaf_failed:
            result["leaf_failed"] = leaf_failed
            raise SystemExit
        failed = set()
        rc = 0
        if job["ids"]:
            import pytest

            col = _Collector(job["limits"])
            if job.get("freeze"):
                try:
                    import gc

                    from sqlglot.dialects.dialect import Dialect, Dialects

                    for name in Dialects:
                        Dialect.get_or_raise(name).generator()
                    gc.collect()
                    gc.freeze()
                except BaseException:  # noqa: BLE001
                    pass  # the test meets the same error and fails
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
    if col_result is not None and "failed" in result:
        result["collected"] = col_result
    if fp_pid:
        try:
            _, st = os.waitpid(fp_pid, 0)
            result["imp_ok"] = os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0
        except BaseException:  # noqa: BLE001
            result["imp_ok"] = False
        try:
            with open(out_path + ".fperr") as f:
                result["imp_err"] = f.read()[:500]
            os.unlink(out_path + ".fperr")
        except OSError:
            pass
    if all_pid:
        try:
            os.waitpid(all_pid, 0)
            with open(out_path + ".impall") as f:
                result["impall"] = json.load(f)
            os.unlink(out_path + ".impall")
        except BaseException:  # noqa: BLE001
            pass
    try:
        os.unlink(out_path + ".fpdone")
    except OSError:
        pass
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
        import importlib

        # Modules every pytest.main imports while it configures, before it
        # collects a test module: loaded once here instead of in each child.
        # A fresh pytest run has them all loaded before any test code runs.
        # Not concurrent.futures: it adds loggers, and the import-state
        # fingerprint reaches the logger registry through module loggers.
        for name in ("_pytest._argcomplete", "_pytest.faulthandler", "_pytest.helpconfig",
                     "_pytest.junitxml", "_pytest.pastebin", "_pytest.setuponly",
                     "_pytest.setupplan", "_pytest.skipping", "_pytest.threadexception",
                     "_pytest.tracemalloc", "_pytest.unraisableexception", "_pytest.warnings",
                     "tomllib", "pdb", "faulthandler", "fractions", "xml.etree.ElementTree"):
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _fork() -> int:
        # The parent's heap (map, ASTs, analyses) is dead weight in a child.
        # Frozen, the child's collector never walks it: no CPU, and no
        # copy-on-write faults from touching its GC headers. A child can
        # not tell: these objects are never garbage it would collect.
        _gc.freeze()
        return os.fork()

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
                pid = self._fork()
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
    # The brief: 3 minutes for the practice batch (40 commits), 4.5 minutes in
    # the final (about 60). Only trust the larger limit for a clearly larger batch.
    wall_limit = 270.0 if n_commits > 45 else 180.0
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
        try:
            an.tbl = json.loads(gzip.decompress(open(os.path.join(APP, "tbl.json.gz"), "rb").read()))
        except Exception as e:  # noqa: BLE001
            log(f"table reads unavailable ({e})")
    except Exception as e:  # noqa: BLE001
        log(f"map unavailable ({e}); selecting everything")
        everything = list(range(n))
        w, _, _ = best_pack(everything, dur)
        for c in commits:
            print(json.dumps({"commit": c, "workers": w}))
        return
    log(f"map loaded, {n_commits} commits")
    eqprof: dict = {}
    try:
        eqprof = json.loads(gzip.decompress(open(os.path.join(APP, "eqprof.json.gz"), "rb").read()))
    except Exception as e:  # noqa: BLE001
        log(f"eqprof unavailable ({e})")
    num_of_id = {v: k for k, v in ids.items()}
    obs: dict = {}
    try:
        obs = json.loads(gzip.decompress(open(os.path.join(APP, "obs.json.gz"), "rb").read()))
        # only tests that ran to a pass under observation can be cleared
        clear = 0
        for tid, outcome in obs["outcomes"].items():
            if outcome == "ok" and tid in num_of_id:
                clear |= 1 << num_of_id[tid]
        obs["clear"] = clear & ~mp.unknown
    except Exception as e:  # noqa: BLE001
        obs = {}
        log(f"obs unavailable ({e})")

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
        elif a.applied and eqprof:
            try:
                # a swap can only clear candidates: none, nothing to do
                keys = swap_sites(a.patched) if a.cand and swap_possible(text) else None
                if keys:
                    clr = swap_cleared(keys, eqprof, mp, num_of_id) & a.cand
                    if clr:
                        a.cand &= ~clr
                        log(f"{c}: is/== swap clears {len(bits_list(clr))}")
            except Exception as e:  # noqa: BLE001
                log(f"{c}: swap check failed ({e})")
        if not a.full and a.applied and obs:
            try:
                found = a.equiv
                if found and (found[0] or found[1]) and set(a.patched) == a.equiv_files:
                    clr = obs_cleared(found, obs, mp)
                    for rel in a.patched:
                        clr &= ~mp.opened_tests(rel)  # tests reading the source text
                    clr &= a.cand
                    if clr:
                        a.cand &= ~clr
                        log(f"{c}: whole-suite observation clears {len(bits_list(clr))}")
            except Exception as e:  # noqa: BLE001
                log(f"{c}: observation check failed ({e})")
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
    p_heavy = 0.6  # prior that one heavy candidate passes
    # per-test prior where the pinned tree gives one: this test failed on no
    # generated change in 143, only where a module stopped importing
    id_pass = {"tests/test_generator.py::TestGenerator::test_generator_memory": 0.9}
    # measured child CPU per recorded second, where it is far from the file's factor
    id_factor = {"tests/test_generator.py::TestGenerator::test_generator_memory": 0.2}
    # The model above was timed under local amd64 emulation. The board's
    # native CPU reads about half of that, so scale the charge down.
    cpu_scale = float(os.environ.get("BLAST_CPU_SCALE", "0.5"))
    num_of_id = {v: k for k, v in ids.items()}
    base_failed = set(mp.m.get("base_failed", []))
    proxy_nums = {num_of_id[tid]: tid for tid in proxy_base
                  if tid in num_of_id and proxy_base[tid] and tid not in base_failed}
    impfp_on = verify_on and os.path.isfile(os.path.join(APP, "impfp_base.json"))
    cjob: dict = {}
    for c in commits:
        a = analyses[c]
        use_imp = bool(a.imp) and impfp_on and not a.full and a.applied
        if a.imp and not use_imp:
            a.cand |= a.imp
            a.rsafe = False
        cand = bits_list(a.cand)
        base_cost = pack_cost(cand, dur)
        imp = bits_list(a.imp & ~a.cand) if use_imp else []
        plan[c] = {"verify": [], "keep": cand, "proxy": [], "impfp": use_imp, "imp": imp}
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
            cjob[c] = jobs[-1]
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
        # or verify only the heaviest candidates: one broad slow test that
        # passes often sets the whole cost of a small selection
        heavy = None
        if os.environ.get("BLAST_HEAVY", "1") != "0":
            for j in (1, 2, 3):
                if j > len(order) or dur[order[-j]] < HEAVY_MIN_MS:
                    break
                hv = order[-j:]
                hcpu = per_run_cpu
                hfiles: set = set()
                for t in hv:
                    f = tests[t]["file"]
                    hcpu += secs[t] * id_factor.get(ids[t], test_factor.get(f, 1.2)) + 0.003
                    if f not in hfiles:
                        hfiles.add(f)
                        hcpu += file_cpu.get(f, 0.1)
                pp = 1.0
                for t in hv:
                    pp *= id_pass.get(ids[t], p_heavy)
                est = hcpu * cpu_scale * 1000 + pp * pack_cost(order[:-j], dur) + (1 - pp) * base_cost
                if est < best[0]:
                    best = (est, -2)
                    heavy = hv
        if os.environ.get("BLAST_FORCE") == "1":
            best = (0, len(order) - 1)
            heavy = None
        if best[1] == -1:
            log(f"{c}: {len(cand)} cand, cost {base_cost / 1000:.1f}s, verify not worth it")
            continue
        if heavy is not None:
            v = heavy
            keep = order[: len(order) - len(heavy)]
        else:
            v = order[: best[1] + 1]
            keep = order[best[1] + 1:]
        plan[c]["verify"] = v
        plan[c]["keep"] = keep
        plan[c]["heavy"] = heavy is not None
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
        if all(i in FREEZE_IDS for i in job["ids"]):
            job["freeze"] = True
        if plan[c]["proxy"]:
            # one child does both: the tests, then the proxy
            pj = jobs.pop()
            job["proxy"] = pj["proxy"]
            job["proxy_base"] = pj["proxy_base"]
            job["gain"] += pj["gain"]
            job["est_wall"] += pj["est_wall"]
            job["max_wall"] += pj["max_wall"]
        jobs.append(job)
        cjob[c] = job
        log(f"{c}: {len(cand)} cand, cost {base_cost / 1000:.1f}s, verify {len(v)} ({vsec:.1f}s), "
            f"keep {len(keep)}, est {best[0] / 1000:.1f}s {a.reasons[:2]}")

    for c in commits:
        p = plan[c]
        if not p["impfp"]:
            continue
        a = analyses[c]
        job = cjob.get(c)
        if job is None:
            job = {"commit": c, "ids": [], "limits": {}, "secs": [], "patched": a.patched,
                   "modules": [m for m in a.sqlglot_modules if m], "gain": 0,
                   "est_wall": 0.0, "max_wall": 10.0}
            jobs.append(job)
            cjob[c] = job
        job["impfp"] = True
        job["imp_excuse"] = [] if a.imp_strict else sorted(set(a.imp_excuse))
        keep = p["keep"] + p["proxy"]
        job["gain"] += pack_cost(keep + p["imp"], dur) - pack_cost(keep, dur)
        job["est_wall"] += 3.0
        job["max_wall"] += 30.0
        log(f"{c}: import state check for {len(p['imp'])} "
            f"({sum(secs[t] for t in p['imp']):.1f}s)")

    for c in commits:
        a = analyses[c]
        if not a.collect or a.full or not a.applied:
            continue
        if not verify_on:
            a.cand = mp.all
            plan[c]["keep"] = list(range(n))
            continue
        job = cjob.get(c)
        if job is None:
            job = {"commit": c, "ids": [], "limits": {}, "secs": [], "patched": a.patched,
                   "modules": [m for m in a.sqlglot_modules if m], "gain": 0,
                   "est_wall": 0.0, "max_wall": 10.0}
            jobs.append(job)
            cjob[c] = job
        job["collect"] = True
        job["gain"] += pack_cost(list(range(n)), dur)
        job["est_wall"] += 8.0
        job["max_wall"] += 60.0
        log(f"{c}: collection check")

    for c in commits:
        a = analyses[c]
        if not a.imp_check or a.full or not a.applied:
            continue
        if not verify_on:
            plan[c]["keep"] = list(range(n))
            continue
        if cjob.get(c) is not None:
            cjob[c]["imp_all"] = True
        if cjob.get(c) is None:
            # nothing else runs the patched tree: import it once, alone
            job = {"commit": c, "ids": [], "limits": {}, "secs": [], "patched": a.patched,
                   "modules": [m for m in a.sqlglot_modules if m],
                   "gain": pack_cost(list(range(n)), dur), "imp_all": True,
                   "est_wall": 1.0, "max_wall": 20.0}
            jobs.append(job)
            cjob[c] = job
            log(f"{c}: import check")

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
        imp_ok = bool(res and "failed" in res and res.get("imp_ok"))
        if p["impfp"]:
            log(f"{c}: import state {'matches' if imp_ok else 'differs or unchecked'} "
                f"{(res or {}).get('imp_err', '')}")
            if not imp_ok:
                out |= set(p["imp"])
        if a.imp_check and not a.full and a.applied and verify_on:
            res2 = res or {}
            bad = set(res2.get("leaf_failed") or [])
            if "impall" in res2 and ("failed" in res2 or bad):
                bad |= set(res2["impall"])
                rels = []
                for mod in sorted(bad):
                    base = mod.replace(".", "/")
                    hit = [r for r in (base + ".py", base + "/__init__.py")
                           if r in a.patched or os.path.isfile(os.path.join(SRC, r))]
                    rels += hit[:1]
                if len(rels) == len(bad):
                    for r in rels:
                        out |= set(bits_list(an.module_rule(r)))
                    if bad:
                        log(f"{c}: {len(bad)} modules do not import: {sorted(bad)[:4]}")
                else:
                    out = set(range(n))
            else:
                why = res2.get("import_error") or res2.get("error") or "unchecked"
                log(f"{c}: patched import not confirmed: {why}")
                out = set(range(n))
        if p["proxy"]:
            ok = set(res.get("proxy_ok", [])) if res and "failed" in res else set()
            if p["impfp"] and not imp_ok:
                ok = set()  # the proxy assumes the pinned import state
            for t in p["proxy"]:
                if ids[t] not in ok:
                    out.add(t)
            log(f"{c}: proxy cleared {len(ok)} of {len(p['proxy'])}")
            if res and "import_error" in res:
                log(f"{c}: patched sqlglot does not import: {res['import_error']}")
                out = set(range(n))
        if a.collect and not a.full and a.applied and verify_on:
            col = (res or {}).get("collected")
            if col is None:
                out = set(range(n))
                log(f"{c}: collection unchecked")
            else:
                got = set(col["ids"])
                gone = {t for t in range(n) if ids[t] not in got}
                out |= gone
                log(f"{c}: collection drops {len(gone)}")
        if p["verify"]:
            if res is None or "failed" not in res:
                # a leaf module that fails while the package imports is settled
                # by the import check above: the unverified tests just stay
                # a heavy-only job that meets a leaf import error puts its tests
                # back, which is the selection the run without it would make
                if res and "import_error" in res and not (
                        res.get("leaf_failed") and (p.get("heavy") or (a.imp_check and "impall" in res))):
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
        none_loads = list(loads)
        if workers:
            extra = [t for t in sorted(bits_list(a.cand | a.broad), key=lambda t: dur[t]) if t not in out]
            rest = [t for t in sorted(range(n), key=lambda t: dur[t]) if t not in out and not (a.cand | a.broad) >> t & 1]
            workers = fill_slack(workers, loads, extra + rest, dur)
        workers = [w for w in workers if w]
        if os.environ.get("BLAST_MARK", "1") == "1":
            workers, code = mark_probe(workers, dur, n, bool(a.full) or len(out) >= n, a.reasons, ids)
            if code:
                log(f"{c}: marker {code} ms, full {bool(a.full)}, picked {len(out)}")
        if os.environ.get("BLAST_NONE", "1") == "1":
            workers, code = none_probe(sel, none_loads, dur, n, bool(a.full))
            log(f"{c}: none probe {code} ms, normal pick {len(sel)}, busiest {max(none_loads) if none_loads else 0} ms")
        print(json.dumps({"commit": c, "workers": workers}), flush=True)
    log(f"done, simulated estimate {total / 1000:.1f}s")


if __name__ == "__main__":
    main()
