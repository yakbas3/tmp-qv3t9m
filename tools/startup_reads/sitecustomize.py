"""Audit hook installed before pytest starts (tools/startup_reads.sh).

Records, for one pytest --collect-only session, every path under the repo
root that is opened or listed, and every module under tests/ that ends up
imported. Written to $STARTUP_OUT at exit."""
import atexit
import json
import os
import sys

ROOT = os.path.realpath(os.environ.get("STARTUP_ROOT", "/opt/sqlglot")) + os.sep
_opened = set()
_listed = set()


def _rel(p):
    try:
        p = os.fsdecode(p if p is not None else ".")
    except Exception:
        return None
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    p = os.path.normpath(p)
    if p + os.sep == ROOT:
        return "."
    return p[len(ROOT):] if p.startswith(ROOT) else None


def _hook(event, args):
    if event == "open":
        if isinstance(args[0], (str, bytes, os.PathLike)):
            r = _rel(args[0])
            if r and "__pycache__" not in r:
                _opened.add(r)
    elif event in ("os.listdir", "os.scandir", "glob.glob", "os.walk"):
        r = _rel(args[0] if args else ".")
        if r:
            _listed.add(r)


sys.addaudithook(_hook)


@atexit.register
def _dump():
    mods = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        r = _rel(f) if f else None
        if r and r.startswith("tests/"):
            mods.add(r)
    out = os.environ.get("STARTUP_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump({"opened": sorted(_opened), "listed": sorted(_listed),
                       "tests_modules": sorted(mods)}, fh, indent=1)
