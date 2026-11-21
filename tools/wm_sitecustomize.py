# Records, in every multiprocessing child, the sqlglot modules loaded when it exits.
import os
def _dump():
    import sys, json, multiprocessing
    if multiprocessing.parent_process() is None:
        return
    out = os.environ.get("WM_OUT", "/out")
    mods = sorted(m for m in sys.modules if m == "sqlglot" or m.startswith("sqlglot."))
    with open(f"{out}/w{os.getpid()}.json", "w") as f:
        json.dump(mods, f)
try:
    import multiprocessing.util as _u
    _u.Finalize(None, _dump, exitpriority=1000)
except Exception:
    pass
