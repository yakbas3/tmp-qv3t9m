"""First body line of a code object, for sys.monitoring PY_START.

A LINE event fires only when the line number changes. A function whose body
starts on its own header line (a one-line lambda such as
`exp.If: {"annotator": lambda self, e: ...}`, or `def f(x): return x`) starts
on the line its RESUME sits on, so no LINE event ever reports that line.
PY_START fires on every call; the line of the first real instruction is the
line LINE missed. Used only by the map plugins at build time.
"""

from __future__ import annotations

import dis

_SKIP = {"RESUME", "RETURN_GENERATOR", "POP_TOP", "COPY_FREE_VARS", "MAKE_CELL", "NOP", "CACHE"}
_cache: dict = {}


def first_line(code):
    key = id(code)
    hit = _cache.get(key)
    if hit is not None and hit[0] is code:
        return hit[1]
    line = None
    try:
        for ins in dis.get_instructions(code):
            if ins.opname in _SKIP:
                continue
            ln = ins.positions.lineno if ins.positions else None
            if ln is not None:
                line = ln
                break
    except Exception:
        line = None
    _cache[key] = (code, line)
    return line
