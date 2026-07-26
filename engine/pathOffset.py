"""
pathOffset.py (compatibility shim)
--------------------------------------------------------------------------
pathOptimizstion.py does `from engine import pathOffset as offset` and then
reads `offset.offset_paths` at import time. That original pathOffset.py
module no longer exists -- sara's groove_offsetting.py replaced it.

This file does NOT reimplement any offsetting logic. It only provides the
one attribute pathOptimizstion.py expects. The real caller (main.py /
run_advanced_optimizer.py) must set `pathOffset.offset_paths = <raw rings
from generate_groove_offset_paths>` BEFORE importing engine.pathOptimizstion,
since pathOptimizstion.py reads it immediately at import time (module-level
code, not inside a function).
--------------------------------------------------------------------------
"""

offset_paths = []