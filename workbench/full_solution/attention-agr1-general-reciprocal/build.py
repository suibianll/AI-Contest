"""Append the A-GR1 block to the root copy (idempotent build, byte-level)."""

import importlib.util
import pathlib

WB = pathlib.Path(__file__).resolve().parent
CAND = WB / "candidate" / "solution.py"
ROOT = WB.parents[2] / "solution.py"

spec = importlib.util.spec_from_file_location("agr1_block", WB / "agr1_block.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

data = CAND.read_bytes()
if b"A-GR1 (2026-09-10)" in data:
    print("already appended")
else:
    block = mod.AGR1_BLOCK.replace("\r\n", "\n").encode("utf-8")
    CAND.write_bytes(data.rstrip(b"\n") + b"\n" + block)
    print("appended")
