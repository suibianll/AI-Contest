"""Append the A-GR2 block to the v231-archive copy (idempotent, byte-level)."""

import importlib.util
import pathlib

WB = pathlib.Path(__file__).resolve().parent
CAND = WB / "candidate" / "solution.py"

spec = importlib.util.spec_from_file_location("agr2_block", WB / "agr2_block.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

data = CAND.read_bytes()
if b"A-GR2 (2026-09-10)" in data:
    print("already appended")
else:
    block = mod.AGR2_BLOCK.replace("\r\n", "\n").encode("utf-8")
    CAND.write_bytes(data.rstrip(b"\n") + b"\n" + block)
    print("appended")
