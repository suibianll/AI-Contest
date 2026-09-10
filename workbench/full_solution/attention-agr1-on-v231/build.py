"""Rebuild v234's A-GR1 block on the v231 archive (idempotent, byte-level)."""

import importlib.util
import pathlib

WB = pathlib.Path(__file__).resolve().parent
CAND = WB / "candidate" / "solution.py"
AGR1_WB = WB.parent / "attention-agr1-general-reciprocal"

spec = importlib.util.spec_from_file_location("agr1_block", AGR1_WB / "agr1_block.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

data = CAND.read_bytes()
if b"A-GR1 (2026-09-10)" in data:
    print("already appended")
else:
    block = mod.AGR1_BLOCK.replace("\r\n", "\n").encode("utf-8")
    CAND.write_bytes(data.rstrip(b"\n") + b"\n" + block)
    print("appended A-GR1 block onto v231 parent")
