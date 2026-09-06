"""Official-contract fuzz for the rotation candidates (v107 WA discipline).

The official verdict for A2/R2 was attention `wrong answer` while R1 passed.
Per the v100/v107 post-mortem, WA means an uncaught runtime exception in any
case; the official surface includes unseen dynamic seq_lens, alternate head
geometries, inference_mode/no_grad harnesses, CPU-only execution, and
per-sample variable lengths.  This script hunts for any exception on that
surface.  Exit code 1 lists the repro cases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref  # noqa: E402

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pair(tokens: int, channels: int, seed: int, extreme: bool = False):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(tokens, channels, generator=generator)
    if extreme:
        dense = dense * 330000.0
    blocks = channels // 16
    grouped = dense.unflatten(-1, (blocks, 16))
    scale = (grouped.abs().amax(-1) / 6.0).clamp(min=1e-6, max=1e6)
    normalized = grouped / scale[..., None]
    sign = normalized.sign()
    index = torch.argmin((normalized.abs()[..., None] - FP4_GRID).abs(), dim=-1)
    quant = (sign * FP4_GRID[index]).flatten(-2, -1)
    return quant, scale


def windows_for(lengths, q_heads, kv_heads, head_dim, seed=5, mixed_q=False):
    items = []
    for i, length in enumerate(lengths):
        q_len = max(1, length // 2) if mixed_q else length
        items.append({
            "q": make_pair(q_len, q_heads * head_dim, seed * 100 + i),
            "k": make_pair(length, kv_heads * head_dim, seed * 200 + i),
            "v": make_pair(length, kv_heads * head_dim, seed * 300 + i),
        })
    return items


GEOMETRIES = [
    ("qwen", 14, 2, 64),
    ("mha1", 1, 1, 64),
    ("mha12", 12, 12, 64),
    ("gqa32x8", 32, 8, 128),
    ("nonpow2dim", 16, 16, 80),
]

DYNAMIC_LENGTHS = [1, 2, 7, 100, 2048, 5000]


def run_candidate(tag: str, path: Path) -> list[str]:
    failures = []
    module = load(path, f"fuzz_{tag}")

    for geo_name, q_heads, kv_heads, head_dim in GEOMETRIES:
        label = f"{tag}/{geo_name}"
        try:
            states = module.hif4_calibration_attention(
                windows_for([10, 128, 512, 1024, 1024], q_heads, kv_heads, head_dim),
                q_heads, kv_heads, head_dim,
            )
            for name in ("q_state", "k_state", "v_state"):
                ref.validate_state(states[name])
        except Exception as error:  # noqa: BLE001 - hunting WA causes
            failures.append(f"{label} calibration: {type(error).__name__}: {error}")
            continue

        for dyn_len in DYNAMIC_LENGTHS:
            for api, heads, state_name in (
                ("q", q_heads, "q_state"), ("k", kv_heads, "k_state"), ("v", kv_heads, "v_state"),
            ):
                quant, scale = make_pair(dyn_len, heads * head_dim, 900 + dyn_len)
                try:
                    params = getattr(module, f"hif4_dynamic_quantize_{api}")(
                        quant, scale, heads, head_dim, states[state_name]
                    )
                    ref.validate_hif4_params(params, quant.shape)
                except Exception as error:  # noqa: BLE001
                    failures.append(
                        f"{label} dynamic {api} len={dyn_len}: {type(error).__name__}: {error}"
                    )

    # inference_mode + no_grad harness
    for mode_name, ctx in (("inference_mode", torch.inference_mode()), ("no_grad", torch.no_grad())):
        try:
            with ctx:
                states = module.hif4_calibration_attention(
                    windows_for([10, 128], 14, 2, 64), 14, 2, 64
                )
                for name in ("q_state", "k_state", "v_state"):
                    ref.validate_state(states[name])
                quant, scale = make_pair(16, 896, 77)
                params = module.hif4_dynamic_quantize_q(quant, scale, 14, 64, states["q_state"])
                ref.validate_hif4_params(params, quant.shape)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{tag}/{mode_name}: {type(error).__name__}: {error}")

    # mixed per-sample Q length (L_q != L_k)
    try:
        states = module.hif4_calibration_attention(
            windows_for([32, 48], 14, 2, 64, mixed_q=True), 14, 2, 64
        )
        ref.validate_state(states["q_state"])
    except Exception as error:  # noqa: BLE001
        failures.append(f"{tag}/mixed-q-calibration: {type(error).__name__}: {error}")

    # extreme values
    try:
        states = module.hif4_calibration_attention(
            windows_for([128, 128], 14, 2, 64, seed=13), 14, 2, 64
        )
        quant, scale = make_pair(64, 896, 55, extreme=True)
        params = module.hif4_dynamic_quantize_q(quant, scale, 14, 64, states["q_state"])
        ref.validate_hif4_params(params, quant.shape)
    except Exception as error:  # noqa: BLE001
        failures.append(f"{tag}/extreme: {type(error).__name__}: {error}")

    # repeat calibration for determinism of the API contract (same inputs twice)
    try:
        wins = windows_for([10, 128, 512, 1024, 1024], 14, 2, 64, seed=21)
        s1 = module.hif4_calibration_attention(wins, 14, 2, 64)
        s2 = module.hif4_calibration_attention(wins, 14, 2, 64)
        r1 = s1["q_state"].get("r")
        r2 = s2["q_state"].get("r")
        if (r1 is None) != (r2 is None):
            failures.append(f"{tag}/repeat: deployment decision differs between identical runs")
        elif r1 is not None and not torch.equal(r1, r2):
            failures.append(f"{tag}/repeat: trained R not reproducible run-to-run (max diff "
                            f"{float((r1 - r2).abs().max()):.3e})")
    except Exception as error:  # noqa: BLE001
        failures.append(f"{tag}/repeat: {type(error).__name__}: {error}")

    return failures


def main() -> int:
    targets = [
        ("a2", ROOT / "workbench/v162_attention/candidate/solution.py"),
        ("r2", ROOT / "workbench/v162_attention/candidate_v3/solution.py"),
        ("a2b", ROOT / "workbench/v162_attention/candidate_b/solution.py"),
        ("r2b", ROOT / "workbench/v162_attention/candidate_v3b/solution.py"),
        ("r1", ROOT / "workbench/v162_attention/candidate_v2/solution.py"),
    ]
    all_failures: list[str] = []
    for tag, path in targets:
        if not path.exists():
            print(f"skip {tag}: {path} missing")
            continue
        failures = run_candidate(tag, path)
        if failures:
            print(f"--- {tag}: {len(failures)} failure(s)")
            for item in failures:
                print("   ", item)
        else:
            print(f"--- {tag}: CLEAN")
        all_failures.extend(failures)
    if all_failures:
        print(f"FUZZ TOTAL: {len(all_failures)} failure(s)")
        return 1
    print("FUZZ TOTAL: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
