"""A3 runs for the deployed gate arm: six-shard ID + OOD + real-input control.

Holds the GPU lock across all steps.  Fresh default timing runs separately
(run_a3_default.py) because it uses the compatibility backend.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
sys.path.insert(0, str(ROOT / "evaluator"))
import gpu_lock  # noqa: E402

ZERO = ROOT / "solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py"
CANDIDATE = ROOT / "workbench/v162_attention/candidate/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"
PY = ROOT / ".venv/Scripts/python.exe"


def run_eval(name: str, ood: bool) -> int:
    out_dir = ROOT / f"artifacts/proxy_v3/v162-independent/attention/{name}"
    command = [
        str(PY), str(ROOT / "evaluator/eval.py"),
        "--baseline-solution", str(ZERO),
        "--solution", str(CANDIDATE),
        "--name", name,
        "--attention-only",
        "--shards", "0,1,2,3,4,5",
        "--cache", str(CACHE),
        "--algorithm-device", "cuda",
        "--calibration-cache-mode", "auto",
        "--stop-after-nonpositive", "7",
        "--reuse-existing",
        "--output-dir", str(out_dir),
    ]
    if ood:
        command.append("--ood")
    print(f"=== {name} ===", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    print(f"=== {name} finished in {time.perf_counter() - started:.0f}s rc={completed.returncode} ===", flush=True)
    return completed.returncode


def real_input_control() -> int:
    """Bitwise Linear/V control on real cache tensors (plan §5 contract)."""

    import importlib.util

    def load(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    baseline = load(ZERO, "a3_control_baseline")
    candidate = load(CANDIDATE, "a3_control_candidate")
    import reference_hif4 as ref
    import official_eval as v2

    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    checked = 0
    for layer in (0, 8, 15, 23):
        for role in ("q", "k", "v", "o", "fc_gate", "fc_up", "proj"):
            w_quant, w_scale = v2._pair(pack["weights"][layer][role])
            base = baseline.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
            cand = candidate.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
            for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
                if not torch.equal(base["weight_params"][key], cand["weight_params"][key]):
                    print(f"CONTROL FAIL weight layer {layer} role {role} field {key}")
                    return 1
            assert cand["activation_state"] == {}
            a_quant, a_scale = v2._pair(pack["calibration_activations"][role][0][layer])
            base_a = baseline.hif4_dynamic_quantize_activation(a_quant, a_scale, {})
            cand_a = candidate.hif4_dynamic_quantize_activation(a_quant, a_scale, {})
            for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
                if not torch.equal(base_a[key], cand_a[key]):
                    print(f"CONTROL FAIL activation layer {layer} role {role} field {key}")
                    return 1
            checked += 2
        q, k, v = pack["calibration_qkv"][0][layer]
        v_quant, v_scale = v2._pair(v)
        base_v = baseline.hif4_dynamic_quantize_v(v_quant, v_scale, pack["kv_heads"], pack["head_dim"], {})
        cand_v = candidate.hif4_dynamic_quantize_v(v_quant, v_scale, pack["kv_heads"], pack["head_dim"], {})
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            if not torch.equal(base_v[key], cand_v[key]):
                print(f"CONTROL FAIL dynamic V layer {layer}")
                return 1
        checked += 1
    print(f"REAL-INPUT CONTROL PASS ({checked} comparisons, Linear/dyn-act/dyn-V bitwise identical)")
    return 0


def standalone_import_check() -> int:
    """Copy the candidate outside the repo tree and import + call all six APIs."""

    import importlib.util
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "standalone_solution.py"
        target.write_text(CANDIDATE.read_text(encoding="utf-8"), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("standalone_solution", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        w = torch.randn(64, 128)
        result = module.hif4_calibration_and_quantize_weight(w, torch.ones(64, 8), [])
        states = module.hif4_calibration_attention(
            [{"q": (torch.randn(8, 896), torch.ones(8, 56)),
              "k": (torch.randn(8, 128), torch.ones(8, 8)),
              "v": (torch.randn(8, 128), torch.ones(8, 8))} for _ in range(3)],
            14, 2, 64,
        )
        q_params = module.hif4_dynamic_quantize_q(
            torch.randn(8, 896), torch.ones(8, 56), 14, 64, states["q_state"]
        )
        k_params = module.hif4_dynamic_quantize_k(
            torch.randn(8, 128), torch.ones(8, 8), 2, 64, states["k_state"]
        )
        v_params = module.hif4_dynamic_quantize_v(
            torch.randn(8, 128), torch.ones(8, 8), 2, 64, states["v_state"]
        )
        a_params = module.hif4_dynamic_quantize_activation(
            torch.randn(8, 128), torch.ones(8, 8), {}
        )
        count = sum(1 for item in (result, states, q_params, k_params, v_params, a_params) if item)
        print(f"STANDALONE IMPORT PASS ({count}/6 API groups returned)")
        return 0


def main() -> int:
    skip_evals = "--control-only" in sys.argv
    code = gpu_lock.acquire("A", "a3-id-ood-control")
    if code != 0:
        return code
    try:
        failures = []
        if not skip_evals:
            for name, ood in (("a3-id", False), ("a3-ood", True)):
                if run_eval(name, ood) != 0:
                    failures.append(name)
        if real_input_control() != 0:
            failures.append("control")
        if standalone_import_check() != 0:
            failures.append("standalone")
    finally:
        gpu_lock.release("A", "a3-id-ood-control")
    if failures:
        print(f"A3 FAILURES: {failures}", flush=True)
        return 1
    print("A3 ID/OOD/CONTROL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
