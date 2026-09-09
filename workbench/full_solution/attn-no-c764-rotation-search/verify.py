"""Small-scale verification for v205 attn-no-c764-rotation-search.

Runs ONE real attention layer (layer 0 of the 4B proxy-v2 cache) through
``hif4_calibration_attention`` for both the root solution and the v205
candidate, then checks:

1. candidate flag is off and the module imports standalone (six APIs).
2. non-no-op evidence: candidate ``_attention_deployed_mse`` call count is
   lower than root by the C76.4 search budget (3 blocks x 4 seeds = 12
   candidate evaluations, plus one base evaluation when the A2 arm did not
   already populate ``base_causal``).
3. legality: ``validate_state`` on q/k/v states (reference_hif4).
4. dynamic APIs: one calibration window through quantize q/k/v, params pass
   ``validate_hif4_params`` and decode to finite tensors.

GPU: one layer only (~6s per calibration on the shared 3060 Ti).
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import reference_hif4 as ref  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def move_pair(pair):
    return (pair[0].to(DEVICE), pair[1].to(DEVICE))


def run_layer(module, calib, q_heads, kv_heads, head_dim):
    calls = {"n": 0}
    original = module._attention_deployed_mse

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    module._attention_deployed_mse = counting
    try:
        t0 = time.perf_counter()
        states = module.hif4_calibration_attention(
            calib, q_heads, kv_heads, head_dim
        )
        wall = time.perf_counter() - t0
    finally:
        module._attention_deployed_mse = original
    return states, calls["n"], wall


def check_states(module, states, calib_window, q_heads, kv_heads, head_dim):
    for name in ("q_state", "k_state", "v_state"):
        ref.validate_state(states[name])

    q_params = module.hif4_dynamic_quantize_q(
        *calib_window["q"], q_heads, head_dim, states["q_state"]
    )
    k_params = module.hif4_dynamic_quantize_k(
        *calib_window["k"], kv_heads, head_dim, states["k_state"]
    )
    v_params = module.hif4_dynamic_quantize_v(
        *calib_window["v"], kv_heads, head_dim, states["v_state"]
    )
    dense_q = v2.dequantize_nvfp4(*calib_window["q"])
    dense_k = v2.dequantize_nvfp4(*calib_window["k"])
    dense_v = v2.dequantize_nvfp4(*calib_window["v"])
    out = {}
    for label, params, dense in (
        ("q", q_params, dense_q),
        ("k", k_params, dense_k),
        ("v", v_params, dense_v),
    ):
        ref.validate_hif4_params(params, dense.shape)
        decoded = module._dequantize_hif4(params)
        if not bool(torch.isfinite(decoded).all()):
            raise RuntimeError(f"{label} decode is not finite")
        out[label] = float(decoded.square().mean())
    return out


def main():
    root_sol = load_module("v205_root", ROOT / "solution.py")
    cand_sol = load_module("v205_candidate", HERE / "candidate" / "solution.py")

    apis = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    )
    for name in apis:
        assert callable(getattr(cand_sol, name, None)), f"missing API {name}"
    assert cand_sol._ATTN_ROTATION_ENABLED is False
    assert root_sol._ATTN_ROTATION_ENABLED is True

    raw = v2.load_pack(
        ROOT / "artifacts" / "official_eval" / "cache" / "qwen3.5-4b-proxy-v2.pt"
    )
    attn_layers = sorted(int(x) for x in raw.metadata.get("attention_layers"))
    layer = attn_layers[0]
    calib = []
    for sample in range(len(raw.calibration_windows)):
        q, k, v = raw.calibration_qkv[sample][layer]
        calib.append(
            {
                "q": move_pair(v2._pair(q)),
                "k": move_pair(v2._pair(k)),
                "v": move_pair(v2._pair(v)),
            }
        )
    print(
        f"layer={layer} windows={len(calib)} q_heads={raw.q_heads} "
        f"kv_heads={raw.kv_heads} head_dim={raw.head_dim} device={DEVICE}"
    )

    root_states, root_calls, root_wall = run_layer(
        root_sol, calib, raw.q_heads, raw.kv_heads, raw.head_dim
    )
    cand_states, cand_calls, cand_wall = run_layer(
        cand_sol, calib, raw.q_heads, raw.kv_heads, raw.head_dim
    )
    print(f"root: deployed_mse calls={root_calls} wall={root_wall:.2f}s")
    print(f"cand: deployed_mse calls={cand_calls} wall={cand_wall:.2f}s")

    root_stats = check_states(
        root_sol, root_states, calib[0], raw.q_heads, raw.kv_heads, raw.head_dim
    )
    cand_stats = check_states(
        cand_sol, cand_states, calib[0], raw.q_heads, raw.kv_heads, raw.head_dim
    )

    rotation_keys = ("rotation", "rotation_block")
    root_rot = {
        n: {k: root_states[n].get(k) for k in rotation_keys if k in root_states[n]}
        for n in ("q_state", "k_state")
    }
    cand_rot = {
        n: {k: cand_states[n].get(k) for k in rotation_keys if k in cand_states[n]}
        for n in ("q_state", "k_state")
    }
    report = {
        "layer": layer,
        "root_deployed_mse_calls": root_calls,
        "cand_deployed_mse_calls": cand_calls,
        "call_drop": root_calls - cand_calls,
        "root_wall_seconds": root_wall,
        "cand_wall_seconds": cand_wall,
        "root_rotation_keys": {k: sorted(v) for k, v in root_rot.items()},
        "cand_rotation_keys": {k: sorted(v) for k, v in cand_rot.items()},
        "state_validation": "ok",
        "dynamic_param_validation": "ok",
        "root_decode_power": root_stats,
        "cand_decode_power": cand_stats,
    }
    out = HERE / "verify-report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    expected_drop = 12  # H16/H32/H64 all divide head_dim=256; 4 seeds each
    if not report["call_drop"] >= expected_drop:
        raise RuntimeError(
            f"call drop {report['call_drop']} < expected {expected_drop}: "
            "C76.4 search may still be running"
        )
    print("VERIFY OK")


if __name__ == "__main__":
    main()
