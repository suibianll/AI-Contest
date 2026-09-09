"""Focused static and small-tensor verification for the A-C76.5 candidate."""

from pathlib import Path
import importlib.util
import json
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidate" / "solution.py"


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    candidate = load_solution(CANDIDATE, "attention_c765_residual_directed")
    required = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    )
    missing = [name for name in required if not hasattr(candidate, name)]
    if missing:
        raise AssertionError(f"missing public API: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(925)

    coupling = torch.randn(2, 64, 64, generator=gen).to(device)
    coupling = coupling + coupling.transpose(-1, -2)
    signs = candidate._c765_signs_from_coupling(coupling, 16)
    if tuple(signs.shape) != (2, 64):
        raise AssertionError(f"unexpected sign shape: {tuple(signs.shape)}")
    if not bool(((signs == 1) | (signs == -1)).all()):
        raise AssertionError("residual signs are not +/-1")
    seeds = [
        candidate._attention_rotation_signs(2, 64, int(seed))
        for seed in candidate._ATTN_ROTATION_SEEDS
    ]
    if any(bool(torch.equal(signs, seed_signs)) for seed_signs in seeds):
        raise AssertionError("residual signs duplicate an existing seed pattern")

    q = torch.randn(7, 2 * 64, generator=gen).to(device)
    k = torch.randn(7, 2 * 64, generator=gen).to(device)
    q_rot = candidate._apply_attention_rotation(q, 2, 64, signs, 16)
    k_rot = candidate._apply_attention_rotation(k, 2, 64, signs, 16)
    qh = q.reshape(7, 2, 64)
    kh = k.reshape(7, 2, 64)
    qrh = q_rot.reshape(7, 2, 64)
    krh = k_rot.reshape(7, 2, 64)
    ip_ref = torch.einsum("thk,thk->th", qh, kh)
    ip_rot = torch.einsum("thk,thk->th", qrh, krh)
    ip_err = float((ip_ref - ip_rot).abs().max())
    if ip_err > 1e-4:
        raise AssertionError(f"QK inner product not preserved: {ip_err}")

    generator = torch.Generator(device=device).manual_seed(226)
    q_quant = torch.randn((20, 128), generator=generator, device=device)
    q_scale = torch.ones((20, 8), dtype=torch.float32, device=device)
    k_quant = torch.randn((20, 64), generator=generator, device=device)
    k_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    v_quant = torch.randn((20, 64), generator=generator, device=device)
    v_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    windows = [
        {
            "q": (q_quant + 0.01 * i, q_scale),
            "k": (k_quant - 0.01 * i, k_scale),
            "v": (v_quant, v_scale),
        }
        for i in range(3)
    ]
    states = candidate.hif4_calibration_attention(windows, 2, 1, 64)
    if set(states) != {"q_state", "k_state", "v_state"}:
        raise AssertionError(f"unexpected Attention state keys: {set(states)}")
    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "c765_reference")
    reference.validate_state(states)
    for api_name, pair, heads, state_name in (
        ("hif4_dynamic_quantize_q", windows[0]["q"], 2, "q_state"),
        ("hif4_dynamic_quantize_k", windows[0]["k"], 1, "k_state"),
        ("hif4_dynamic_quantize_v", windows[0]["v"], 1, "v_state"),
    ):
        params = getattr(candidate, api_name)(
            pair[0], pair[1], heads, 64, states[state_name]
        )
        dense = candidate._dequantize_hif4(params)
        if not bool(torch.isfinite(dense).all()):
            raise AssertionError(f"{api_name} produced non-finite output")

    report = {
        "status": "passed",
        "device": device.type,
        "c765_blocks": list(candidate._C765_BLOCKS),
        "residual_signs_shape": list(signs.shape),
        "qk_inner_product_max_error": ip_err,
        "checks": {
            "six_api_import": True,
            "residual_signs_valid": True,
            "residual_signs_not_seed_duplicate": True,
            "qk_inner_product_preserved": True,
            "legal_attention_state": True,
            "finite_outputs": True,
        },
    }
    (HERE / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
