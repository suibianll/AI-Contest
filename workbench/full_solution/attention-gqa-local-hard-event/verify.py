"""Focused static and small-tensor verification for the A-H3 group event search."""

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
    candidate = load_solution(CANDIDATE, "attention_gqa_local_hard_event")
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
    gen = torch.Generator(device="cpu").manual_seed(917)

    skew = torch.randn(2, 8, 8, generator=gen).to(device)
    skew = skew - skew.transpose(-1, -2)
    eps = 1e-3
    d_cayley = (candidate._ah3_cayley(eps * skew) - candidate._ah3_cayley(-eps * skew)) / (
        2.0 * eps
    )
    rel = float((d_cayley + 2.0 * skew).norm() / (2.0 * skew).norm().clamp_min(1e-12))
    if rel > 1e-4:
        raise AssertionError(f"Cayley derivative mismatch: rel={rel}")
    eye8 = torch.eye(8, device=device)
    ortho_err = float(
        (candidate._ah3_cayley(skew) @ candidate._ah3_cayley(skew).transpose(-1, -2) - eye8)
        .abs().max()
    )
    if ortho_err > 1e-5:
        raise AssertionError(f"Cayley transform not orthogonal: {ortho_err}")

    x0 = torch.tensor([1.0, -1.0, 0.0], device=device)
    dx = torch.tensor([-0.5, 0.5, 0.25], device=device)
    denom = torch.ones(3, device=device)
    t = candidate._ah3_boundary_times(x0, dx, denom)
    expected = sorted(
        [2.0 * (1.0 - b) for b in (0.125, 0.375, 0.625, 0.875)]
        + [2.0 * (1.0 - b) for b in (0.125, 0.375, 0.625, 0.875)]
        + [b / 0.25 for b in (0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625)]
    )
    got = sorted(float(v) for v in t)
    if len(got) != len(expected):
        raise AssertionError(f"boundary count mismatch: {len(got)} != {len(expected)}")
    for g, e in zip(got, expected):
        if abs(g - e) > 1e-5:
            raise AssertionError(f"boundary value mismatch: {g} != {e}")
    if min(got) <= 0.0:
        raise AssertionError("non-positive boundary time leaked")

    generator = torch.Generator(device=device).manual_seed(223)
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
    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "ah3_reference")
    reference.validate_state(states)
    status = states["q_state"].get("ah3_status")
    if status not in {"accepted", "no-improvement", "unavailable-parent"}:
        raise AssertionError(f"unexpected A-H3 status: {status}")
    if int(states["q_state"].get("ah3_attempted", -1)) != int(
        states["q_state"].get("ah3_groups", -2)
    ):
        raise AssertionError("attempted group count does not match group count")
    if int(states["q_state"].get("ah3_t0_identical", 0)) != 1:
        raise AssertionError("t=0 did not reproduce the deployed parent")
    if len(states["q_state"].get("ah3_group_accepted", [])) != int(
        states["q_state"].get("ah3_groups", 0)
    ):
        raise AssertionError("per-group record length mismatch")

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

    q_ref = candidate._dequantize_nvfp4_float32(*windows[0]["q"]).to(torch.float32)
    k_ref = candidate._dequantize_nvfp4_float32(*windows[0]["k"]).to(torch.float32)
    theta_s = torch.randn(1, 64, 64, generator=gen).to(device) * 0.2
    c_s, _ = candidate._m_cayley_pair(theta_s)
    base = candidate._a2_hadamard_orthogonal(64).to(device)
    rotation_s = torch.einsum("dk,gkl->gdl", base, c_s)
    center_s = torch.randn(1, 64, generator=gen).to(device) * 0.1
    q_state_r = dict(
        states["q_state"],
        learned_rotation=rotation_s.detach().cpu().to(torch.float32),
    )
    k_state_r = dict(
        states["k_state"],
        learned_rotation=rotation_s.detach().cpu().to(torch.float32),
        learned_center=center_s.detach().cpu().to(torch.float32),
    )
    x_q = candidate._a2_apply_group_rotation(
        candidate._ah3_pretransform_dense(q_ref, q_state_r, 2, False), 2, rotation_s
    )
    x_k = (
        candidate._a2_apply_group_rotation(
            candidate._ah3_pretransform_dense(k_ref, k_state_r, 1, True),
            1,
            rotation_s,
        ).reshape(-1, 1, 64)
        + center_s.to(device)[None]
    ).reshape(k_ref.shape)
    for name, x, pair, heads, state in (
        ("q", x_q, windows[0]["q"], 2, q_state_r),
        ("k", x_k, windows[0]["k"], 1, k_state_r),
    ):
        api = (
            candidate.hif4_dynamic_quantize_q
            if name == "q"
            else candidate.hif4_dynamic_quantize_k
        )
        deployed = api(pair[0], pair[1], heads, 64, state)
        direct = candidate._dense_to_hif4(
            x,
            importance=state["importance"],
            search_offsets=state["offsets"],
            error_threshold=float(state["error_threshold"]),
            accept_margin=float(state["accept_margin"]),
            max_refine_ratio=float(state["max_refine_ratio"]),
            max_refine_blocks=int(state["max_refine_blocks"]),
        )
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            if not bool(torch.equal(deployed[key], direct[key])):
                raise AssertionError(f"pretransform/{name} field {key} mismatch")

    report = {
        "status": "passed",
        "device": device.type,
        "a2_arm": states["q_state"].get("a2_arm"),
        "ah3_parent_arm": states["q_state"].get("ah3_parent_arm"),
        "ah3_status": status,
        "ah3_t0_identical": states["q_state"].get("ah3_t0_identical"),
        "ah3_groups": states["q_state"].get("ah3_groups"),
        "ah3_accepted": states["q_state"].get("ah3_accepted"),
        "ah3_group_accepted": states["q_state"].get("ah3_group_accepted"),
        "cayley_derivative_rel_error": rel,
        "cayley_orthogonality_error": ortho_err,
        "checks": {
            "six_api_import": True,
            "cayley_derivative_minus_2s": True,
            "cayley_orthogonal": True,
            "boundary_times_known_example": True,
            "legal_attention_state": True,
            "finite_outputs": True,
            "t0_matches_deployed_parent": True,
            "pretransform_matches_deployed_encode": True,
            "per_group_record_length": True,
        },
    }
    (HERE / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
