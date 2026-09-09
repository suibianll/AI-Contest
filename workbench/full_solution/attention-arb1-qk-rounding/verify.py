"""Focused A-RB1 legality, parent-control and reachability check."""

from pathlib import Path
import importlib.util
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_params(params: dict[str, torch.Tensor], rows: int, channels: int) -> None:
    expected = {
        "scale_factor": (rows, channels // 64, 1, 1, 1),
        "scale_lv2": (rows, channels // 64, 8, 1, 1),
        "scale_lv3": (rows, channels // 64, 8, 2, 1),
        "sign": (rows, channels // 64, 8, 2, 4),
        "mant": (rows, channels // 64, 8, 2, 4),
    }
    if set(params) != set(expected):
        raise AssertionError(f"unexpected HiF4 keys: {set(params)}")
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(params[key].shape)} != {shape}")


def same_params(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(
        bool(torch.equal(a[key], b[key])) for key in a
    )


def expected_switch_count(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    boundaries: torch.Tensor,
) -> int:
    """Analytic count of code-changing elements for a fixed-scale encode."""

    rows, channels = map(int, dense.shape)
    blocks = channels // 64
    scale = params["scale_factor"].to(torch.float32).reshape(rows, blocks, 1, 1, 1)
    lv2 = params["scale_lv2"].to(torch.float32).reshape(rows, blocks, 8, 1, 1)
    lv3 = params["scale_lv3"].to(torch.float32).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(rows, channels)
    code = torch.round(
        params["mant"].to(torch.float32).reshape(rows, channels) / 0.25
    ).clamp_(0.0, 7.0)
    u = 4.0 * dense.abs() / denominator.clamp_min(1.0e-12)
    floor_code = torch.floor(u)
    frac = u - floor_code
    changed = torch.zeros_like(code, dtype=torch.bool)
    for lower in range(1, 7):
        tau = float(boundaries[lower])
        mask = (floor_code == float(lower)) & (frac >= tau)
        if bool(mask.any()):
            new_code = torch.where(
                mask, torch.full_like(code, float(lower) + 1.0), code
            )
            changed |= mask & (new_code != code)
    return int(changed.sum())


def main() -> None:
    candidate = load_solution(HERE / "candidate" / "solution.py", "arb1_candidate")
    parent = load_solution(ROOT / "solution.py", "arb1_parent")
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
    generator = torch.Generator(device=device).manual_seed(227)
    q_heads, kv_heads, head_dim = 8, 8, 8
    q_channels, kv_channels = q_heads * head_dim, kv_heads * head_dim

    def make_pair(tokens: int, channels: int):
        return (
            torch.randn((tokens, channels), generator=generator, device=device),
            torch.ones((tokens, channels // 16), dtype=torch.float32, device=device),
        )

    calib = [
        {"q": make_pair(6, q_channels), "k": make_pair(6, kv_channels),
         "v": make_pair(6, kv_channels)},
        {"q": make_pair(5, q_channels), "k": make_pair(5, kv_channels),
         "v": make_pair(5, kv_channels)},
    ]

    parent_states = parent.hif4_calibration_attention(
        calib, q_heads, kv_heads, head_dim
    )
    states = candidate.hif4_calibration_attention(calib, q_heads, kv_heads, head_dim)
    if set(states) != {"q_state", "k_state", "v_state"}:
        raise AssertionError(f"unexpected attention state keys: {set(states)}")
    q_state = states["q_state"]
    if q_state.get("arb1_attempted") != 1:
        raise AssertionError("A-RB1 did not reach its fixed boundary fit")
    if q_state.get("arb1_coordinate") != (
        "qk-softmax-output-shared-rounding-boundary"
    ):
        raise AssertionError("A-RB1 coordinate provenance is missing")

    q_pair = calib[0]["q"]
    k_pair = calib[0]["k"]
    parent_q = parent.hif4_dynamic_quantize_q(
        q_pair[0], q_pair[1], q_heads, head_dim, parent_states["q_state"]
    )
    parent_k = parent.hif4_dynamic_quantize_k(
        k_pair[0], k_pair[1], kv_heads, head_dim, parent_states["k_state"]
    )
    assert_params(parent_q, 6, q_channels)
    assert_params(parent_k, 6, kv_channels)

    # Control A: absent table and an explicit all-0.5 table must both restore
    # the parent five fields bit for bit on the deployed encoder.
    half = torch.full((7,), 0.5, device=device, dtype=torch.float32)
    plain_state = dict(parent_states["q_state"])
    half_state = dict(parent_states["q_state"], boundaries=half)
    cand_plain = candidate.hif4_dynamic_quantize_q(
        q_pair[0], q_pair[1], q_heads, head_dim, plain_state
    )
    cand_half = candidate.hif4_dynamic_quantize_q(
        q_pair[0], q_pair[1], q_heads, head_dim, half_state
    )
    if not same_params(cand_plain, parent_q):
        raise AssertionError("A-RB1 without a table changed the parent Q fields")
    if not same_params(cand_half, parent_q):
        raise AssertionError("A-RB1 all-0.5 table changed the parent Q fields")
    if "boundaries" in plain_state:
        raise AssertionError("control state was mutated")

    # Control B: on a fixed-scale encode the synthetic non-parent table must
    # change exactly the analytically expected eligible mantissas.
    synthetic = half.clone()
    synthetic[3] = 0.25
    probe = torch.randn((3, 64), generator=generator, device=device)
    base_fixed = candidate._dense_to_hif4(
        probe, boundaries=half, max_refine_ratio=0.0
    )
    synth_fixed = candidate._dense_to_hif4(
        probe, boundaries=synthetic, max_refine_ratio=0.0
    )
    expected = expected_switch_count(probe, base_fixed, synthetic)
    if expected == 0:
        raise AssertionError("synthetic table has no reachable eligible element")
    actual = int((synth_fixed["mant"] != base_fixed["mant"]).sum())
    if actual != expected:
        raise AssertionError(
            f"synthetic table changed {actual} mantissas != expected {expected}"
        )

    # Control C: the threshold must cross both active mantissa sites -- the
    # initial encode and the per-candidate exact hierarchy encode.
    for label, kwargs in (
        ("initial-only", dict(max_refine_ratio=0.0)),
        ("refined", dict(
            search_offsets=(0, 1), error_threshold=0.0,
            max_refine_ratio=1.0, max_refine_blocks=8,
        )),
    ):
        base = candidate._dense_to_hif4(probe, boundaries=half, **kwargs)
        moved = candidate._dense_to_hif4(probe, boundaries=synthetic, **kwargs)
        changed = int((base["mant"] != moved["mant"]).sum())
        if changed == 0:
            raise AssertionError(f"boundary did not reach the {label} mantissa site")
        print(f"  control C {label}: changed mantissas={changed}")

    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "arb1_reference")
    for name in ("q_state", "k_state", "v_state"):
        reference.validate_state(states[name])
    reference.validate_hif4_params(parent_q, (6, q_channels))
    print(
        f"A-RB1 verify passed: device={device.type} arm={q_state.get('arb1_arm')} "
        f"windows={q_state.get('arb1_windows')} "
        f"proposals={q_state.get('arb1_proposals')} "
        f"eligible={q_state.get('arb1_eligible_q')}/{q_state.get('arb1_eligible_k')} "
        f"fixed_changed={q_state.get('arb1_fixed_hierarchy_changed_q')}/"
        f"{q_state.get('arb1_fixed_hierarchy_changed_k')} "
        f"causal_delta={q_state.get('arb1_causal_delta')} "
        f"synthetic_expected={expected}"
    )


if __name__ == "__main__":
    main()
