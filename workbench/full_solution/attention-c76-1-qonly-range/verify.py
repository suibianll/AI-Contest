"""Focused v218 standalone-import and Attention state check."""

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


def main() -> None:
    candidate = load_solution(HERE / "candidate" / "solution.py", "v218_candidate")
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
    if candidate._ATTN_OUTPUT_HEADWISE_PERMUTATION is not True:
        raise AssertionError("v218 did not enable the Q-only headwise permutation arm")
    if int(candidate._ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES) != 1:
        raise AssertionError("v218 does not fix the independent candidate cap at one")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(218)
    q_quant = torch.randn((20, 128), generator=generator, device=device)
    q_scale = torch.ones((20, 8), dtype=torch.float32, device=device)
    k_quant = torch.randn((20, 64), generator=generator, device=device)
    k_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    v_quant = torch.randn((20, 64), generator=generator, device=device)
    v_scale = torch.ones((20, 4), dtype=torch.float32, device=device)
    result = candidate.hif4_calibration_attention(
        [
            {"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)},
            {"q": (q_quant, q_scale), "k": (k_quant, k_scale), "v": (v_quant, v_scale)},
        ],
        2,
        1,
        64,
    )
    if set(result) != {"q_state", "k_state", "v_state"}:
        raise AssertionError(f"unexpected Attention result keys: {set(result)}")
    q_state = result["q_state"]
    k_state = result["k_state"]
    v_state = result["v_state"]
    if q_state.get("num_heads") != 2 or k_state.get("num_heads") != 1:
        raise AssertionError("v218 Attention head metadata is invalid")

    q_params = candidate.hif4_dynamic_quantize_q(q_quant, q_scale, 2, 64, q_state)
    k_params = candidate.hif4_dynamic_quantize_k(k_quant, k_scale, 1, 64, k_state)
    v_params = candidate.hif4_dynamic_quantize_v(v_quant, v_scale, 1, 64, v_state)
    assert_params(q_params, 20, 128)
    assert_params(k_params, 20, 64)
    assert_params(v_params, 20, 64)

    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "v218_reference")
    reference.validate_state(result)
    reference.validate_hif4_params(q_params, (20, 128))
    reference.validate_hif4_params(k_params, (20, 64))
    reference.validate_hif4_params(v_params, (20, 64))
    print(
        f"v218 verify passed: device={device.type} q_only=True "
        f"candidate_cap={candidate._ATTN_OUTPUT_HEADWISE_MAX_CANDIDATES} "
        f"q_arm={q_state.get('a2_arm')} k_arm={k_state.get('a2_arm')}"
    )


if __name__ == "__main__":
    main()
